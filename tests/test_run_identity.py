"""Run identity: ``core/run_identity.py`` (§12 M3, issue #63).

Four layers:

1. **``RuntimeConfig``/``LaunchConfig`` schema.** Round-trips losslessly,
   rejects unknown/missing/malformed keys (its own launcher-only keys and,
   transitively through ``RunConfig.from_dict``, every embedded protocol
   key), and enforces its own ``schema_version`` pin.
2. **Classification completeness.** Every leaf field of a real
   ``LaunchConfig`` is classified; the exact material/non-material split the
   design constraints pin is asserted field by field; an unclassified leaf
   is a loud error, never a silent default; ``device`` is diffed by kind,
   never by index.
3. **Identity + provenance.** ``run_id`` is filesystem-safe and
   launch-unique; ``write_provenance``/``read_stored_config``/
   ``read_run_record`` round-trip.
4. **Resume/fork resolution.** A resume with an unchanged config proceeds
   unchanged; a resume with a non-material difference proceeds and reports
   it; a resume with any material difference refuses, naming every
   offending field, with **no override** anywhere in this module's surface;
   a fork never diffs, always gets a fresh identity, and never touches the
   parent's files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core.run_identity import (
    LAUNCH_SCHEMA_VERSION,
    LaunchConfig,
    Lineage,
    MaterialConfigDiffError,
    RunRecord,
    RuntimeConfig,
    UnclassifiedFieldError,
    compute_config_hash,
    device_kind,
    diff_launch_configs,
    generate_run_id,
    load_launch_config,
    read_run_record,
    read_stored_config,
    resolve_fork,
    resolve_resume,
    run_root,
    write_provenance,
)
from core.runconfig import MICRO_RUN_CONFIG_PATH, RunConfig

ROOT = Path(__file__).resolve().parent.parent


def _base_raw():
    """The pinned micro config, doc keys stripped, as a fresh mutable dict."""
    raw = json.loads(MICRO_RUN_CONFIG_PATH.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _launch_raw(**launch_overrides):
    """The micro config plus a valid launcher block, with optional overrides."""
    raw = _base_raw()
    raw["num_actors"] = 4
    raw["device"] = "cuda"
    raw["schema_version"] = LAUNCH_SCHEMA_VERSION
    raw["runtime"] = {
        "refresh_poll_interval": 1.0,
        "pacing_poll_interval": 1.0,
        "ceiling_poll_interval": 1.0,
    }
    raw.update(launch_overrides)
    return raw


def _launch_config(**launch_overrides) -> LaunchConfig:
    return LaunchConfig.from_dict(_launch_raw(**launch_overrides))


# ==============================================================================
# 1. RuntimeConfig / LaunchConfig schema
# ==============================================================================


def test_runtime_config_round_trips():
    rt = RuntimeConfig(
        refresh_poll_interval=2.0, pacing_poll_interval=3.0, ceiling_poll_interval=4.0
    )
    assert RuntimeConfig.from_dict(rt.to_dict()) == rt


def test_runtime_config_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="must be > 0"):
        RuntimeConfig(
            refresh_poll_interval=0.0, pacing_poll_interval=1.0, ceiling_poll_interval=1.0
        )


def test_runtime_config_unknown_and_missing_keys_rejected():
    with pytest.raises(ValueError, match="unknown config keys"):
        RuntimeConfig.from_dict(
            {
                "refresh_poll_interval": 1.0,
                "pacing_poll_interval": 1.0,
                "ceiling_poll_interval": 1.0,
                "extra": 1.0,
            }
        )
    with pytest.raises(ValueError, match="missing config keys"):
        RuntimeConfig.from_dict({"refresh_poll_interval": 1.0})


def test_launch_config_round_trips():
    lc = _launch_config()
    assert LaunchConfig.from_dict(lc.to_dict()) == lc
    assert isinstance(lc.run, RunConfig)


def test_launch_config_missing_launcher_key_is_rejected():
    raw = _launch_raw()
    del raw["num_actors"]
    with pytest.raises(ValueError, match=r"missing config keys \['num_actors'\]"):
        LaunchConfig.from_dict(raw)


def test_launch_config_unknown_top_level_key_is_rejected():
    """A stray key falls through to ``RunConfig.from_dict``'s own strict check."""
    raw = _launch_raw()
    raw["num_actorz"] = 4  # typo -- not a launcher key, not a RunConfig key
    with pytest.raises(ValueError, match="unknown config keys"):
        LaunchConfig.from_dict(raw)


def test_launch_config_schema_version_mismatch_rejected():
    raw = _launch_raw(schema_version=LAUNCH_SCHEMA_VERSION + 1)
    with pytest.raises(ValueError, match="schema_version mismatch"):
        LaunchConfig.from_dict(raw)


def test_launch_config_num_actors_must_be_positive():
    with pytest.raises(ValueError, match="num_actors"):
        _launch_config(num_actors=0)


def test_launch_config_device_must_be_non_empty():
    with pytest.raises(ValueError, match="device"):
        _launch_config(device="")


def test_load_launch_config_reads_a_file(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(_launch_raw()))
    lc = load_launch_config(path)
    assert lc == _launch_config()


def test_load_launch_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_launch_config(tmp_path / "nope.json")


def test_documentation_keys_are_ignored():
    raw = _launch_raw()
    raw["_note"] = "a comment"
    assert LaunchConfig.from_dict(raw) == _launch_config()


# ==============================================================================
# 2. Classification completeness + the material/non-material split
# ==============================================================================


def test_a_real_launch_config_is_fully_classified():
    """No UnclassifiedFieldError against a real config diffed with itself."""
    lc = _launch_config()
    diff = diff_launch_configs(lc, lc)
    assert diff.material == {}
    assert diff.non_material == {}


# Excluded from the generic round-trip battery below because this repo has
# no *second* schema-valid value to switch to: "game" (only blokus_duo
# declares GAME_CONFIGS -- see core.runconfig), "training.checkpoint_selection"
# (CHECKPOINT_SELECTIONS == ("final",) only), and "schema_version" (any other
# value is rejected by LaunchConfig.from_dict itself, before a diff is ever
# computed -- exactly the module docstring's documented behavior). Each gets
# its own direct classification-map assertion instead, below.
MATERIAL_FIELDS = (
    "game_config",
    "run_seed",
    "self_play.sims",
    "self_play.k_temp",
    "self_play.dirichlet_eps",
    "self_play.dirichlet_alpha_numerator",
    "self_play.root_noise",
    "training.games",
    "training.learner_steps",
    "training.steps_per_game",
    "training.batch_size",
    "training.replay_window",
    "training.learning_rate",
    "training.warmup_steps",
    "training.cosine_total_steps",
    "training.aux_loss_weight",
    "training.publish_interval",
    "training.checkpoint_count",
    "training.replay_warmup_positions",
    "num_actors",
)

NON_MATERIAL_FIELDS = (
    "name",
    "run_dir",
    "evaluation.agent_form",
    "evaluation.sims",
    "evaluation.root_noise",
    "evaluation.move_selection",
    "evaluation.opponent",
    "evaluation.n_pairs",
    "evaluation.eval_seed",
    "evaluation.min_score_rate",
    "loss_predicates.head_window_steps",
    "loss_predicates.tail_window_steps",
    "loss_predicates.policy_max_ratio",
    "loss_predicates.value_max_ratio",
    "throughput.warmup_games",
    "throughput.measure_games",
    "throughput.projection_sims",
    "throughput.projection_plies_per_game",
    "throughput.min_projected_games_per_hour",
    "runtime.refresh_poll_interval",
    "runtime.pacing_poll_interval",
    "runtime.ceiling_poll_interval",
)


def _flat(lc: LaunchConfig) -> dict:
    from core.run_identity import _flatten

    return _flatten(lc.to_dict())


def _get_dotted(raw: dict, path: str):
    node = raw
    for part in path.split("."):
        node = node[part]
    return node


def _bump(value):
    """Return a value that compares unequal to ``value`` but keeps its type."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + "-changed"
    raise TypeError(f"cannot bump {value!r}")


def _override_training_games(raw: dict) -> None:
    raw["training"]["games"] += 1
    raw["training"]["learner_steps"] = raw["training"]["games"] * raw["training"]["steps_per_game"]


def _override_training_learner_steps(raw: dict) -> None:
    raw["training"]["learner_steps"] += 1
    raw["training"]["games"] = raw["training"]["learner_steps"] // raw["training"]["steps_per_game"]
    raw["training"]["learner_steps"] = raw["training"]["games"] * raw["training"]["steps_per_game"]


def _override_training_steps_per_game(raw: dict) -> None:
    raw["training"]["steps_per_game"] += 1
    raw["training"]["learner_steps"] = raw["training"]["games"] * raw["training"]["steps_per_game"]


# Fields whose only other schema-valid value needs picking deliberately
# (a unit-interval range, a small fixed enum, or a cross-field coherence
# identity in core.runconfig.TrainingConfig) rather than a generic +1/"-changed"
# bump, which would itself fail RunConfig's own validation.
FIELD_OVERRIDES = {
    "game_config": lambda raw: _set_dotted(raw, "game_config", "FULL_CONFIG"),
    "self_play.dirichlet_eps": lambda raw: _set_dotted(raw, "self_play.dirichlet_eps", 0.5),
    "training.games": _override_training_games,
    "training.learner_steps": _override_training_learner_steps,
    "training.steps_per_game": _override_training_steps_per_game,
    "evaluation.move_selection": lambda raw: _set_dotted(
        raw, "evaluation.move_selection", "sample_n"
    ),
    "evaluation.min_score_rate": lambda raw: _set_dotted(raw, "evaluation.min_score_rate", 0.5),
    "loss_predicates.policy_max_ratio": lambda raw: _set_dotted(
        raw, "loss_predicates.policy_max_ratio", 0.5
    ),
    "loss_predicates.value_max_ratio": lambda raw: _set_dotted(
        raw, "loss_predicates.value_max_ratio", 0.5
    ),
}


def _apply_override(raw: dict, path: str) -> None:
    if path in FIELD_OVERRIDES:
        FIELD_OVERRIDES[path](raw)
    else:
        _set_dotted(raw, path, _bump(_get_dotted(raw, path)))


@pytest.mark.parametrize("path", MATERIAL_FIELDS)
def test_changing_a_material_field_is_refused_on_resume(path):
    new_raw = _launch_raw()
    _apply_override(new_raw, path)
    new_lc = LaunchConfig.from_dict(new_raw)
    diff = diff_launch_configs(_launch_config(), new_lc)
    assert path in diff.material, f"{path} was not classified material"
    assert path not in diff.non_material


@pytest.mark.parametrize("path", NON_MATERIAL_FIELDS)
def test_changing_a_non_material_field_proceeds_on_resume(path):
    new_raw = _launch_raw()
    _apply_override(new_raw, path)
    new_lc = LaunchConfig.from_dict(new_raw)
    diff = diff_launch_configs(_launch_config(), new_lc)
    assert path in diff.non_material, f"{path} was not classified non-material"
    assert path not in diff.material


def test_game_field_is_material_documented_judgment_call():
    """No second adapter declares GAME_CONFIGS today (only blokus_duo), so this
    field cannot be exercised through a real from_dict round trip -- asserted
    directly against the classification map instead."""
    from core.run_identity import FIELD_CLASSIFICATION, MATERIAL

    assert FIELD_CLASSIFICATION["game"] == MATERIAL


def test_schema_version_field_is_material_and_enforced_at_parse_time():
    from core.run_identity import FIELD_CLASSIFICATION, MATERIAL

    assert FIELD_CLASSIFICATION["schema_version"] == MATERIAL
    # A stored config's schema_version disagreeing with the live one is
    # rejected by LaunchConfig.from_dict itself, before any diff runs.
    raw = _launch_raw(schema_version=LAUNCH_SCHEMA_VERSION + 1)
    with pytest.raises(ValueError, match="schema_version mismatch"):
        LaunchConfig.from_dict(raw)


def _set_dotted(raw: dict, path: str, value) -> None:
    """Set a dotted path inside a nested-dict config, in place."""
    parts = path.split(".")
    node = raw
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def test_device_kind_extraction():
    assert device_kind("cpu") == "cpu"
    assert device_kind("cuda") == "cuda"
    assert device_kind("cuda:0") == "cuda"
    assert device_kind("CUDA:1") == "cuda"
    assert device_kind("mps") == "mps"


def test_device_is_diffed_by_kind_not_by_index():
    old = _launch_config(device="cuda:0")
    same_kind = _launch_config(device="cuda:1")
    diff = diff_launch_configs(old, same_kind)
    assert "device" not in diff.material
    assert diff.non_material.get("device") == ("cuda:0", "cuda:1")

    different_kind = _launch_config(device="cpu")
    diff2 = diff_launch_configs(old, different_kind)
    assert diff2.material.get("device") == ("cuda:0", "cpu")


def test_unclassified_field_is_a_loud_error(monkeypatch):
    import core.run_identity as ri

    patched = dict(ri.FIELD_CLASSIFICATION)
    del patched["num_actors"]
    monkeypatch.setattr(ri, "FIELD_CLASSIFICATION", patched)

    old = _launch_config(num_actors=4)
    new = _launch_config(num_actors=5)
    with pytest.raises(UnclassifiedFieldError, match="num_actors"):
        diff_launch_configs(old, new)


def test_checkpoint_selection_is_material_documented_judgment_call():
    """Named explicitly: conservative, not one of the issue's enumerated examples."""
    from core.run_identity import FIELD_CLASSIFICATION, MATERIAL

    assert FIELD_CLASSIFICATION["training.checkpoint_selection"] == MATERIAL


def test_evaluation_and_loss_predicates_and_throughput_are_non_material():
    """Documented judgment call: these protocols are read by separate, later
    tooling and have no causal effect on what an M3 run's actors/learner
    produce."""
    from core.run_identity import FIELD_CLASSIFICATION, NON_MATERIAL

    for prefix in ("evaluation.", "loss_predicates.", "throughput."):
        for key, value in FIELD_CLASSIFICATION.items():
            if key.startswith(prefix):
                assert value == NON_MATERIAL, key


# ==============================================================================
# 3. Identity + provenance
# ==============================================================================


_RUN_ID_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def test_generate_run_id_is_filesystem_safe():
    lc = _launch_config()
    run_id = generate_run_id(lc, now=1_700_000_000.123456)
    assert _RUN_ID_SAFE.match(run_id)
    assert "/" not in run_id


def test_generate_run_id_is_deterministic_given_now_and_config():
    lc = _launch_config()
    a = generate_run_id(lc, now=1_700_000_000.0)
    b = generate_run_id(lc, now=1_700_000_000.0)
    assert a == b


def test_generate_run_id_differs_across_timestamps_and_configs():
    lc = _launch_config()
    a = generate_run_id(lc, now=1_700_000_000.0)
    b = generate_run_id(lc, now=1_700_000_000.5)
    assert a != b

    lc2 = _launch_config(num_actors=8)
    c = generate_run_id(lc2, now=1_700_000_000.0)
    assert a != c


def test_compute_config_hash_is_deterministic_and_content_sensitive():
    lc = _launch_config()
    assert compute_config_hash(lc) == compute_config_hash(_launch_config())
    changed = _launch_config(num_actors=lc.num_actors + 1)
    assert compute_config_hash(lc) != compute_config_hash(changed)


def test_run_root_appends_run_id_to_the_family_root():
    lc = _launch_config()
    root = run_root(lc, "some-run-id")
    assert root == Path(lc.run.run_dir) / "some-run-id"


def test_write_and_read_provenance_round_trips(tmp_path):
    lc = _launch_config(**{})
    lc_raw = _launch_raw()
    lc_raw["run_dir"] = str(tmp_path / "runs")
    lc = LaunchConfig.from_dict(lc_raw)
    root = run_root(lc, "run-abc")
    record = RunRecord(
        run_id="run-abc", created_at="2026-01-01T00:00:00Z", entry_condition={"x": 1}
    )

    write_provenance(root, lc, record)

    assert (root / "config.json").is_file()
    assert (root / "run_record.json").is_file()
    assert read_stored_config(root) == lc
    assert read_run_record(root) == record


def test_run_record_with_lineage_round_trips(tmp_path):
    lc_raw = _launch_raw()
    lc_raw["run_dir"] = str(tmp_path / "runs")
    lc = LaunchConfig.from_dict(lc_raw)
    root = run_root(lc, "fork-run")
    lineage = Lineage(
        parent_run_id="parent-run",
        parent_config_hash="deadbeef",
        parent_run_dir=str(tmp_path / "runs" / "parent-run"),
        forked_at="2026-01-01T00:00:00Z",
    )
    record = RunRecord(
        run_id="fork-run",
        created_at="2026-01-01T00:00:00Z",
        entry_condition={"x": 1},
        lineage=lineage,
    )
    write_provenance(root, lc, record)
    reloaded = read_run_record(root)
    assert reloaded.lineage == lineage
    assert reloaded.lineage.imported_weights_version is None


def test_read_stored_config_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_stored_config(tmp_path / "nope")


# ==============================================================================
# 4. Resume / fork resolution
# ==============================================================================


def _write_run(tmp_path, run_id="run-1", **launch_overrides) -> tuple[Path, LaunchConfig]:
    raw = _launch_raw(**launch_overrides)
    raw["run_dir"] = str(tmp_path / "runs")
    lc = LaunchConfig.from_dict(raw)
    root = run_root(lc, run_id)
    write_provenance(
        root, lc, RunRecord(run_id=run_id, created_at="2026-01-01T00:00:00Z", entry_condition={})
    )
    return root, lc


def test_resolve_resume_with_unchanged_config_proceeds(tmp_path):
    root, lc = _write_run(tmp_path)
    resolution = resolve_resume(root, lc)
    assert resolution.run_id == "run-1"
    assert resolution.run_root == root
    assert resolution.effective_config == lc
    assert resolution.non_material_diff == {}


def test_resolve_resume_with_non_material_diff_proceeds_and_reports(tmp_path):
    root, lc = _write_run(tmp_path)
    new_raw = lc.to_dict()
    new_raw["runtime"]["refresh_poll_interval"] = lc.runtime.refresh_poll_interval + 5.0
    new_lc = LaunchConfig.from_dict(new_raw)

    resolution = resolve_resume(root, new_lc)
    assert resolution.run_id == "run-1"
    assert "runtime.refresh_poll_interval" in resolution.non_material_diff
    assert resolution.effective_config == new_lc


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.__setitem__("run_seed", raw["run_seed"] + 1),
        lambda raw: raw.__setitem__("num_actors", raw["num_actors"] + 1),
        lambda raw: raw["self_play"].__setitem__("sims", raw["self_play"]["sims"] + 1),
        lambda raw: raw["training"].__setitem__("batch_size", raw["training"]["batch_size"] + 1),
        lambda raw: raw.__setitem__("device", "cpu"),
    ],
    ids=["run_seed", "num_actors", "sims", "batch_size", "device_kind"],
)
def test_resolve_resume_refuses_on_material_diff_naming_the_field(tmp_path, mutate):
    root, lc = _write_run(tmp_path, device="cuda")
    new_raw = lc.to_dict()
    mutate(new_raw)
    new_lc = LaunchConfig.from_dict(new_raw)

    with pytest.raises(MaterialConfigDiffError) as exc_info:
        resolve_resume(root, new_lc)
    assert exc_info.value.material  # non-empty, names the field(s)


def test_resolve_resume_has_no_override_flag_anywhere_in_its_signature():
    import inspect

    params = inspect.signature(resolve_resume).parameters
    assert "override" not in params
    assert "force" not in params


def test_resolve_resume_missing_run_dir_raises(tmp_path):
    lc = _launch_config()
    with pytest.raises(FileNotFoundError):
        resolve_resume(tmp_path / "nope", lc)


def test_resolve_fork_never_diffs_and_gets_a_fresh_identity(tmp_path):
    root, lc = _write_run(tmp_path, run_id="parent-run")
    forked_raw = lc.to_dict()
    forked_raw["run_seed"] = lc.run.run_seed + 999  # material -- forks may change it freely
    forked_lc = LaunchConfig.from_dict(forked_raw)

    resolution = resolve_fork(root, forked_lc, now=1_700_000_500.0)
    assert resolution.run_id != "parent-run"
    assert resolution.run_root != root
    assert resolution.lineage.parent_run_id == "parent-run"
    assert resolution.lineage.parent_config_hash == compute_config_hash(lc)
    assert resolution.lineage.parent_run_dir == str(root)
    assert resolution.lineage.imported_weights_version is None


def test_resolve_fork_never_writes_the_parent_directory(tmp_path):
    root, lc = _write_run(tmp_path, run_id="parent-run")
    before = (root / "config.json").read_text()
    forked_raw = lc.to_dict()
    forked_raw["run_seed"] = lc.run.run_seed + 1
    resolve_fork(root, LaunchConfig.from_dict(forked_raw), now=1_700_000_500.0)
    after = (root / "config.json").read_text()
    assert before == after
    # No sibling file/dir was added to the parent's directory either.
    assert sorted(p.name for p in root.iterdir()) == ["config.json", "run_record.json"]


def test_resolve_fork_missing_parent_raises(tmp_path):
    lc = _launch_config()
    with pytest.raises(FileNotFoundError):
        resolve_fork(tmp_path / "nope", lc)
