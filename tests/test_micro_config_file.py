"""config↔doc golden: ``configs/blokus_micro.json`` equals the §5.3/§12 pins.

The M2.5 gate is pre-registered: §12 M2.5 fixes every setting that could move
after seeing results, and ``configs/blokus_micro.json`` is that protocol in
machine-readable form. This file is the tripwire. Every expected scalar below is
**hardcoded from the design doc**, deliberately duplicating the JSON — a test
that re-read the file it is checking would pass for any edit, which is exactly
the failure mode a pre-registered protocol must not have. Changing a number here
without a doc change is the mistake; changing it *with* one is the intended
doc-first flow.

Three layers:

1. The raw-JSON golden — the file's literal contents, key by key.
2. The loader — parses into the typed frozen dataclasses, round-trips through
   ``to_dict``, resolves ``game_config`` to the adapter's ``MICRO_CONFIG``
   object, and ties ``aux_loss_weight`` to
   ``games.blokus_duo.targets.AUX_LOSS_WEIGHT`` (λ_aux is one number with one
   source of truth, §7).
3. Loud rejection — unknown keys, wrong types, out-of-range values, and
   unregistered game-config names all raise rather than silently degrade.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from core.runconfig import (
    MICRO_RUN_CONFIG_PATH,
    RunConfig,
    SelfPlayConfig,
    load_run_config,
    resolve_game_config,
)
from games.blokus_duo.config import MICRO_CONFIG
from games.blokus_duo.targets import AUX_LOSS_WEIGHT

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "blokus_micro.json"

# The complete §12 M2.5 pre-registered protocol, transcribed from the design doc
# (§5.3 for the instance, §12 M2.5 for the gates) — NOT read back from the file.
EXPECTED: dict[str, Any] = {
    "name": "blokus_micro",
    "game": "blokus_duo",
    "game_config": "MICRO_CONFIG",
    # "64 sims/move (fixed — no PCR at M2.5); k_temp = 4; D7 root noise
    # ε = 0.25, α = 10.8/#legal, root-only and self-play-only".
    "self_play": {
        "sims": 64,
        "k_temp": 4,
        "dirichlet_eps": 0.25,
        "dirichlet_alpha_numerator": 10.8,
        "root_noise": True,
    },
    # "2,000 self-play games; batch 32; replay window 3,000 samples; pacing 1
    # learner step per completed game → 2,000 learner steps; base LR 0.02 with
    # 200 warmup steps and cosine decay over 2,000; λ_aux = 0.25. The final
    # end-of-run checkpoint is the one evaluated."
    "training": {
        "games": 2000,
        "learner_steps": 2000,
        "steps_per_game": 1,
        "batch_size": 32,
        "replay_window": 3000,
        "learning_rate": 0.02,
        "warmup_steps": 200,
        "cosine_total_steps": 2000,
        "aux_loss_weight": 0.25,
        "checkpoint_selection": "final",
    },
    # "the rung-7 agent form at 64 eval sims, no Dirichlet noise and
    # deterministic argmax-N move choice. Opponent: rung 1 (uniform random).
    # 100 mirrored pairs (200 games). Evaluation seed = 97531. Win-rate half:
    # total_score_a / (2 × n_pairs) ≥ 0.70."
    "evaluation": {
        "agent_form": "rung7_mcts_policy_value",
        "sims": 64,
        "root_noise": False,
        "move_selection": "argmax_n",
        "opponent": "rung1_uniform_random",
        "n_pairs": 100,
        "eval_seed": 97531,
        "min_score_rate": 0.7,
    },
    # "mean(policy_loss over the last 200 recorded learner steps) ≤ 0.70 ×
    # mean(... first 200)"; the value half at 0.80 over the same windows.
    "loss_predicates": {
        "head_window_steps": 200,
        "tail_window_steps": 200,
        "policy_max_ratio": 0.7,
        "value_max_ratio": 0.8,
    },
    # "first 50 games are warm-up and excluded; the measurement interval is the
    # next 200 games ... at M3's fixed 128 sims and ≈ 35 plies/game ... GO iff
    # games_per_hour_full ≥ 100."
    "throughput": {
        "warmup_games": 50,
        "measure_games": 200,
        "projection_sims": 128,
        "projection_plies_per_game": 35,
        "min_projected_games_per_hour": 100,
    },
    # "Run seed = 2500" (§12 M2.5); the run record's output directory.
    "run_seed": 2500,
    "run_dir": "runs/blokus_micro",
}


def _strip_doc_keys(value: Any) -> Any:
    """Return ``value`` with ``_``-prefixed documentation keys removed.

    Args:
        value: A parsed JSON value.

    Returns:
        The same structure, minus every mapping key beginning with ``_``.
    """
    if isinstance(value, dict):
        return {k: _strip_doc_keys(v) for k, v in value.items() if not k.startswith("_")}
    return value


def _raw() -> dict[str, Any]:
    """Return the parsed config file, documentation keys stripped.

    Returns:
        The file's JSON content without ``_doc``-style keys.
    """
    return _strip_doc_keys(json.loads(CONFIG_PATH.read_text()))


# --------------------------------------------------------------------------
# 1. The raw-JSON golden.
# --------------------------------------------------------------------------


def test_config_file_exists_at_the_pinned_path():
    """§12 M2.5 names ``configs/blokus_micro.json`` explicitly."""
    assert CONFIG_PATH.is_file()
    assert MICRO_RUN_CONFIG_PATH == CONFIG_PATH


def test_config_file_matches_the_doc_pins_exactly():
    """The whole protocol, key by key — no extra keys, no missing keys."""
    assert _raw() == EXPECTED


@pytest.mark.parametrize("section", sorted(k for k, v in EXPECTED.items() if isinstance(v, dict)))
def test_each_section_matches_the_doc_pins(section: str):
    """Per-section assertion, so a failure names the section that drifted.

    Args:
        section: Name of the sub-object under test.
    """
    assert _raw()[section] == EXPECTED[section]


def test_json_types_are_exact():
    """Booleans stay booleans and counts stay ints — the loader rejects coercions."""
    raw = _raw()
    assert raw["self_play"]["root_noise"] is True
    assert raw["evaluation"]["root_noise"] is False
    for key in ("sims", "k_temp"):
        assert isinstance(raw["self_play"][key], int)
    assert isinstance(raw["run_seed"], int)


# --------------------------------------------------------------------------
# 2. The loader: typed, frozen, hashable, round-tripping.
# --------------------------------------------------------------------------


def test_loader_parses_the_pinned_file():
    """Every loaded scalar equals its doc pin, through the typed dataclasses."""
    cfg = load_run_config()

    assert cfg.name == "blokus_micro"
    assert cfg.game == "blokus_duo"
    assert cfg.game_config == "MICRO_CONFIG"
    assert cfg.run_seed == 2500
    assert cfg.run_dir == "runs/blokus_micro"

    assert cfg.self_play == SelfPlayConfig(
        sims=64,
        k_temp=4,
        dirichlet_eps=0.25,
        dirichlet_alpha_numerator=10.8,
        root_noise=True,
    )

    assert cfg.training.games == 2000
    assert cfg.training.learner_steps == 2000
    assert cfg.training.steps_per_game == 1
    assert cfg.training.batch_size == 32
    assert cfg.training.replay_window == 3000
    assert cfg.training.learning_rate == 0.02
    assert cfg.training.warmup_steps == 200
    assert cfg.training.cosine_total_steps == 2000
    assert cfg.training.checkpoint_selection == "final"

    assert cfg.evaluation.agent_form == "rung7_mcts_policy_value"
    assert cfg.evaluation.sims == 64
    assert cfg.evaluation.root_noise is False
    assert cfg.evaluation.move_selection == "argmax_n"
    assert cfg.evaluation.opponent == "rung1_uniform_random"
    assert cfg.evaluation.n_pairs == 100
    assert cfg.evaluation.eval_seed == 97531
    assert cfg.evaluation.min_score_rate == 0.7

    assert cfg.loss_predicates.head_window_steps == 200
    assert cfg.loss_predicates.tail_window_steps == 200
    assert cfg.loss_predicates.policy_max_ratio == 0.7
    assert cfg.loss_predicates.value_max_ratio == 0.8

    assert cfg.throughput.warmup_games == 50
    assert cfg.throughput.measure_games == 200
    assert cfg.throughput.projection_sims == 128
    assert cfg.throughput.projection_plies_per_game == 35
    assert cfg.throughput.min_projected_games_per_hour == 100


def test_loaded_config_round_trips():
    """``to_dict`` reproduces the file (minus doc keys) and reparses to itself."""
    cfg = load_run_config()
    assert cfg.to_dict() == _raw()
    assert RunConfig.from_dict(cfg.to_dict()) == cfg


def test_config_objects_are_frozen_and_hashable():
    """Frozen + hashable so a config can key a cache and cannot drift mid-run."""
    cfg = load_run_config()
    assert hash(cfg) == hash(load_run_config())
    assert hash(cfg.self_play) == hash(load_run_config().self_play)
    with pytest.raises((AttributeError, TypeError)):
        cfg.run_seed = 1  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        cfg.self_play.sims = 128  # type: ignore[misc]


def test_eval_seed_is_independent_of_the_run_seed():
    """§12 M2.5: the fixed paired set must not be coupled to training."""
    cfg = load_run_config()
    assert cfg.evaluation.eval_seed != cfg.run_seed


def test_self_play_root_noise_is_self_play_only():
    """D7 is root-only *and* self-play-only: evaluation runs noiseless."""
    cfg = load_run_config()
    assert cfg.self_play.root_noise is True
    assert cfg.evaluation.root_noise is False


def test_aux_loss_weight_has_one_source_of_truth():
    """λ_aux in the run config must equal the adapter's declared weight (§7)."""
    cfg = load_run_config()
    assert cfg.training.aux_loss_weight == AUX_LOSS_WEIGHT == 0.25


def test_game_config_resolves_to_the_micro_instance():
    """The name resolves through the registry to the §5.3 ``BlokusConfig``."""
    cfg = load_run_config()
    resolved = cfg.resolve_game_config()
    assert resolved is MICRO_CONFIG
    assert resolved.board_size == 5
    assert resolved.start_squares == ((1, 1), (3, 3))
    assert resolved.piece_names == ("I1", "I2", "I3", "V3")


def test_unknown_game_config_names_are_rejected():
    """Resolution is an allow-list lookup, never ``eval``/``getattr`` on input."""
    with pytest.raises(ValueError, match="unknown game_config"):
        resolve_game_config("blokus_duo", "MICRO_CONFIGG")
    with pytest.raises(ValueError, match="unknown game"):
        resolve_game_config("blokus_micro", "MICRO_CONFIG")
    # Names that would be dangerous under a naive getattr.
    with pytest.raises(ValueError, match="unknown game_config"):
        resolve_game_config("blokus_duo", "__loader__")


# --------------------------------------------------------------------------
# 3. Loud rejection of malformed configs.
# --------------------------------------------------------------------------


def _mutated(**overrides: Any) -> dict[str, Any]:
    """Return the pinned config with a deep-copied top level, then overrides.

    Args:
        **overrides: Top-level keys to replace.

    Returns:
        A mutable copy of the parsed config with ``overrides`` applied.
    """
    raw = copy.deepcopy(_raw())
    raw.update(overrides)
    return raw


def test_documentation_keys_are_ignored_not_rejected():
    """``_doc`` and friends are documentation at every level, never config."""
    raw = copy.deepcopy(json.loads(CONFIG_PATH.read_text()))
    assert "_doc" in raw and "_doc" in raw["self_play"]  # the file really has them
    raw["_note"] = "another comment"
    raw["training"]["_note"] = "and one here"
    assert RunConfig.from_dict(raw) == load_run_config()


def test_unknown_key_is_rejected():
    """A typo'd or stale key must fail loudly, not be silently ignored."""
    with pytest.raises(ValueError, match="unknown config keys"):
        RunConfig.from_dict(_mutated(sims=64))
    raw = _mutated()
    raw["training"]["batch_sixe"] = 32
    with pytest.raises(ValueError, match="unknown config keys"):
        RunConfig.from_dict(raw)


def test_missing_key_is_rejected():
    """A protocol scalar may not be omitted — there are no silent defaults."""
    raw = _mutated()
    del raw["throughput"]["projection_sims"]
    with pytest.raises(ValueError, match="missing config keys"):
        RunConfig.from_dict(raw)
    raw = _mutated()
    del raw["run_seed"]
    with pytest.raises(ValueError, match="missing config keys"):
        RunConfig.from_dict(raw)


def test_wrong_type_is_rejected():
    """Types are exact: no string-to-int coercion, no truthy bools."""
    raw = _mutated()
    raw["self_play"]["sims"] = "64"
    with pytest.raises(TypeError, match="expected an int"):
        RunConfig.from_dict(raw)

    raw = _mutated()
    raw["self_play"]["root_noise"] = 1
    with pytest.raises(TypeError, match="expected a bool"):
        RunConfig.from_dict(raw)

    raw = _mutated()
    raw["self_play"]["k_temp"] = True
    with pytest.raises(TypeError, match="expected an int"):
        RunConfig.from_dict(raw)

    raw = _mutated(run_dir=None)
    with pytest.raises(TypeError, match="expected a string"):
        RunConfig.from_dict(raw)

    raw = _mutated(training=["games", 2000])
    with pytest.raises(TypeError, match="expected an object"):
        RunConfig.from_dict(raw)


def test_out_of_range_values_are_rejected():
    """Ranges are enforced eagerly, at parse time, per field."""
    raw = _mutated()
    raw["self_play"]["sims"] = 0
    with pytest.raises(ValueError, match="self_play.sims must be > 0"):
        RunConfig.from_dict(raw)

    raw = _mutated()
    raw["self_play"]["dirichlet_eps"] = 1.5
    with pytest.raises(ValueError, match=r"dirichlet_eps must be in \[0, 1\]"):
        RunConfig.from_dict(raw)

    raw = _mutated()
    raw["evaluation"]["min_score_rate"] = 1.4
    with pytest.raises(ValueError, match=r"min_score_rate must be in \[0, 1\]"):
        RunConfig.from_dict(raw)

    raw = _mutated()
    raw["loss_predicates"]["policy_max_ratio"] = 1.2
    with pytest.raises(ValueError, match=r"policy_max_ratio must be in \[0, 1\]"):
        RunConfig.from_dict(raw)

    raw = _mutated(run_seed=-1)
    with pytest.raises(ValueError, match="run_seed must be >= 0"):
        RunConfig.from_dict(raw)


def test_incoherent_schedule_and_pacing_are_rejected():
    """Cross-field coherence: warmup ≤ cosine span, and the D-pinned pacing."""
    raw = _mutated()
    raw["training"]["warmup_steps"] = 5000
    with pytest.raises(ValueError, match="warmup_steps"):
        RunConfig.from_dict(raw)

    raw = _mutated()
    raw["training"]["steps_per_game"] = 2
    with pytest.raises(ValueError, match="pacing is incoherent"):
        RunConfig.from_dict(raw)


def test_unknown_enumerated_values_are_rejected():
    """Selection rules are enumerated, not free-form strings."""
    raw = _mutated()
    raw["training"]["checkpoint_selection"] = "best"
    with pytest.raises(ValueError, match="checkpoint_selection must be one of"):
        RunConfig.from_dict(raw)

    raw = _mutated()
    raw["evaluation"]["move_selection"] = "greedy"
    with pytest.raises(ValueError, match="move_selection must be one of"):
        RunConfig.from_dict(raw)

    raw = _mutated(game_config="FULL_CONFIGURATION")
    with pytest.raises(ValueError, match="unknown game_config"):
        RunConfig.from_dict(raw)


def test_missing_file_fails_loudly(tmp_path: Path):
    """A mistyped path is an error, never an empty default config.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """
    with pytest.raises(FileNotFoundError):
        load_run_config(tmp_path / "nope.json")
