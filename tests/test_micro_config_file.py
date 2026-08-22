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
3. The dependency direction — name resolution is *adapter-declared*: the
   ``GAME_CONFIGS`` mapping lives in ``games/``, and ``core/runconfig.py``
   names no game, so adding a game needs zero ``core/`` edits.
4. Loud rejection — unknown keys, wrong types, out-of-range values, and
   undeclared game-config names all raise rather than silently degrade.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import textwrap
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

import core.runconfig as runconfig
import games.blokus_duo as blokus_duo
from core.runconfig import (
    MICRO_RUN_CONFIG_PATH,
    RunConfig,
    SelfPlayConfig,
    available_games,
    game_configs,
    load_run_config,
    resolve_game_config,
)
from games.blokus_duo.config import FULL_CONFIG, MICRO_CONFIG
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
        # M3 §6.2 pin (PR #76): publish every 200 learner steps, K = 30
        # checkpoints (core.learner.LearnerDriver's own async stop condition,
        # independent of learner_steps above). replay_warmup_positions is an
        # implementation-level (not design-doc-pinned) knob.
        "publish_interval": 200,
        "checkpoint_count": 30,
        "replay_warmup_positions": 128,
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
    assert cfg.training.publish_interval == 200
    assert cfg.training.checkpoint_count == 30
    assert cfg.training.replay_warmup_positions == 128

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
    """The name resolves through the adapter's mapping to the §5.3 ``BlokusConfig``."""
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
    # ...and game names that would be dangerous fed to an unchecked import.
    for hostile in ("blokus_duo.config", "..core", "os", "games", "", "__init__"):
        with pytest.raises(ValueError, match="unknown game"):
            resolve_game_config(hostile, "MICRO_CONFIG")


# --------------------------------------------------------------------------
# 3. The dependency direction: adapters declare, core resolves by convention.
# --------------------------------------------------------------------------

# Every game currently in the repo, plus the Blokus instance-config names. None
# of these may appear in ``core/runconfig.py``'s resolution path.
GAME_TOKENS = ("blokus", "othello", "tictactoe", "connect4", "micro_config", "full_config")

# The one per-game string ``core/runconfig.py`` is allowed to carry: the default
# run-config *file* name (§12 M2.5 names the path). It is data about which run to
# load, not part of resolving a game name to an adapter, and it costs a new game
# nothing — a new game ships its own ``configs/*.json`` and passes the path.
ALLOWED_GAME_MENTIONS = {"blokus_micro.json"}

# The functions that turn the JSON ``game``/``game_config`` strings into an
# adapter object. This is the code that must stay generic.
RESOLUTION_PATH = (
    runconfig.available_games,
    runconfig.game_configs,
    runconfig.resolve_game_config,
    RunConfig.__post_init__,
    RunConfig.resolve_game_config,
)


def _code_strings(source: str) -> Iterator[str]:
    """Yield every identifier and string literal in ``source``, minus docstrings.

    Comments and docstrings are prose — a Blokus *example* in a docstring is
    fine, a Blokus *name* in the code is not — so comments (absent from the AST)
    and docstrings (skipped explicitly) are exempt. What is left is what the code
    actually does.

    Args:
        source: Python source text: a whole module, or one dedented definition.

    Yields:
        Each identifier (names, attributes, imported module names, ...) and each
        non-docstring string constant, as written.
    """
    tree = ast.parse(textwrap.dedent(source))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    for node in ast.walk(tree):
        if id(node) in docstrings:
            continue
        for _, value in ast.iter_fields(node):
            values = value if isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, str):
                    yield item


def test_core_runconfig_declares_no_game():
    """``core/runconfig.py`` carries no per-game name — the repo-layout rule.

    CLAUDE.md: "adding a game touches only ``games/`` + ``configs/``". A registry
    of ``game -> config name`` literals in ``core/`` would break that; the
    resolution path instead reads a mapping the adapter declares.
    """
    source = (ROOT / "core" / "runconfig.py").read_text()
    offenders = {
        text for text in _code_strings(source) if any(t in text.lower() for t in GAME_TOKENS)
    }
    assert offenders == ALLOWED_GAME_MENTIONS


@pytest.mark.parametrize("function", RESOLUTION_PATH, ids=lambda f: f.__qualname__)
def test_game_config_resolution_is_generic(function: Any):
    """No game name appears in the code that resolves ``game``/``game_config``.

    Args:
        function: One function on the name-resolution path.
    """
    offenders = {
        text
        for text in _code_strings(inspect.getsource(function))
        if any(t in text.lower() for t in GAME_TOKENS)
    }
    assert offenders == set()


def test_adapter_declares_its_own_named_configs():
    """The mapping core reads is the adapter's own object, exported by the package."""
    assert game_configs("blokus_duo") is blokus_duo.GAME_CONFIGS
    assert dict(blokus_duo.GAME_CONFIGS) == {
        "FULL_CONFIG": FULL_CONFIG,
        "MICRO_CONFIG": MICRO_CONFIG,
    }
    assert resolve_game_config("blokus_duo", "MICRO_CONFIG") is MICRO_CONFIG
    assert resolve_game_config("blokus_duo", "FULL_CONFIG") is FULL_CONFIG


def test_available_games_is_derived_from_the_games_package():
    """The game allow-list is discovered under ``games/``, not listed in ``core/``."""
    discovered = available_games()
    on_disk = sorted(p.name for p in (ROOT / "games").iterdir() if (p / "__init__.py").is_file())
    assert list(discovered) == on_disk
    assert "blokus_duo" in discovered


def test_game_declaring_no_configs_fails_loudly():
    """An adapter that declares nothing is an error, not a silent empty set.

    Only the Blokus adapter is config-parameterized today (M2.5); the M0/M1.5
    reference games have no named instances, and naming one must say so rather
    than resolve to nothing.
    """
    for game in available_games():
        if hasattr(import_module(f"games.{game}"), "GAME_CONFIGS"):
            continue
        with pytest.raises(ValueError, match="declares no named game configs"):
            resolve_game_config(game, "MICRO_CONFIG")


# --------------------------------------------------------------------------
# 4. Loud rejection of malformed configs.
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
