"""Run configs: one training run's pre-registered protocol, loaded from JSON (§12 M2.5).

The M2.5 gate is pass/fail on *persisted* evidence, so every setting that could
move after observing results is pinned in the design doc (§5.3 for the game
instance, §12 M2.5 for the protocol) and mirrored in ``configs/*.json``.
``tests/test_micro_config_file.py`` golden-checks the file against those pins;
this module is the typed, immutable, hashable in-memory form of it.

Shape (each sub-object is its own frozen dataclass, so a consumer takes exactly
the slice it needs)::

    RunConfig
      ├─ self_play      SelfPlayConfig      (sims, k_temp, D7 constants)
      ├─ training       TrainingConfig      (budget, batch, LR schedule, replay)
      ├─ evaluation     EvaluationConfig    (agent form, opponent, paired set)
      ├─ loss_predicates LossPredicateConfig (§12 M2.5 exit predicates 2–3)
      └─ throughput     ThroughputConfig    (the go/no-go spike scalars)

:class:`SelfPlayConfig` is deliberately *not* micro-specific: it is the shared
sub-shape M3's ``core/run.py::RunConfig`` composes (``tasks/m3/012``), which is
why the sim count is a plain field here — M3's "fixed 128 sims" assertion lives
in its actor layer, not in the config type.

Validation is loud and eager, in two layers: the JSON reader rejects unknown
keys (except ``_``-prefixed documentation keys, which are ignored), missing
keys, and wrong types; the dataclasses' ``__post_init__`` enforces ranges and
cross-field coherence, so a config built in Python is checked exactly as
strictly as one parsed from disk.

``game_config`` is a *name*, resolved through :data:`GAME_CONFIG_REGISTRY` —
an explicit allow-list of ``(game, name) -> (module, attribute)`` — never
``eval`` or ``getattr`` on a caller-supplied string. The import is lazy, so
``core`` stays game-agnostic at import time (nothing in ``core/`` imports
``games/``); the registry is the one place a new game's instance configs are
declared to the runner.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

# Repo-root ``configs/`` — the run-config files live beside the code, not inside
# the installed package (``pyproject`` ships ``core*``/``games*`` only).
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

# The pinned M2.5 micro-Blokus run config (§12 M2.5).
MICRO_RUN_CONFIG_PATH = CONFIG_DIR / "blokus_micro.json"

# Allow-list of instance configs a run may name: ``game -> config name ->
# (module, attribute)``. Adding a game adds an entry here plus its adapter
# package; the values are module/attribute *literals*, so resolution never
# reflects on a string that came from the JSON file.
GAME_CONFIG_REGISTRY: Mapping[str, Mapping[str, tuple[str, str]]] = {
    "blokus_duo": {
        "FULL_CONFIG": ("games.blokus_duo.config", "FULL_CONFIG"),
        "MICRO_CONFIG": ("games.blokus_duo.config", "MICRO_CONFIG"),
    },
}

# Checkpoint-selection rules the runner knows how to apply. §12 M2.5 pins
# "the final end-of-run checkpoint is the one evaluated"; a new rule is a
# doc-first change plus a runner branch, never a free-form string.
CHECKPOINT_SELECTIONS = ("final",)

# Move-selection rules (D10): sample ∝ N, or deterministic argmax N. Evaluation
# pins ``argmax_n`` (§12 M2.5).
MOVE_SELECTIONS = ("argmax_n", "sample_n")


def resolve_game_config(game: str, game_config: str) -> Any:
    """Resolve a ``(game, game_config)`` name pair to the adapter's config object.

    Args:
        game: Adapter package name, e.g. ``"blokus_duo"``.
        game_config: Registered instance-config name, e.g. ``"MICRO_CONFIG"``.

    Returns:
        The adapter's config object (for Blokus, a
        ``games.blokus_duo.config.BlokusConfig``).

    Raises:
        ValueError: If ``game`` or ``game_config`` is not in
            :data:`GAME_CONFIG_REGISTRY`.
    """
    try:
        configs = GAME_CONFIG_REGISTRY[game]
    except KeyError:
        raise ValueError(
            f"unknown game {game!r}; registered games are {sorted(GAME_CONFIG_REGISTRY)}"
        ) from None
    try:
        module_name, attribute = configs[game_config]
    except KeyError:
        raise ValueError(
            f"unknown game_config {game_config!r} for game {game!r}; "
            f"registered configs are {sorted(configs)}"
        ) from None
    return getattr(import_module(module_name), attribute)


def _check_keys(raw: Mapping[str, Any], expected: Iterable[str], where: str) -> None:
    """Reject unknown and missing keys in one config object.

    Keys beginning with ``_`` (``_doc`` and friends) are documentation carried
    in the JSON for the reader's benefit and are ignored, never rejected.

    Args:
        raw: The parsed JSON object.
        expected: The exact key set the object must carry.
        where: Dotted path used in error messages, e.g. ``"training"``.

    Raises:
        ValueError: If any key is unknown or any expected key is missing.
    """
    present = {key for key in raw if not key.startswith("_")}
    expected_set = set(expected)
    unknown = sorted(present - expected_set)
    if unknown:
        raise ValueError(f"{where}: unknown config keys {unknown}")
    missing = sorted(expected_set - present)
    if missing:
        raise ValueError(f"{where}: missing config keys {missing}")


def _mapping(raw: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    """Read a nested config object.

    Args:
        raw: The parent object.
        key: Key to read.
        where: Dotted path of the parent, for error messages.

    Returns:
        The nested mapping.

    Raises:
        TypeError: If the value is not a JSON object.
    """
    value = raw[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"{where}.{key}: expected an object, got {type(value).__name__}")
    return value


def _int(raw: Mapping[str, Any], key: str, where: str) -> int:
    """Read an integer field, rejecting ``bool`` (which reads as a flag).

    Args:
        raw: The config object.
        key: Key to read.
        where: Dotted path of the object, for error messages.

    Returns:
        The integer value.

    Raises:
        TypeError: If the value is not an ``int`` (or is a ``bool``).
    """
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}.{key}: expected an int, got {type(value).__name__}")
    return value


def _float(raw: Mapping[str, Any], key: str, where: str) -> float:
    """Read a float field; JSON integers are accepted and widened.

    Args:
        raw: The config object.
        key: Key to read.
        where: Dotted path of the object, for error messages.

    Returns:
        The value as a ``float``.

    Raises:
        TypeError: If the value is neither ``int`` nor ``float`` (``bool`` is
            rejected: it is an ``int`` subclass and reads as a flag).
    """
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{where}.{key}: expected a number, got {type(value).__name__}")
    return float(value)


def _bool(raw: Mapping[str, Any], key: str, where: str) -> bool:
    """Read a boolean field.

    Args:
        raw: The config object.
        key: Key to read.
        where: Dotted path of the object, for error messages.

    Returns:
        The boolean value.

    Raises:
        TypeError: If the value is not a ``bool`` (``0``/``1`` are not accepted).
    """
    value = raw[key]
    if not isinstance(value, bool):
        raise TypeError(f"{where}.{key}: expected a bool, got {type(value).__name__}")
    return value


def _str(raw: Mapping[str, Any], key: str, where: str) -> str:
    """Read a string field.

    Args:
        raw: The config object.
        key: Key to read.
        where: Dotted path of the object, for error messages.

    Returns:
        The string value.

    Raises:
        TypeError: If the value is not a ``str``.
    """
    value = raw[key]
    if not isinstance(value, str):
        raise TypeError(f"{where}.{key}: expected a string, got {type(value).__name__}")
    return value


def _positive(value: float, name: str) -> None:
    """Assert a scalar is strictly positive.

    Args:
        value: The scalar.
        name: Field name, for the error message.

    Raises:
        ValueError: If ``value <= 0``.
    """
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _non_negative(value: float, name: str) -> None:
    """Assert a scalar is non-negative.

    Args:
        value: The scalar.
        name: Field name, for the error message.

    Raises:
        ValueError: If ``value < 0``.
    """
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def _unit_interval(value: float, name: str) -> None:
    """Assert a scalar lies in ``[0, 1]``.

    Args:
        value: The scalar.
        name: Field name, for the error message.

    Raises:
        ValueError: If ``value`` is outside ``[0, 1]``.
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _one_of(value: str, allowed: tuple[str, ...], name: str) -> None:
    """Assert a string is one of the known enumerated values.

    Args:
        value: The string.
        allowed: The permitted values.
        name: Field name, for the error message.

    Raises:
        ValueError: If ``value`` is not in ``allowed``.
    """
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")


def _non_empty(value: str, name: str) -> None:
    """Assert a string field is not blank.

    Args:
        value: The string.
        name: Field name, for the error message.

    Raises:
        ValueError: If ``value`` is empty or whitespace-only.
    """
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class SelfPlayConfig:
    """Self-play search scalars — the sub-shape M3's ``RunConfig`` also composes.

    Attributes:
        sims: MCTS simulations per move (§12 M2.5 pins 64 for the micro run;
            M3's fixed-128 baseline sets the same field).
        k_temp: D10 temperature ply cutoff — sample ∝ raw N below it, argmax N
            at and after it.
        dirichlet_eps: D7 root-noise mixing weight ε.
        dirichlet_alpha_numerator: D7 α numerator; the per-state concentration
            is ``dirichlet_alpha_numerator / len(legal_moves)``.
        root_noise: Whether root Dirichlet noise is applied at all (D7:
            root-only, self-play-only — evaluation sets this ``False``).
    """

    sims: int
    k_temp: int
    dirichlet_eps: float
    dirichlet_alpha_numerator: float
    root_noise: bool

    def __post_init__(self) -> None:
        """Validate the search scalars.

        Raises:
            ValueError: If ``sims`` is not positive, ``k_temp`` is negative,
                ``dirichlet_eps`` is outside ``[0, 1]``, or
                ``dirichlet_alpha_numerator`` is not positive.
        """
        _positive(self.sims, "self_play.sims")
        _non_negative(self.k_temp, "self_play.k_temp")
        _unit_interval(self.dirichlet_eps, "self_play.dirichlet_eps")
        _positive(self.dirichlet_alpha_numerator, "self_play.dirichlet_alpha_numerator")


@dataclass(frozen=True)
class TrainingConfig:
    """Training budget, optimizer schedule, and replay window (§12 M2.5).

    Attributes:
        games: Self-play games in the run.
        learner_steps: Total learner steps — the pacing consequence of
            ``games × steps_per_game``.
        steps_per_game: Learner steps per completed self-play game.
        batch_size: Learner minibatch size.
        replay_window: Replay ring-buffer capacity, in samples.
        learning_rate: D5 base LR the warmup+cosine schedule multiplies.
        warmup_steps: Linear warmup length, in learner steps.
        cosine_total_steps: Total steps the cosine decay spans.
        aux_loss_weight: λ_aux (§7) — must equal the adapter's declared weight;
            ``tests/test_micro_config_file.py`` pins the two together.
        checkpoint_selection: Which checkpoint the evaluation gate reads; one of
            :data:`CHECKPOINT_SELECTIONS`.
    """

    games: int
    learner_steps: int
    steps_per_game: int
    batch_size: int
    replay_window: int
    learning_rate: float
    warmup_steps: int
    cosine_total_steps: int
    aux_loss_weight: float
    checkpoint_selection: str

    def __post_init__(self) -> None:
        """Validate the budget, the schedule, and their coherence.

        Raises:
            ValueError: If any count is not positive, ``warmup_steps`` is
                negative or exceeds ``cosine_total_steps``, ``learning_rate`` is
                not positive, ``aux_loss_weight`` is negative,
                ``checkpoint_selection`` is unknown, or the pacing identity
                ``learner_steps == games * steps_per_game`` does not hold.
        """
        _positive(self.games, "training.games")
        _positive(self.learner_steps, "training.learner_steps")
        _positive(self.steps_per_game, "training.steps_per_game")
        _positive(self.batch_size, "training.batch_size")
        _positive(self.replay_window, "training.replay_window")
        _positive(self.learning_rate, "training.learning_rate")
        _positive(self.cosine_total_steps, "training.cosine_total_steps")
        _non_negative(self.warmup_steps, "training.warmup_steps")
        _non_negative(self.aux_loss_weight, "training.aux_loss_weight")
        if self.warmup_steps > self.cosine_total_steps:
            raise ValueError(
                f"training.warmup_steps ({self.warmup_steps}) must not exceed "
                f"training.cosine_total_steps ({self.cosine_total_steps})"
            )
        if self.learner_steps != self.games * self.steps_per_game:
            raise ValueError(
                f"training pacing is incoherent: learner_steps ({self.learner_steps}) != "
                f"games ({self.games}) * steps_per_game ({self.steps_per_game})"
            )
        _one_of(self.checkpoint_selection, CHECKPOINT_SELECTIONS, "training.checkpoint_selection")


@dataclass(frozen=True)
class EvaluationConfig:
    """The fixed paired-evaluation protocol behind exit predicate 1 (§12 M2.5).

    Attributes:
        agent_form: Ladder form the trained side plays as, e.g.
            ``"rung7_mcts_policy_value"``.
        sims: MCTS simulations per move during evaluation.
        root_noise: Dirichlet noise during evaluation (pinned ``False``: D7 is
            self-play-only).
        move_selection: Move rule during evaluation; one of
            :data:`MOVE_SELECTIONS`.
        opponent: Frozen ladder opponent, e.g. ``"rung1_uniform_random"``.
        n_pairs: Mirrored pairs played (``2 × n_pairs`` games).
        eval_seed: Evaluation RNG seed — deliberately independent of
            ``RunConfig.run_seed`` so the paired set is not coupled to training.
        min_score_rate: Score-rate floor the trained side must reach, where a
            draw scores 0.5.
    """

    agent_form: str
    sims: int
    root_noise: bool
    move_selection: str
    opponent: str
    n_pairs: int
    eval_seed: int
    min_score_rate: float

    def __post_init__(self) -> None:
        """Validate the evaluation protocol.

        Raises:
            ValueError: If ``agent_form``/``opponent`` are blank, ``sims`` or
                ``n_pairs`` is not positive, ``eval_seed`` is negative,
                ``move_selection`` is unknown, or ``min_score_rate`` is outside
                ``[0, 1]``.
        """
        _non_empty(self.agent_form, "evaluation.agent_form")
        _non_empty(self.opponent, "evaluation.opponent")
        _positive(self.sims, "evaluation.sims")
        _positive(self.n_pairs, "evaluation.n_pairs")
        _non_negative(self.eval_seed, "evaluation.eval_seed")
        _one_of(self.move_selection, MOVE_SELECTIONS, "evaluation.move_selection")
        _unit_interval(self.min_score_rate, "evaluation.min_score_rate")


@dataclass(frozen=True)
class LossPredicateConfig:
    """Exit predicates 2–3: tail-vs-head mean loss ratios (§12 M2.5).

    Attributes:
        head_window_steps: Length of the head window, in recorded learner steps.
        tail_window_steps: Length of the tail window, in recorded learner steps.
        policy_max_ratio: Predicate 2 — ``mean(tail policy loss)`` must be at
            most this multiple of ``mean(head policy loss)``.
        value_max_ratio: Predicate 3 — the same relation for the value loss.
    """

    head_window_steps: int
    tail_window_steps: int
    policy_max_ratio: float
    value_max_ratio: float

    def __post_init__(self) -> None:
        """Validate the predicate windows and ratios.

        Raises:
            ValueError: If either window is not positive, or either ratio falls
                outside ``(0, 1]`` — a ratio above 1 would permit the loss to
                grow and make the predicate vacuous.
        """
        _positive(self.head_window_steps, "loss_predicates.head_window_steps")
        _positive(self.tail_window_steps, "loss_predicates.tail_window_steps")
        for name, ratio in (
            ("loss_predicates.policy_max_ratio", self.policy_max_ratio),
            ("loss_predicates.value_max_ratio", self.value_max_ratio),
        ):
            _positive(ratio, name)
            _unit_interval(ratio, name)


@dataclass(frozen=True)
class ThroughputConfig:
    """The throughput go/no-go spike scalars (§12 M2.5).

    Attributes:
        warmup_games: Leading games excluded from the measurement.
        measure_games: Games in the measurement interval.
        projection_sims: Sim count the full-game projection assumes (M3's
            fixed 128).
        projection_plies_per_game: Plies/game the projection assumes.
        min_projected_games_per_hour: GO floor for the projected full-game
            self-play rate.
    """

    warmup_games: int
    measure_games: int
    projection_sims: int
    projection_plies_per_game: int
    min_projected_games_per_hour: float

    def __post_init__(self) -> None:
        """Validate the spike scalars.

        Raises:
            ValueError: If ``warmup_games`` is negative, or any of the
                measurement/projection scalars is not positive.
        """
        _non_negative(self.warmup_games, "throughput.warmup_games")
        _positive(self.measure_games, "throughput.measure_games")
        _positive(self.projection_sims, "throughput.projection_sims")
        _positive(self.projection_plies_per_game, "throughput.projection_plies_per_game")
        _positive(self.min_projected_games_per_hour, "throughput.min_projected_games_per_hour")


@dataclass(frozen=True)
class RunConfig:
    """One run's complete pre-registered protocol.

    Attributes:
        name: Run name, e.g. ``"blokus_micro"``.
        game: Adapter package name; a key of :data:`GAME_CONFIG_REGISTRY`.
        game_config: Registered instance-config name, resolved by
            :meth:`resolve_game_config`.
        self_play: The shared self-play sub-shape.
        training: Budget, optimizer schedule, replay window.
        evaluation: The fixed paired-evaluation protocol.
        loss_predicates: The tail-vs-head loss predicates.
        throughput: The throughput go/no-go scalars.
        run_seed: The single recorded run seed every stream derives from
            (``core.seeding``).
        run_dir: Output directory for the run record, relative to the repo root.
    """

    name: str
    game: str
    game_config: str
    self_play: SelfPlayConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    loss_predicates: LossPredicateConfig
    throughput: ThroughputConfig
    run_seed: int
    run_dir: str

    def __post_init__(self) -> None:
        """Validate the top-level fields, including the game-config names.

        The name pair is checked against :data:`GAME_CONFIG_REGISTRY` here (a
        dict lookup, no import); :meth:`resolve_game_config` performs the lazy
        import when a caller actually needs the object.

        Raises:
            ValueError: If ``name``/``run_dir`` are blank, ``run_seed`` is
                negative, or ``game``/``game_config`` is not registered.
            TypeError: If any sub-config is not the expected dataclass — a raw
                dict from a hand-built call is rejected rather than silently
                carried.
        """
        _non_empty(self.name, "name")
        _non_empty(self.run_dir, "run_dir")
        _non_negative(self.run_seed, "run_seed")
        for field_name, expected_type in (
            ("self_play", SelfPlayConfig),
            ("training", TrainingConfig),
            ("evaluation", EvaluationConfig),
            ("loss_predicates", LossPredicateConfig),
            ("throughput", ThroughputConfig),
        ):
            value = getattr(self, field_name)
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"{field_name}: expected {expected_type.__name__}, got {type(value).__name__}"
                )
        if self.game not in GAME_CONFIG_REGISTRY:
            raise ValueError(
                f"unknown game {self.game!r}; registered games are {sorted(GAME_CONFIG_REGISTRY)}"
            )
        configs = GAME_CONFIG_REGISTRY[self.game]
        if self.game_config not in configs:
            raise ValueError(
                f"unknown game_config {self.game_config!r} for game {self.game!r}; "
                f"registered configs are {sorted(configs)}"
            )

    def resolve_game_config(self) -> Any:
        """Return the adapter config object this run names.

        Returns:
            The adapter's instance config (for Blokus, a ``BlokusConfig``).

        Raises:
            ValueError: If the name pair is not registered (unreachable for an
                instance that passed ``__post_init__``).
        """
        return resolve_game_config(self.game, self.game_config)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as nested plain dicts.

        The result round-trips: ``RunConfig.from_dict(cfg.to_dict()) == cfg``,
        and it equals the source JSON with the ``_``-prefixed documentation keys
        removed (JSON integers in float fields compare equal to their widened
        values).

        Returns:
            A nested ``dict`` mirroring the JSON layout.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RunConfig:
        """Build a :class:`RunConfig` from a parsed JSON object.

        Args:
            raw: The parsed config object. ``_``-prefixed keys are ignored at
                every level.

        Returns:
            The validated, frozen config.

        Raises:
            ValueError: If any object has unknown or missing keys, or any value
                is out of range / not a registered name.
            TypeError: If ``raw`` (or any nested object) is not a mapping, or a
                value has the wrong JSON type.
        """
        if not isinstance(raw, Mapping):
            raise TypeError(f"run config: expected an object, got {type(raw).__name__}")
        _check_keys(
            raw,
            (
                "name",
                "game",
                "game_config",
                "self_play",
                "training",
                "evaluation",
                "loss_predicates",
                "throughput",
                "run_seed",
                "run_dir",
            ),
            "run config",
        )
        return cls(
            name=_str(raw, "name", "run config"),
            game=_str(raw, "game", "run config"),
            game_config=_str(raw, "game_config", "run config"),
            self_play=_self_play_from_dict(_mapping(raw, "self_play", "run config")),
            training=_training_from_dict(_mapping(raw, "training", "run config")),
            evaluation=_evaluation_from_dict(_mapping(raw, "evaluation", "run config")),
            loss_predicates=_loss_predicates_from_dict(
                _mapping(raw, "loss_predicates", "run config")
            ),
            throughput=_throughput_from_dict(_mapping(raw, "throughput", "run config")),
            run_seed=_int(raw, "run_seed", "run config"),
            run_dir=_str(raw, "run_dir", "run config"),
        )


def _self_play_from_dict(raw: Mapping[str, Any]) -> SelfPlayConfig:
    """Build the self-play sub-config.

    Args:
        raw: The ``self_play`` object.

    Returns:
        The validated :class:`SelfPlayConfig`.

    Raises:
        ValueError: On unknown/missing keys or out-of-range values.
        TypeError: On a wrong value type.
    """
    where = "self_play"
    _check_keys(
        raw, ("sims", "k_temp", "dirichlet_eps", "dirichlet_alpha_numerator", "root_noise"), where
    )
    return SelfPlayConfig(
        sims=_int(raw, "sims", where),
        k_temp=_int(raw, "k_temp", where),
        dirichlet_eps=_float(raw, "dirichlet_eps", where),
        dirichlet_alpha_numerator=_float(raw, "dirichlet_alpha_numerator", where),
        root_noise=_bool(raw, "root_noise", where),
    )


def _training_from_dict(raw: Mapping[str, Any]) -> TrainingConfig:
    """Build the training sub-config.

    Args:
        raw: The ``training`` object.

    Returns:
        The validated :class:`TrainingConfig`.

    Raises:
        ValueError: On unknown/missing keys or out-of-range values.
        TypeError: On a wrong value type.
    """
    where = "training"
    _check_keys(
        raw,
        (
            "games",
            "learner_steps",
            "steps_per_game",
            "batch_size",
            "replay_window",
            "learning_rate",
            "warmup_steps",
            "cosine_total_steps",
            "aux_loss_weight",
            "checkpoint_selection",
        ),
        where,
    )
    return TrainingConfig(
        games=_int(raw, "games", where),
        learner_steps=_int(raw, "learner_steps", where),
        steps_per_game=_int(raw, "steps_per_game", where),
        batch_size=_int(raw, "batch_size", where),
        replay_window=_int(raw, "replay_window", where),
        learning_rate=_float(raw, "learning_rate", where),
        warmup_steps=_int(raw, "warmup_steps", where),
        cosine_total_steps=_int(raw, "cosine_total_steps", where),
        aux_loss_weight=_float(raw, "aux_loss_weight", where),
        checkpoint_selection=_str(raw, "checkpoint_selection", where),
    )


def _evaluation_from_dict(raw: Mapping[str, Any]) -> EvaluationConfig:
    """Build the evaluation sub-config.

    Args:
        raw: The ``evaluation`` object.

    Returns:
        The validated :class:`EvaluationConfig`.

    Raises:
        ValueError: On unknown/missing keys or out-of-range values.
        TypeError: On a wrong value type.
    """
    where = "evaluation"
    _check_keys(
        raw,
        (
            "agent_form",
            "sims",
            "root_noise",
            "move_selection",
            "opponent",
            "n_pairs",
            "eval_seed",
            "min_score_rate",
        ),
        where,
    )
    return EvaluationConfig(
        agent_form=_str(raw, "agent_form", where),
        sims=_int(raw, "sims", where),
        root_noise=_bool(raw, "root_noise", where),
        move_selection=_str(raw, "move_selection", where),
        opponent=_str(raw, "opponent", where),
        n_pairs=_int(raw, "n_pairs", where),
        eval_seed=_int(raw, "eval_seed", where),
        min_score_rate=_float(raw, "min_score_rate", where),
    )


def _loss_predicates_from_dict(raw: Mapping[str, Any]) -> LossPredicateConfig:
    """Build the loss-predicate sub-config.

    Args:
        raw: The ``loss_predicates`` object.

    Returns:
        The validated :class:`LossPredicateConfig`.

    Raises:
        ValueError: On unknown/missing keys or out-of-range values.
        TypeError: On a wrong value type.
    """
    where = "loss_predicates"
    _check_keys(
        raw,
        ("head_window_steps", "tail_window_steps", "policy_max_ratio", "value_max_ratio"),
        where,
    )
    return LossPredicateConfig(
        head_window_steps=_int(raw, "head_window_steps", where),
        tail_window_steps=_int(raw, "tail_window_steps", where),
        policy_max_ratio=_float(raw, "policy_max_ratio", where),
        value_max_ratio=_float(raw, "value_max_ratio", where),
    )


def _throughput_from_dict(raw: Mapping[str, Any]) -> ThroughputConfig:
    """Build the throughput sub-config.

    Args:
        raw: The ``throughput`` object.

    Returns:
        The validated :class:`ThroughputConfig`.

    Raises:
        ValueError: On unknown/missing keys or out-of-range values.
        TypeError: On a wrong value type.
    """
    where = "throughput"
    _check_keys(
        raw,
        (
            "warmup_games",
            "measure_games",
            "projection_sims",
            "projection_plies_per_game",
            "min_projected_games_per_hour",
        ),
        where,
    )
    return ThroughputConfig(
        warmup_games=_int(raw, "warmup_games", where),
        measure_games=_int(raw, "measure_games", where),
        projection_sims=_int(raw, "projection_sims", where),
        projection_plies_per_game=_int(raw, "projection_plies_per_game", where),
        min_projected_games_per_hour=_float(raw, "min_projected_games_per_hour", where),
    )


def load_run_config(path: Path | str = MICRO_RUN_CONFIG_PATH) -> RunConfig:
    """Load and validate a run config from a JSON file.

    Args:
        path: Path to the JSON file; defaults to the pinned M2.5 micro-Blokus
            config, :data:`MICRO_RUN_CONFIG_PATH`.

    Returns:
        The validated, frozen :class:`RunConfig`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If any object has unknown or missing keys, or any value is
            out of range / not a registered name.
        TypeError: If the document (or any nested object) is not an object, or
            a value has the wrong JSON type.
    """
    return RunConfig.from_dict(json.loads(Path(path).read_text()))
