"""The M4 eval harness's own launch config, provenance, and scheduling arithmetic.

Design doc §9; tasks/m4/009 subtask 9.1 ("the config and correctness spine").
Joins ``core.runconfig``/``core.run_identity`` (the M2.5/M3 run-config
machinery) as the M4 harness's own launch-time layer, over the same
JSON-parse discipline and the same "no override, refuse and name the fields"
resume/relaunch pattern, but for a different kind of process: not a training
run, but a long-lived reader/player that watches one already-launched run
and appends to ``core.eval_store``'s per-cell record store.

This module owns six things (the report/CLI integration is a later subtask,
9.3):

* :class:`EvalConfig` -- one eval harness launch's full protocol, loaded from
  JSON with the same loud, eager validation ``core.runconfig.RunConfig`` uses.
* Launch provenance + the relaunch guard (:func:`write_eval_provenance`,
  :func:`resolve_eval_launch`) -- ``<run_dir>/eval/config.json`` records the
  config verbatim, stamped with the current ``(protocol_version,
  protocol_fingerprint)`` from ``core.eval_protocol``; a later relaunch
  refuses on any difference in either the config's own material fields or
  the protocol stamp, naming every offending field, with **no override
  flag** -- exactly ``core.run_identity``'s ``--resume`` discipline, applied
  to the eval namespace instead of the training run.
* Membership arithmetic (:func:`schedulable_versions`) -- the schedulable
  checkpoint versions are exactly ``1..K`` as their immutable
  ``ckpt-<version>.pt`` files appear; version 0, the rolling ``resume.pt``
  snapshot, and the ``latest`` pointer can never enter this set, structurally
  (``core.checkpoint.list_published_versions``' own glob never matches them).
* Cell-scheduling arithmetic (:func:`required_cell_ids`,
  :func:`cell_seed`) -- one member's full required cell-id set (forms x the
  game's declared network-free rungs, plus rung-8 historical matchups for
  form 7), and the per-cell seed each cell's games are played under.
* The watch/resume/catch-up loop itself (:func:`run_watch_loop`, subtask
  9.2) -- the single long-lived process that turns the arithmetic above into
  actual played games: poll the watched run for schedulable members, register
  each one's required cells in the task-5 manifest, and play every cell not
  yet complete (or resume a partial one at its next missing pair) through
  ``core.runner.play_pairs``, idempotently, across arbitrarily many kills and
  relaunches.
* The eval-lag observable (:func:`eval_lag`, subtask 9.2) -- how far behind
  the harness's own scoring is relative to the watched run's newest published
  member, for 9.3's reporting and the M5.5 liveness check (design doc §9 pin
  11).

**Game-generic by construction:** this module never imports ``games.*`` and
never names a game-specific agent or balancer. Every function that needs a
game's ladder takes a :class:`core.eval_profile.EvalProfile` as a plain
argument, resolved by the caller (the script layer, via ``games.registry`` --
mirroring exactly how ``games.registry.build_game_factory`` resolves a
``Game`` for ``core.ipc.launch_run`` without ``core/`` ever importing
``games/``).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.artifact_fingerprint import build_fingerprint
from core.checkpoint import list_published_versions, published_checkpoint_path
from core.eval_agents import (
    NetworkPolicyAgent,
    SearchAgent,
    historical_opponent_factory,
    historical_opponents,
    load_eval_network,
)
from core.eval_profile import EvalProfile
from core.eval_protocol import PROTOCOL_VERSION, protocol_fingerprint
from core.eval_stats import _validate_admissible_B
from core.eval_store import (
    CellId,
    append_pair_record,
    build_cell_id,
    build_header,
    cell_path,
    complete_cell,
    eval_dir,
    is_cell_complete,
    load_snapshot,
    open_cell_for_write,
    pair_result_to_record,
    parse_cell_id,
    register_member,
)
from core.game import Game
from core.mcts import Evaluator
from core.replay_shard import _atomic_write_json
from core.run_identity import read_run_record, read_stored_config
from core.runconfig import _check_keys, _int, _non_empty, _positive, _str
from core.runner import AgentFactory, play_pairs
from core.seeding import PURPOSE_EVAL, derive_seed

# --- EvalConfig ----------------------------------------------------------------

#: This module's own config schema version. A mismatch is rejected at parse
#: time (``EvalConfig.__post_init__``), before any other field is even read --
#: mirrors ``core.run_identity.LAUNCH_SCHEMA_VERSION``'s role exactly.
EVAL_CONFIG_SCHEMA_VERSION = 1

#: The only checkpoint-parameterized agent forms this harness's ladder ever
#: names (design doc §9's cell semantics) -- ``core.eval_agents``' own
#: constructors (``rung5_agent_factory``, ``rung_search_agent_factory``) are
#: the sole source of agents at these ids, so a ``forms`` value outside this
#: set names an agent nothing in the codebase can ever build.
VALID_FORMS = (5, 6, 7)

_EVAL_CONFIG_KEYS = (
    "run_dir",
    "pairs_per_cell",
    "eval_sims",
    "rung8_lag_divisor",
    "rung8_earliest_version",
    "forms",
    "eval_seed",
    "device",
    "bootstrap_b",
    "schema_version",
)


def _int_tuple(raw: Mapping[str, Any], key: str, where: str) -> tuple[int, ...]:
    """Read a JSON array of ints as a tuple (the ``forms`` field's shape).

    Args:
        raw: The config object.
        key: Key to read.
        where: Dotted path of the object, for error messages.

    Returns:
        The array's values, in file order, as a tuple of ``int``.

    Raises:
        TypeError: If the value is not a JSON array, or any element is not an
            ``int`` (``bool`` rejected, matching ``core.runconfig._int``'s own
            rule: a flag is never an integer field).
    """
    value = raw[key]
    if not isinstance(value, list):
        raise TypeError(f"{where}.{key}: expected an array, got {type(value).__name__}")
    out: list[int] = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{where}.{key}[{i}]: expected an int, got {type(item).__name__}")
        out.append(item)
    return tuple(out)


@dataclass(frozen=True)
class EvalConfig:
    """One eval harness launch's full, pre-registered protocol (design doc §9).

    Mirrors ``core.runconfig.RunConfig``'s JSON-parse discipline exactly:
    ``__post_init__`` enforces every range/coherence rule a config built
    directly in Python is checked against too, so :func:`load_eval_config`
    and a hand-built instance are validated identically.

    Every field below except ``run_dir``/``schema_version`` is "material" for
    the relaunch guard (:func:`resolve_eval_launch`) -- there is no
    non-material carve-out here the way
    ``core.run_identity.FIELD_CLASSIFICATION`` has for the training run's
    paths and poll cadences, because nothing left in this config is a mere
    path or cadence: every one of these values changes which games get
    played or how they are scored.

    Attributes:
        run_dir: The watched production run's root directory --
            ``core.run_identity`` provenance lives there (``config.json``,
            ``run_record.json``), and this harness's own artifacts live under
            its ``eval/`` subdirectory (``core.eval_store.eval_dir``).
        pairs_per_cell: Mirrored pairs per (candidate, rung, opponent) cell.
            Pinned production value: ``core.eval_protocol.PAIRS_PER_CELL``.
        eval_sims: The rung 6/7 eval search-form simulation budget *S*.
            Pinned production value: ``core.eval_protocol.EVAL_SIMS``.
        rung8_lag_divisor: The rung-8 historical-opponent rule's lag divisor
            (``core.eval_agents.historical_opponents``' ``ceil(K / this)``
            term). Pinned production value:
            ``core.eval_protocol.RUNG8_LAG_DIVISOR``.
        rung8_earliest_version: The rung-8 rule's always-included earliest
            opponent version. Pinned production value:
            ``core.eval_protocol.RUNG8_EARLIEST_VERSION``.
        forms: The checkpoint-parameterized agent forms to instantiate per
            member -- a non-empty, deduplicated subset of :data:`VALID_FORMS`
            (normalized ascending by :meth:`__post_init__`). Production plays
            all three: ``(5, 6, 7)``.
        eval_seed: The harness's own root seed, deliberately independent of
            the watched run's ``run_seed`` (the M2.5
            ``core.runconfig.EvaluationConfig.eval_seed`` precedent), so the
            evaluation evidence is never coupled to how training happened to
            be seeded. Every per-cell seed derives from this one value
            (:func:`cell_seed`).
        device: Torch device string for eval inference. ``"cpu"`` in
            production (the concurrent harness must not starve the learner's
            GPU use -- M5 owns batched inference); ``"cuda"`` is allowed for
            an offline re-scoring pass.
        bootstrap_b: The task-7 bootstrap replicate count *B*. Must be
            admissible under the pinned order-statistic rank rule
            (``B % 40 == 39`` -- see ``core.eval_stats._validate_admissible_B``).
            Pinned production value:
            ``core.eval_protocol.BOOTSTRAP_B_PRODUCTION`` (1,999).
        schema_version: This config's own schema version
            (:data:`EVAL_CONFIG_SCHEMA_VERSION`).
    """

    run_dir: str
    pairs_per_cell: int
    eval_sims: int
    rung8_lag_divisor: int
    rung8_earliest_version: int
    forms: tuple[int, ...]
    eval_seed: int
    device: str
    bootstrap_b: int
    schema_version: int = EVAL_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate every scalar and normalize ``forms`` to a sorted tuple.

        Raises:
            ValueError: If ``run_dir``/``device`` is blank; any of
                ``pairs_per_cell``/``eval_sims``/``rung8_lag_divisor``/
                ``rung8_earliest_version`` is not positive; ``forms`` is
                empty, contains a duplicate, or names a form outside
                :data:`VALID_FORMS`; ``bootstrap_b`` is not admissible; or
                ``schema_version`` disagrees with
                :data:`EVAL_CONFIG_SCHEMA_VERSION`.
            TypeError: If ``forms`` is not a sequence of ``int``.
        """
        _non_empty(self.run_dir, "run_dir")
        _positive(self.pairs_per_cell, "pairs_per_cell")
        _positive(self.eval_sims, "eval_sims")
        _positive(self.rung8_lag_divisor, "rung8_lag_divisor")
        _positive(self.rung8_earliest_version, "rung8_earliest_version")
        _non_empty(self.device, "device")

        if any(isinstance(f, bool) or not isinstance(f, int) for f in self.forms):
            raise TypeError(f"forms must be a sequence of int, got {self.forms!r}")
        if not self.forms:
            raise ValueError("forms must be non-empty")
        if len(set(self.forms)) != len(self.forms):
            raise ValueError(f"forms must not contain duplicates, got {tuple(self.forms)!r}")
        unknown_forms = sorted(set(self.forms) - set(VALID_FORMS))
        if unknown_forms:
            raise ValueError(
                f"forms contains unrecognized form id(s) {unknown_forms}; "
                f"the only defined agent forms are {list(VALID_FORMS)}"
            )
        object.__setattr__(self, "forms", tuple(sorted(self.forms)))

        _validate_admissible_B(self.bootstrap_b)

        if self.schema_version != EVAL_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"eval config schema_version mismatch: stored={self.schema_version!r} "
                f"live={EVAL_CONFIG_SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return this config as a flat, JSON-serializable dict."""
        return {
            "run_dir": self.run_dir,
            "pairs_per_cell": self.pairs_per_cell,
            "eval_sims": self.eval_sims,
            "rung8_lag_divisor": self.rung8_lag_divisor,
            "rung8_earliest_version": self.rung8_earliest_version,
            "forms": list(self.forms),
            "eval_seed": self.eval_seed,
            "device": self.device,
            "bootstrap_b": self.bootstrap_b,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EvalConfig:
        """Build an :class:`EvalConfig` from a parsed JSON object.

        Args:
            raw: The parsed config object. ``_``-prefixed keys (``_doc`` and
                friends) are documentation and are ignored, never rejected.

        Returns:
            The validated, frozen config.

        Raises:
            ValueError: On unknown/missing keys or any range/coherence
                violation (see :meth:`__post_init__`).
            TypeError: On a wrong value type.
        """
        if not isinstance(raw, Mapping):
            raise TypeError(f"eval config: expected an object, got {type(raw).__name__}")
        where = "eval config"
        _check_keys(raw, _EVAL_CONFIG_KEYS, where)
        return cls(
            run_dir=_str(raw, "run_dir", where),
            pairs_per_cell=_int(raw, "pairs_per_cell", where),
            eval_sims=_int(raw, "eval_sims", where),
            rung8_lag_divisor=_int(raw, "rung8_lag_divisor", where),
            rung8_earliest_version=_int(raw, "rung8_earliest_version", where),
            forms=_int_tuple(raw, "forms", where),
            eval_seed=_int(raw, "eval_seed", where),
            device=_str(raw, "device", where),
            bootstrap_b=_int(raw, "bootstrap_b", where),
            schema_version=_int(raw, "schema_version", where),
        )

    def as_eval_config_snapshot(self) -> dict[str, int]:
        """Return this config's pinned-value subset, for a cell header stamp.

        Matches ``core.eval_protocol.eval_config_snapshot()``'s shape
        exactly, but built from *this config's own* fields rather than from
        the module's live constants -- the mirror
        ``core.eval_store.CellHeader.eval_config`` is meant to catch: a
        resumed cell's stored snapshot is compared against the caller's
        *current* config, independently of whether ``core.eval_protocol``'s
        own constants ever changed at all.

        Returns:
            ``{"pairs_per_cell", "eval_sims", "rung8_lag_divisor",
            "rung8_earliest_version"}`` read off this config.
        """
        return {
            "pairs_per_cell": self.pairs_per_cell,
            "eval_sims": self.eval_sims,
            "rung8_lag_divisor": self.rung8_lag_divisor,
            "rung8_earliest_version": self.rung8_earliest_version,
        }


def load_eval_config(path: Path | str) -> EvalConfig:
    """Load and validate an eval config from a JSON file.

    Args:
        path: Path to the JSON file (e.g. ``configs/m4_eval.json``).

    Returns:
        The validated, frozen :class:`EvalConfig`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If any key is unknown/missing or any value is out of
            range.
        TypeError: If the document is not an object, or a value has the
            wrong JSON type.
    """
    return EvalConfig.from_dict(json.loads(Path(path).read_text()))


# --- launch provenance -----------------------------------------------------------

_EVAL_CONFIG_FILENAME = "config.json"
_PROVENANCE_ONLY_KEYS = ("protocol_version", "protocol_fingerprint")


@dataclass(frozen=True)
class StoredEvalConfig:
    """An :class:`EvalConfig` plus the protocol stamp recorded at first launch.

    The exact shape ``<run_dir>/eval/config.json`` is written and read as:
    the launch config's own fields, verbatim, plus the two values that let
    :func:`resolve_eval_launch` catch a code-level convention drift touching
    no :class:`EvalConfig` field at all -- mirrors
    ``core.eval_store.CellHeader``'s own "two independent drift checks"
    design (that module's docstring), one level up, at the whole-launch
    grain instead of the per-cell one.

    Attributes:
        config: The recorded launch config.
        protocol_version: ``core.eval_protocol.PROTOCOL_VERSION`` at launch time.
        protocol_fingerprint: ``core.eval_protocol.protocol_fingerprint()`` at
            launch time.
    """

    config: EvalConfig
    protocol_version: int
    protocol_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Return the flat, JSON-shaped dict this record round-trips through."""
        payload = dict(self.config.to_dict())
        payload["protocol_version"] = self.protocol_version
        payload["protocol_fingerprint"] = self.protocol_fingerprint
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> StoredEvalConfig:
        """Build a :class:`StoredEvalConfig` from a parsed JSON object.

        Args:
            raw: The parsed ``eval/config.json`` content.

        Returns:
            The reconstructed record.

        Raises:
            ValueError: If a provenance-only key is missing, or (via
                :meth:`EvalConfig.from_dict`) the embedded config has
                unknown/missing keys or an out-of-range value.
            TypeError: On a wrong value type, from either layer.
        """
        if not isinstance(raw, Mapping):
            raise TypeError(f"stored eval config: expected an object, got {type(raw).__name__}")
        present = {k for k in raw if not k.startswith("_")}
        missing = sorted(set(_PROVENANCE_ONLY_KEYS) - present)
        if missing:
            raise ValueError(f"stored eval config: missing provenance key(s) {missing}")
        inner_raw = {
            k: v for k, v in raw.items() if k.startswith("_") or k not in _PROVENANCE_ONLY_KEYS
        }
        config = EvalConfig.from_dict(inner_raw)
        where = "stored eval config"
        return cls(
            config=config,
            protocol_version=_int(raw, "protocol_version", where),
            protocol_fingerprint=_str(raw, "protocol_fingerprint", where),
        )


def eval_config_path(run_dir: Path | str) -> Path:
    """Return this harness's own provenance file path.

    Args:
        run_dir: The watched run's root directory.

    Returns:
        ``core.eval_store.eval_dir(run_dir) / "config.json"``.
    """
    return eval_dir(run_dir) / _EVAL_CONFIG_FILENAME


def write_eval_provenance(run_dir: Path | str, config: EvalConfig) -> StoredEvalConfig:
    """Durably write this harness's provenance file, stamping the live protocol.

    Called exactly once per eval namespace, at first launch only -- never on
    a later relaunch, which must never mutate the recorded config (mirrors
    ``core.run_identity.write_provenance``'s own rule for the training run).

    Args:
        run_dir: The watched run's root directory (``eval/`` created if
            missing).
        config: The config to record verbatim.

    Returns:
        The stamped record actually written.
    """
    stored = StoredEvalConfig(
        config=config,
        protocol_version=PROTOCOL_VERSION,
        protocol_fingerprint=protocol_fingerprint(),
    )
    path = eval_config_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, stored.to_dict())
    return stored


def read_eval_provenance(run_dir: Path | str) -> StoredEvalConfig:
    """Read this harness's recorded ``eval/config.json``.

    Args:
        run_dir: The watched run's root directory.

    Returns:
        The stored record.

    Raises:
        FileNotFoundError: If no ``eval/config.json`` exists at ``run_dir``.
    """
    path = eval_config_path(run_dir)
    return StoredEvalConfig.from_dict(json.loads(path.read_text()))


# --- relaunch guard: no override, refuse and name every differing field --------

#: Exactly the fields the design doc's relaunch guard names as material
#: (task 9's "Details": pairs-per-cell, S, forms, the rung-8 rule, eval_seed,
#: device, B). ``run_dir`` identifies *which* stored config this is being
#: compared against rather than a value that can itself drift within one
#: comparison, and ``schema_version`` a mismatch there is already rejected
#: earlier, at parse time, by :meth:`EvalConfig.__post_init__` -- neither is
#: compared here.
_MATERIAL_FIELDS = (
    "pairs_per_cell",
    "eval_sims",
    "rung8_lag_divisor",
    "rung8_earliest_version",
    "forms",
    "eval_seed",
    "device",
    "bootstrap_b",
)


class EvalRelaunchRefusedError(ValueError):
    """Raised by :func:`resolve_eval_launch` when a relaunch would silently
    change the running eval protocol.

    **No override flag exists** (design constraint, mirroring
    ``core.run_identity.MaterialConfigDiffError``): a deliberate protocol
    change is a new eval namespace with recorded lineage, never mixed
    evidence under one store.

    Attributes:
        config_diff: Differing material :class:`EvalConfig` fields, name ->
            ``(stored, new)``.
        protocol_diff: Differing protocol-stamp fields, name ->
            ``(stored, current)``.
    """

    def __init__(
        self,
        config_diff: Mapping[str, tuple[Any, Any]],
        protocol_diff: Mapping[str, tuple[Any, Any]],
    ) -> None:
        self.config_diff = dict(config_diff)
        self.protocol_diff = dict(protocol_diff)
        fields = ", ".join(sorted({*self.config_diff, *self.protocol_diff}))
        super().__init__(f"eval relaunch refused: field(s) differ: {fields}")


def _diff_eval_configs(stored: EvalConfig, new: EvalConfig) -> dict[str, tuple[Any, Any]]:
    """Return every material :class:`EvalConfig` field on which ``stored``/``new`` differ.

    Args:
        stored: The previously recorded config.
        new: The config freshly passed on this invocation.

    Returns:
        ``{field_name: (stored_value, new_value)}`` over exactly
        :data:`_MATERIAL_FIELDS`.
    """
    diff: dict[str, tuple[Any, Any]] = {}
    for name in _MATERIAL_FIELDS:
        old_value = getattr(stored, name)
        new_value = getattr(new, name)
        if old_value != new_value:
            diff[name] = (old_value, new_value)
    return diff


def resolve_eval_launch(run_dir: Path | str, config: EvalConfig) -> EvalConfig:
    """Resolve one eval-harness launch against any already-recorded provenance.

    A brand-new eval namespace (no ``eval/config.json`` yet) simply records
    ``config`` (:func:`write_eval_provenance`) and returns it. An existing
    namespace instead compares every material :class:`EvalConfig` field
    against the stored one, and separately recomputes the current
    ``core.eval_protocol`` registry stamp and compares it against the one
    recorded at first launch -- refusing on *any* difference, of either
    kind, naming every offending field. **No override flag exists**: a
    deliberate protocol change is a new eval namespace with recorded
    lineage, never mixed evidence under one store.

    Args:
        run_dir: The watched run's root directory.
        config: The config freshly passed on this invocation.

    Returns:
        The config to actually run with -- ``config`` itself; on a
        successful relaunch this is, by construction, field-for-field equal
        to the stored one already (over :data:`_MATERIAL_FIELDS`).

    Raises:
        EvalRelaunchRefusedError: If any material :class:`EvalConfig` field
            or protocol stamp differs from what was already recorded.
    """
    path = eval_config_path(run_dir)
    if not path.exists():
        write_eval_provenance(run_dir, config)
        return config

    stored = read_eval_provenance(run_dir)
    config_diff = _diff_eval_configs(stored.config, config)

    current_fingerprint = protocol_fingerprint()
    protocol_diff: dict[str, tuple[Any, Any]] = {}
    if stored.protocol_version != PROTOCOL_VERSION:
        protocol_diff["protocol_version"] = (stored.protocol_version, PROTOCOL_VERSION)
    if stored.protocol_fingerprint != current_fingerprint:
        protocol_diff["protocol_fingerprint"] = (stored.protocol_fingerprint, current_fingerprint)

    if config_diff or protocol_diff:
        raise EvalRelaunchRefusedError(config_diff, protocol_diff)
    return config


# --- membership arithmetic: exactly versions 1..K, on disk, right now ----------

#: The run's checkpoint directory, relative to its root -- mirrors the
#: literal ``scripts/run_selfplay.py`` launches the M3 learner against
#: (``ckpt_dir=root / "checkpoints"``) and ``core/acceptance.py`` reads back.
_CHECKPOINTS_DIRNAME = "checkpoints"


def checkpoint_dir(run_dir: Path | str) -> Path:
    """Return the watched run's checkpoint directory.

    Args:
        run_dir: The watched run's root directory.

    Returns:
        ``<run_dir>/checkpoints`` -- ``core.checkpoint``'s namespace.
    """
    return Path(run_dir) / _CHECKPOINTS_DIRNAME


def watched_k_total(run_dir: Path | str) -> int:
    """Return the watched run's fixed, total checkpoint count *K*.

    Args:
        run_dir: The watched run's root directory (must carry
            ``core.run_identity``'s recorded ``config.json``).

    Returns:
        ``core.run_identity.read_stored_config(run_dir).run.training.checkpoint_count``
        -- the same read ``core.eval_stats.build_verdict`` uses for
        ``k_target``.

    Raises:
        FileNotFoundError: If ``run_dir`` has no recorded ``config.json``.
    """
    return read_stored_config(run_dir).run.training.checkpoint_count


def schedulable_versions(run_dir: Path | str, k_total: int) -> tuple[int, ...]:
    """Return the member checkpoint versions currently eligible for scheduling.

    Exactly versions ``1..k_total`` whose immutable ``ckpt-<version>.pt``
    file is present on disk right now, ascending. Version ``0`` (the
    published, recorded seed init) is never a member and is excluded even if
    present; the rolling ``resume.pt`` snapshot and the ``latest`` pointer
    can never appear at all, structurally --
    :func:`core.checkpoint.list_published_versions` only ever matches
    ``ckpt-<digits>.pt`` filenames, so nothing else this directory might
    contain is ever consulted, let alone scheduled.

    Args:
        run_dir: The watched run's root directory.
        k_total: The run's fixed, total checkpoint count *K*
            (:func:`watched_k_total`).

    Returns:
        The schedulable versions, ascending.

    Raises:
        ValueError: If ``k_total < 1``.
    """
    if k_total < 1:
        raise ValueError(f"k_total must be >= 1, got {k_total}")
    published = list_published_versions(checkpoint_dir(run_dir))
    return tuple(v for v in published if 1 <= v <= k_total)


# --- cell-scheduling arithmetic: one member's full required cell-id set --------

#: Only form 7's candidate identity meets rung-8 historical opponents (design
#: doc §9's cell semantics: "form 7 additionally meets the rung-7 forms of
#: selected historical checkpoints... it is this form's rating that §1
#: reports"). Forms 5/6 never do, regardless of what a caller's ``forms``
#: sequence contains.
RUNG8_CANDIDATE_FORM = 7


def agent_identity(rung: int, version: int) -> str:
    """Return the ``f"rung{rung}-v1-{version}"`` agent identity string.

    Mirrors ``core.eval_agents``' ``NetworkPolicyAgent``/``SearchAgent``
    naming and ``core.eval_store.CellId.candidate_identity`` exactly -- a
    read of that pinned convention, not a second, independent source of
    truth for it.

    Args:
        rung: The agent's form/rung number (5, 6, or 7 in v1).
        version: The checkpoint's model-version ordinal.

    Returns:
        The identity string.
    """
    return f"rung{rung}-v1-{version}"


def required_cell_ids(
    member_version: int,
    profile: EvalProfile,
    available_versions: Sequence[int],
    k_total: int,
    forms: Sequence[int],
) -> list[str]:
    """Return one member's full required cell-id set (design doc §9's cell semantics).

    Every configured form of ``member_version`` plays every network-free
    rung the game's :class:`~core.eval_profile.EvalProfile` declares (forms x
    rungs -- 3 x 4 = 12 cells at the production ladder), plus -- form 7 only
    (:data:`RUNG8_CANDIDATE_FORM`), and only if 7 is among ``forms`` -- the
    rung-8 historical opponents :func:`core.eval_agents.historical_opponents`
    selects.

    Args:
        member_version: The candidate checkpoint's version (>= 1).
        profile: The game's declared network-free ladder.
        available_versions: Every member version currently schedulable
            (:func:`schedulable_versions`) -- must include ``member_version``
            itself (``core.eval_agents.historical_opponents``' own domain
            check).
        k_total: The run's fixed, total checkpoint count *K*.
        forms: The checkpoint-parameterized agent forms to schedule (a
            non-empty subset of ``{5, 6, 7}`` in v1 -- typically
            ``EvalConfig.forms``).

    Returns:
        Every required cell id, deduplicated, in ascending (string) order.

    Raises:
        ValueError: If ``member_version < 1``, ``forms`` is empty, or
            propagated from :func:`core.eval_agents.historical_opponents`
            (e.g. ``member_version`` is not itself a member of
            ``available_versions``).
    """
    if member_version < 1:
        raise ValueError(f"member_version must be >= 1, got {member_version}")
    forms_sorted = sorted(set(forms))
    if not forms_sorted:
        raise ValueError("forms must be non-empty")

    opponent_identities = [profile.rung_identity(rung) for rung in profile.rungs()]
    cells = {
        build_cell_id(member_version, form, opponent)
        for form in forms_sorted
        for opponent in opponent_identities
    }
    if RUNG8_CANDIDATE_FORM in forms_sorted:
        for u in historical_opponents(available_versions, member_version, k_total=k_total):
            cells.add(
                build_cell_id(
                    member_version,
                    RUNG8_CANDIDATE_FORM,
                    agent_identity(RUNG8_CANDIDATE_FORM, u),
                )
            )
    return sorted(cells)


def cell_seed(eval_seed: int, cell_id: str) -> int:
    """Return one cell's per-cell seed (design doc §9; tasks/m4/009).

    ``derive_seed(eval_seed, PURPOSE_EVAL, cell_id)`` -- a pure function of
    the harness's own root seed and the finished cell id string, so any cell
    is reproducible in isolation and a resumed cell reaches exactly the same
    seed an uninterrupted run would have used
    (``core.runner.play_pairs``'s own per-pair seeds derive from this in
    turn, keyed by pair index -- see that module's docstring).

    Args:
        eval_seed: The harness's own root seed (``EvalConfig.eval_seed``).
        cell_id: The finished cell id (:func:`core.eval_store.build_cell_id`).

    Returns:
        The cell's seed, in ``[0, 2**64)``.
    """
    return derive_seed(eval_seed, PURPOSE_EVAL, cell_id)


# --- watch/resume/catch-up loop (tasks/m4/009 subtask 9.2) ---------------------
#
# The long-lived process itself: poll the watched run for member checkpoints
# not yet complete in the task-5 manifest, play every missing cell through
# core.runner.play_pairs, and survive being killed and relaunched idempotently
# -- a resumed cell picks up at its next missing pair
# (core.eval_store.open_cell_for_write's own contract), and a checkpoint
# published while this process was down is simply on disk the next time
# schedulable_versions() looks. This process is the sole writer of run_dir's
# eval manifest for the duration of one run_watch_loop call -- the m3
# task-11 single-writer discipline, applied to the eval namespace.


@dataclass
class _CandidateCache:
    """One :func:`run_watch_loop` call's checkpoint-load cache (memory stance).

    :func:`core.eval_agents.load_eval_network` reconstructs a checkpoint's exact
    trained architecture and restores its weights from disk -- expensive to
    repeat for every cell a version appears in. This cache loads a given
    candidate version's evaluator **exactly once** and reuses it to build every
    one of that version's rung-5/6/7 agents directly
    (:class:`~core.eval_agents.NetworkPolicyAgent` /
    :class:`~core.eval_agents.SearchAgent`) -- the same one-load-per-checkpoint
    contract ``rung5_agent_factory``/``rung_search_agent_factory`` each give on
    their own, inlined here so the *one* loaded evaluator is shared across all
    three forms instead of being reloaded once per form.

    A rung-8 historical opponent is cached separately, one entry per opponent
    version, built through :func:`core.eval_agents.historical_opponent_factory`
    -- deliberately never sharing an entry with that same version's own
    candidate-role load above, even though both would restore byte-identical
    weights: the historical path's pre-load fingerprint assert exists
    specifically to re-verify a checkpoint at the moment it is seated as an
    opponent (a version scored as a candidate long ago and only much later
    reused as someone else's opponent is exactly the staleness window that
    assert guards -- see
    :func:`core.eval_agents.assert_historical_checkpoint_matches_live_game`),
    and folding it into the candidate cache would silently skip that
    re-verification whenever a version's candidate load happened to already be
    warm.

    **Memory stance:** entries here are never evicted. Every loaded net is a
    small, batch-1-inference CPU model at production scale (D5's 8x128 trunk),
    and a production run's full membership is only ``K = 30`` versions, so the
    unbounded growth over one run is a deliberate, bounded trade against the
    complexity of deciding what can be freed while other cells might still
    reference it. The cache's lifetime is exactly one :func:`run_watch_loop`
    call: a real process kill and relaunch -- the scenario this loop exists to
    survive -- starts a fresh, empty cache next time and simply pays the reload
    cost again for whatever had already been loaded before the kill.

    Attributes:
        ckpt_dir: The watched run's checkpoint directory.
        game: The live game/adapter every load is validated and evaluated
            against.
        device: Torch device for inference.
        eval_sims: The rung-6/7 (and rung-8) search-form simulation budget *S*.
    """

    ckpt_dir: Path
    game: Game
    device: str
    eval_sims: int
    _candidates: dict[int, tuple[Evaluator, int]] = field(default_factory=dict, init=False)
    _historical: dict[int, AgentFactory] = field(default_factory=dict, init=False)

    def candidate_factory(self, version: int, form: int) -> AgentFactory:
        """Return the ``AgentFactory`` for one candidate version's rung ``form``.

        Loads ``version``'s evaluator on first use only (any form); every
        later call for the same version, at any of the three forms, reuses it.

        Args:
            version: The candidate checkpoint's version.
            form: ``5``, ``6``, or ``7``.

        Returns:
            A ``seed -> Agent`` factory matching ``core.runner.AgentFactory``.

        Raises:
            ValueError: If ``form`` is not ``5``, ``6``, or ``7``.
        """
        if form not in (5, 6, 7):
            raise ValueError(f"form must be 5, 6, or 7, got {form}")
        if version not in self._candidates:
            path = published_checkpoint_path(self.ckpt_dir, version)
            self._candidates[version] = load_eval_network(path, self.game, self.device)
        evaluator, model_version = self._candidates[version]

        if form == 5:

            def factory(seed: int) -> NetworkPolicyAgent:
                del seed  # Rung 5 has no per-agent RNG state.
                return NetworkPolicyAgent(evaluator, model_version)

        else:

            def factory(seed: int) -> SearchAgent:
                del seed  # Rungs 6/7 have no per-agent RNG state.
                return SearchAgent(evaluator, model_version, form=form, sims=self.eval_sims)

        return factory

    def historical_factory(self, version: int) -> AgentFactory:
        """Return the rung-8 ``AgentFactory`` for historical opponent ``version``.

        Built through :func:`core.eval_agents.historical_opponent_factory` on
        first use only -- its pre-load fingerprint assert therefore runs
        exactly once per version, the first time it is seated as an opponent
        in this process; every later call for the same version reuses the
        already-built factory.

        Args:
            version: The historical checkpoint's version.

        Returns:
            A ``seed -> SearchAgent`` factory (rung-7 form).

        Raises:
            core.checkpoint.FingerprintMismatchError: If the pre-load assert
                fails (first use only -- see
                :func:`core.eval_agents.assert_historical_checkpoint_matches_live_game`).
        """
        if version not in self._historical:
            self._historical[version] = historical_opponent_factory(
                self.ckpt_dir, self.game, version, device=self.device, sims=self.eval_sims
            )
        return self._historical[version]


_HISTORICAL_OPPONENT_PATTERN = re.compile(r"^rung7-v1-(\d+)$")


def _historical_opponent_version(opponent_id: str) -> int | None:
    """Invert :func:`agent_identity`'s rung-7 form for a historical opponent.

    Args:
        opponent_id: A cell's opponent identity string.

    Returns:
        The historical checkpoint version ``opponent_id`` names, or ``None``
        if it does not match the ``f"rung7-v1-{version}"`` pattern at all
        (i.e. it is one of the profile's own network-free rung identities,
        e.g. ``"random"``).
    """
    match = _HISTORICAL_OPPONENT_PATTERN.match(opponent_id)
    return int(match.group(1)) if match else None


def _resolve_opponent_factory(
    opponent_id: str, rung_identities: Mapping[str, AgentFactory], cache: _CandidateCache
) -> AgentFactory:
    """Resolve one cell's opponent identity string to the factory that builds it.

    Args:
        opponent_id: The cell's opponent identity (a frozen network-free
            rung's identity, or a rung-8 historical ``"rung7-v1-<u>"`` one).
        rung_identities: ``{profile.rung_identity(r): profile.network_free_rungs[r]
            for r in profile.rungs()}`` -- every frozen rung this game declares.
        cache: This loop's checkpoint-load cache (used for the historical case
            only).

    Returns:
        The opponent's ``AgentFactory``.

    Raises:
        ValueError: If ``opponent_id`` matches neither a declared network-free
            rung identity nor the rung-8 historical pattern -- an id this
            process's own scheduling arithmetic (:func:`required_cell_ids`)
            could never have produced, so this signals a store/profile
            mismatch rather than an expected runtime case.
    """
    if opponent_id in rung_identities:
        return rung_identities[opponent_id]
    historical_version = _historical_opponent_version(opponent_id)
    if historical_version is not None:
        return cache.historical_factory(historical_version)
    raise ValueError(
        f"opponent id {opponent_id!r} is neither one of the profile's declared "
        f"network-free rung identities {sorted(rung_identities)} nor a rung-8 "
        "historical identity"
    )


def _play_pending_cell(
    run_dir: Path,
    run_id: str,
    config: EvalConfig,
    game: Game,
    profile: EvalProfile,
    parsed: CellId,
    candidate_factory: AgentFactory,
    opponent_factory: AgentFactory,
    candidate_fingerprint: Mapping[str, Any],
) -> int:
    """Play one required cell to completion, resuming a partial one if needed.

    Args:
        run_dir: The watched run's root directory.
        run_id: The watched run's identity
            (``core.run_identity.read_run_record(run_dir).run_id``).
        config: This launch's resolved :class:`EvalConfig`.
        game: The live game/adapter to play on.
        profile: The game's declared eval profile (for its opening balancer).
        parsed: The cell's parsed identity (``candidate_version`` included).
        candidate_factory: The candidate agent's factory (mirrored-pair "A").
        opponent_factory: The opponent agent's factory (mirrored-pair "B").
        candidate_fingerprint: The live game's artifact fingerprint
            (``core.artifact_fingerprint.build_fingerprint(game)``).

    Returns:
        The number of pairs newly played and appended this call -- ``0`` for a
        cell whose file already had every pair recorded but was not yet marked
        complete in the manifest (e.g. a crash between the last append and
        :func:`core.eval_store.complete_cell` on a prior attempt).
    """
    cell_id = parsed.to_string()
    header = build_header(
        run_id=run_id,
        cell_id=parsed,
        candidate_identity=agent_identity(parsed.rung, parsed.candidate_version),
        opponent_identity=parsed.opponent_id,
        eval_config=config.as_eval_config_snapshot(),
        candidate_fingerprint=candidate_fingerprint,
    )
    start_index = open_cell_for_write(run_dir, header)
    remaining = config.pairs_per_cell - start_index
    played = 0
    if remaining > 0:
        file_path = cell_path(run_dir, cell_id)
        results = play_pairs(
            game,
            candidate_factory,
            opponent_factory,
            n_pairs=remaining,
            seed=cell_seed(config.eval_seed, cell_id),
            opening_balancer=profile.opening_balancer,
            start_pair_index=start_index,
        )
        for result in results:
            append_pair_record(file_path, pair_result_to_record(result))
            played += 1
    complete_cell(run_dir, cell_id)
    return played


@dataclass(frozen=True)
class WatchLoopResult:
    """Summary of one :func:`run_watch_loop` call.

    Attributes:
        polls: Number of poll cycles this call executed (always >= 1).
        pairs_played: Total mirrored pairs actually played and appended during
            this call (``0`` for a poll that found only already-complete work
            or nothing schedulable at all).
        completed_members: Every member version whose full required cell set
            was complete as of the last poll cycle, ascending -- includes a
            member that was already complete before this call started, not
            only ones finished during it.
        stopped_reason: Why the loop returned: ``"single_pass"`` (exactly one
            poll, regardless of completeness), ``"k_complete"`` (every member
            ``1..K`` is complete), or ``"idle"`` (``max_idle_polls``
            consecutive polls in a row played no new pairs).
    """

    polls: int
    pairs_played: int
    completed_members: tuple[int, ...]
    stopped_reason: str


def run_watch_loop(
    run_dir: Path | str,
    config: EvalConfig,
    profile: EvalProfile,
    game: Game,
    *,
    poll_interval: float = 1.0,
    max_idle_polls: int | None = None,
    single_pass: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> WatchLoopResult:
    """Run the M4 watch/resume/catch-up loop over one watched run (§9; 9.2).

    Resolves ``config`` against any already-recorded launch provenance first
    (:func:`resolve_eval_launch` -- refuses on a material/protocol difference,
    exactly as a fresh process relaunch must), then repeatedly: reads the
    watched run's currently-schedulable member versions
    (:func:`schedulable_versions`); for each, registers its full required
    cell-id set in the task-5 manifest (:func:`core.eval_store.register_member`
    -- idempotent, so a member seen again on a later poll is a no-op) and plays
    every cell not yet complete, resuming a partial one at its next missing
    pair (:func:`core.eval_store.open_cell_for_write`'s own contract) rather
    than replaying it from the start. A cell already marked complete is never
    reopened, and this process is the sole writer of ``run_dir``'s eval
    manifest for the duration of the call (the m3 task-11 single-writer
    discipline, applied to the eval namespace).

    A checkpoint published to ``run_dir`` while this function was not running
    (or in between two of its polls) needs no special handling: the very next
    poll's :func:`schedulable_versions` call simply finds it on disk and
    schedules it like any other member.

    Args:
        run_dir: The watched run's root directory.
        config: This launch's resolved :class:`EvalConfig`.
        profile: The watched game's declared eval profile
            (``games.registry.build_eval_profile`` at the script layer --
            this function never imports ``games.*`` itself).
        game: The watched game's live adapter instance, matching ``profile``
            and the watched run's own encoding surface.
        poll_interval: Seconds to sleep between polls (never consulted when
            ``single_pass`` is set, and never on the final poll before
            returning for any other stop reason).
        max_idle_polls: Stop after this many consecutive polls that played no
            new pairs at all, or run indefinitely (subject to the
            all-members-complete stop condition below) when ``None``.
        single_pass: Run exactly one poll cycle and return, regardless of
            completeness -- the test-facing "process whatever is on disk right
            now, once" mode; a caller also uses two separate calls in this
            mode to model a kill/relaunch or a downtime/catch-up scenario.
        sleep: Injected sleep function (tests pass a no-op or a call-recording
            stub; production leaves this at the default, :func:`time.sleep`).

    Returns:
        A :class:`WatchLoopResult` summarizing the call.

    Raises:
        EvalRelaunchRefusedError: If ``config`` disagrees with already-recorded
            provenance on a material field or the protocol stamp.
        FileNotFoundError: If the watched run has no recorded ``config.json``
            or ``run_record.json`` yet (:func:`watched_k_total`,
            ``core.run_identity.read_run_record``).
    """
    resolve_eval_launch(run_dir, config)
    run_dir = Path(run_dir)
    run_id = read_run_record(run_dir).run_id
    k_total = watched_k_total(run_dir)
    ckpt_dir = checkpoint_dir(run_dir)
    candidate_fingerprint = build_fingerprint(game)
    rung_identities = {
        profile.rung_identity(rung): profile.network_free_rungs[rung] for rung in profile.rungs()
    }
    cache = _CandidateCache(
        ckpt_dir=ckpt_dir, game=game, device=config.device, eval_sims=config.eval_sims
    )

    polls = 0
    idle_polls = 0
    total_pairs_played = 0
    completed_members: set[int] = set()
    stopped_reason = "single_pass"

    while True:
        polls += 1
        available = schedulable_versions(run_dir, k_total)
        pairs_this_poll = 0

        for version in available:
            required = required_cell_ids(version, profile, available, k_total, config.forms)
            register_member(run_dir, version, required)
            pending = [cid for cid in required if not is_cell_complete(run_dir, cid)]
            if not pending:
                completed_members.add(version)
                continue
            for cid in pending:
                parsed = parse_cell_id(cid)
                candidate_factory = cache.candidate_factory(version, parsed.rung)
                opponent_factory = _resolve_opponent_factory(
                    parsed.opponent_id, rung_identities, cache
                )
                pairs_this_poll += _play_pending_cell(
                    run_dir,
                    run_id,
                    config,
                    game,
                    profile,
                    parsed,
                    candidate_factory,
                    opponent_factory,
                    candidate_fingerprint,
                )
            completed_members.add(version)

        total_pairs_played += pairs_this_poll

        if single_pass:
            break
        if len(completed_members) >= k_total:
            stopped_reason = "k_complete"
            break
        if pairs_this_poll == 0:
            idle_polls += 1
            if max_idle_polls is not None and idle_polls >= max_idle_polls:
                stopped_reason = "idle"
                break
        else:
            idle_polls = 0
        sleep(poll_interval)

    return WatchLoopResult(
        polls=polls,
        pairs_played=total_pairs_played,
        completed_members=tuple(sorted(completed_members)),
        stopped_reason=stopped_reason,
    )


def eval_lag(run_dir: Path | str, k_total: int) -> int:
    """Return the harness's current backlog, in checkpoints (§9; the M5.5 liveness check).

    ``newest published member - newest snapshot-complete member``: the highest
    member version currently schedulable (:func:`schedulable_versions`,
    structurally capped at ``k_total``) minus the eval snapshot's contiguous
    complete-member prefix
    (:attr:`core.eval_store.EvalSnapshot.member_prefix` -- the identical
    boundary the §1 report's own K-set completeness gate uses, so "caught up"
    here means exactly what "the report is authoritative" means there).

    Args:
        run_dir: The watched run's root directory.
        k_total: The run's fixed, total checkpoint count *K*.

    Returns:
        The lag, in checkpoints -- ``0`` while nothing is published yet, or
        once every currently-published member is fully scored.
    """
    available = schedulable_versions(run_dir, k_total)
    newest_published = available[-1] if available else 0
    member_prefix = load_snapshot(run_dir).member_prefix
    return max(0, newest_published - member_prefix)
