"""Run identity: material/non-material classification, provenance, lineage (§12 M3, issue #63).

A ``core.runconfig.RunConfig`` is the recorded *protocol*; this module is the
layer above it that turns a protocol into a launchable, resumable,
forkable **run**:

* :class:`RuntimeConfig` + :class:`LaunchConfig` extend the JSON schema with
  the launcher-only scalars a ``RunConfig`` deliberately has no room for --
  actor count, device, this schema's own version, and a handful of
  non-material poll cadences -- without touching ``core/runconfig.py`` itself
  (that module's own golden tests assert its source names no game and its
  schema is exactly the M2.5/M3 protocol fields; this module wraps it rather
  than editing it).
* :data:`FIELD_CLASSIFICATION` declares, for **every** leaf field of a
  :class:`LaunchConfig` (dotted path over its nested-dict shape), whether
  changing it on ``--resume`` is *material* (voids the run's reproducibility
  claim -- refused, no override) or *non-material* (a path, a cadence, a
  label -- proceeds, logged). A leaf field :func:`diff_launch_configs` finds
  that is not in this mapping is a loud :class:`UnclassifiedFieldError`, never
  a silent "assume non-material" default -- the whole point of a frozen
  classification is that a field newly added to the schema cannot slip
  through unclassified.
* :func:`generate_run_id` derives a filesystem-safe, launch-unique identity;
  :func:`write_provenance` / :func:`read_stored_config` / :func:`read_run_record`
  are the run directory's provenance IO (``config.json`` verbatim, plus
  ``run_record.json`` carrying the run id, the milestone's static entry
  condition, and fork lineage).
* :func:`resolve_resume` recomputes the classified diff against a run's
  stored config and raises :class:`MaterialConfigDiffError` naming every
  offending field on any material difference -- **no override flag exists**.
  :func:`resolve_fork` never diffs at all: a fork is a deliberately new
  identity, free to change anything, whose lineage records where it came
  from rather than refusing to depart from it.

This module imports nothing from ``games/`` -- it operates entirely on
``core.runconfig.RunConfig`` and plain JSON-shaped data, matching the
repo-wide rule that ``core/`` stays game-agnostic (``games/`` may import this
module; this module may never import ``games/``).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.replay_shard import _atomic_write_json
from core.runconfig import (
    RunConfig,
    _check_keys,
    _float,
    _int,
    _mapping,
    _non_empty,
    _positive,
    _str,
)

# --- field classification ----------------------------------------------------

MATERIAL = "material"
NON_MATERIAL = "non_material"

#: Every leaf field of a :class:`LaunchConfig`'s nested-dict shape (dotted
#: path), classified once, here, and nowhere else. ``"device"`` is handled
#: specially by :func:`diff_launch_configs` (compared by *kind* -- cpu vs.
#: cuda -- never by index) rather than listed with a plain classification.
#:
#: Judgment calls (documented, not silently assumed):
#:
#: * ``evaluation.*`` and ``loss_predicates.*`` are the M2.5/M4 evaluation and
#:   exit-gate protocols -- read by separate, independently-seeded tooling
#:   *after* self-play/training data already exists. They have zero causal
#:   effect on what an M3 run's actors/learner produce, so changing them
#:   on resume cannot desynchronize a resumed run from its uninterrupted
#:   twin. Classified non-material.
#: * ``throughput.*`` is the M2.5 go/no-go spike-measurement protocol,
#:   likewise orthogonal to what a run's actors/learner do. Non-material.
#: * ``training.checkpoint_selection`` selects *which already-produced*
#:   checkpoint a later evaluation reads; it does not change what the
#:   learner trains or publishes. Classified material anyway, conservatively,
#:   because the issue's spec lists it among the ``training`` scalars without
#:   carving it out -- erring material only costs an extra resume refusal,
#:   never a silent reproducibility gap.
#: * ``name`` is a descriptive label (also the seed for the run id's
#:   filesystem-safe prefix) with no bearing on stochastic streams or data
#:   schema -- non-material, like ``run_dir``.
FIELD_CLASSIFICATION: dict[str, str] = {
    # --- RunConfig top level ---
    "name": NON_MATERIAL,
    "game": MATERIAL,
    "game_config": MATERIAL,
    "run_seed": MATERIAL,
    "run_dir": NON_MATERIAL,
    # --- self_play (D7/D10 pinned scalars) ---
    "self_play.sims": MATERIAL,
    "self_play.k_temp": MATERIAL,
    "self_play.dirichlet_eps": MATERIAL,
    "self_play.dirichlet_alpha_numerator": MATERIAL,
    "self_play.root_noise": MATERIAL,
    # --- training (D5/§6.2 pinned scalars) ---
    "training.games": MATERIAL,
    "training.learner_steps": MATERIAL,
    "training.steps_per_game": MATERIAL,
    "training.batch_size": MATERIAL,
    "training.replay_window": MATERIAL,
    "training.learning_rate": MATERIAL,
    "training.warmup_steps": MATERIAL,
    "training.cosine_total_steps": MATERIAL,
    "training.aux_loss_weight": MATERIAL,
    "training.checkpoint_selection": MATERIAL,
    "training.publish_interval": MATERIAL,
    "training.checkpoint_count": MATERIAL,
    "training.replay_warmup_positions": MATERIAL,
    # --- evaluation (M4 protocol; orthogonal to the M3 run -- see above) ---
    "evaluation.agent_form": NON_MATERIAL,
    "evaluation.sims": NON_MATERIAL,
    "evaluation.root_noise": NON_MATERIAL,
    "evaluation.move_selection": NON_MATERIAL,
    "evaluation.opponent": NON_MATERIAL,
    "evaluation.n_pairs": NON_MATERIAL,
    "evaluation.eval_seed": NON_MATERIAL,
    "evaluation.min_score_rate": NON_MATERIAL,
    # --- loss_predicates (M2.5 exit-gate; orthogonal -- see above) ---
    "loss_predicates.head_window_steps": NON_MATERIAL,
    "loss_predicates.tail_window_steps": NON_MATERIAL,
    "loss_predicates.policy_max_ratio": NON_MATERIAL,
    "loss_predicates.value_max_ratio": NON_MATERIAL,
    # --- throughput (M2.5 go/no-go spike; orthogonal -- see above) ---
    "throughput.warmup_games": NON_MATERIAL,
    "throughput.measure_games": NON_MATERIAL,
    "throughput.projection_sims": NON_MATERIAL,
    "throughput.projection_plies_per_game": NON_MATERIAL,
    "throughput.min_projected_games_per_hour": NON_MATERIAL,
    # --- LaunchConfig's own fields (launcher-level, not part of RunConfig) ---
    "num_actors": MATERIAL,
    # "device" is intentionally absent: core.run_identity.diff_launch_configs
    # special-cases it (kind-only comparison), never a plain classification.
    "schema_version": MATERIAL,
    "runtime.refresh_poll_interval": NON_MATERIAL,
    "runtime.pacing_poll_interval": NON_MATERIAL,
    "runtime.ceiling_poll_interval": NON_MATERIAL,
}

# The one field this classification map deliberately omits -- diffed by
# device *kind* (cpu vs. cuda), never by the raw string (which may carry an
# index, e.g. "cuda:0" vs "cuda:1").
_DEVICE_FIELD = "device"


class UnclassifiedFieldError(Exception):
    """Raised when a config leaf field has no entry in :data:`FIELD_CLASSIFICATION`.

    A field the classifier doesn't know about must never silently default to
    non-material -- this is the loud alternative: extending the schema
    requires extending the classification in the same change.
    """


class MaterialConfigDiffError(ValueError):
    """Raised by :func:`resolve_resume` when the passed config differs materially.

    Attributes:
        material: The offending fields, dotted path -> ``(stored, new)``.
    """

    def __init__(self, material: Mapping[str, tuple[Any, Any]]) -> None:
        self.material = dict(material)
        fields = ", ".join(sorted(self.material))
        super().__init__(f"resume refused: material config field(s) differ: {fields}")


def device_kind(device: str) -> str:
    """Return a device string's *kind* -- ``"cpu"``/``"cuda"``, never the index.

    Args:
        device: A torch device string, e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``.

    Returns:
        ``"cuda"`` for any CUDA device (indexed or not), ``"cpu"`` for the CPU
        device, or the lowercased string up to its first ``":"`` for anything
        else (forward-compatible with a device kind this module doesn't know
        about yet).
    """
    normalized = device.strip().lower()
    if normalized.startswith("cuda"):
        return "cuda"
    return normalized.split(":", 1)[0]


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested-dict structure into ``{"a.b.c": leaf_value}``.

    Args:
        obj: A JSON-shaped value (nested ``dict``\\ s bottoming out in
            scalars); every :class:`LaunchConfig` field is exactly this
            shape, with no lists.
        prefix: The dotted path accumulated so far (internal recursion arg).

    Returns:
        Every leaf, keyed by its dotted path.
    """
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value, path))
        return out
    return {prefix: obj}


# --- RuntimeConfig / LaunchConfig --------------------------------------------

#: This module's own schema version -- bumped only when :class:`LaunchConfig`'s
#: shape changes (a field added/removed/reinterpreted). A config file whose
#: ``schema_version`` disagrees fails to parse at all
#: (:meth:`LaunchConfig.from_dict`), which is a stricter, earlier check than
#: the material-diff comparison ever needs to make for this field --
#: :data:`FIELD_CLASSIFICATION` still lists it as material for documentation
#: completeness (every field must be classified) even though a genuine
#: mismatch is caught by the loader first.
LAUNCH_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RuntimeConfig:
    """Non-material launcher cadences (design doc's own wording: "poll/flush cadences").

    Every field here is forwarded verbatim to ``core.ipc.launch_run``'s
    matching keyword argument; none of them influences what data a run
    produces, only how eagerly its processes poll for it.

    Attributes:
        refresh_poll_interval: Seconds between an actor's retries while no
            checkpoint has been published yet.
        pacing_poll_interval: Seconds between an actor's pacing-hold retries.
        ceiling_poll_interval: Seconds between the learner's D5
            replay-ceiling retries.
    """

    refresh_poll_interval: float = 1.0
    pacing_poll_interval: float = 1.0
    ceiling_poll_interval: float = 1.0

    def __post_init__(self) -> None:
        """Validate every cadence is a positive number of seconds.

        Raises:
            ValueError: If any interval is not positive.
        """
        for name in ("refresh_poll_interval", "pacing_poll_interval", "ceiling_poll_interval"):
            _positive(getattr(self, name), f"runtime.{name}")

    def to_dict(self) -> dict[str, Any]:
        """Return this config as a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RuntimeConfig:
        """Build a :class:`RuntimeConfig` from a parsed JSON object.

        Args:
            raw: The ``runtime`` object.

        Returns:
            The validated config.

        Raises:
            ValueError: On unknown/missing keys or a non-positive interval.
            TypeError: On a wrong value type.
        """
        where = "runtime"
        _check_keys(
            raw,
            ("refresh_poll_interval", "pacing_poll_interval", "ceiling_poll_interval"),
            where,
        )
        return cls(
            refresh_poll_interval=_float(raw, "refresh_poll_interval", where),
            pacing_poll_interval=_float(raw, "pacing_poll_interval", where),
            ceiling_poll_interval=_float(raw, "ceiling_poll_interval", where),
        )


_LAUNCH_ONLY_KEYS = ("num_actors", "device", "schema_version", "runtime")


@dataclass(frozen=True)
class LaunchConfig:
    """A launchable run: a full :class:`~core.runconfig.RunConfig` plus launcher scalars.

    The JSON file this wraps is flat -- every ``RunConfig`` key plus these
    four launcher-only keys all live at the top level (``configs/blokus_duo.json``
    is the reference instance) -- rather than nesting ``RunConfig`` under its
    own sub-key, so the file reads as one coherent protocol.

    Attributes:
        run: The wrapped, adapter-and-schema-validated run protocol.
        num_actors: Number of actor processes to launch. Material (design
            constraint): actor ids seed the per-actor streams and define the
            producer set, so changing this changes the replay distribution.
        device: Torch device string for every process. Material *by kind*
            only (:func:`device_kind`) -- the index is not compared.
        schema_version: This module's own schema version
            (:data:`LAUNCH_SCHEMA_VERSION`); a mismatch is rejected at parse
            time, before any diff is computed.
        runtime: Non-material poll cadences (:class:`RuntimeConfig`).
    """

    run: RunConfig
    num_actors: int
    device: str
    schema_version: int
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        """Validate the launcher-only scalars.

        Raises:
            ValueError: If ``num_actors`` is not positive, ``device`` is
                blank, or ``schema_version`` disagrees with
                :data:`LAUNCH_SCHEMA_VERSION`.
            TypeError: If ``run`` is not a :class:`~core.runconfig.RunConfig`
                or ``runtime`` is not a :class:`RuntimeConfig`.
        """
        if not isinstance(self.run, RunConfig):
            raise TypeError(f"run: expected RunConfig, got {type(self.run).__name__}")
        if not isinstance(self.runtime, RuntimeConfig):
            raise TypeError(f"runtime: expected RuntimeConfig, got {type(self.runtime).__name__}")
        _positive(self.num_actors, "num_actors")
        _non_empty(self.device, "device")
        if self.schema_version != LAUNCH_SCHEMA_VERSION:
            raise ValueError(
                f"launch config schema_version mismatch: stored={self.schema_version!r} "
                f"live={LAUNCH_SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the flat, JSON-shaped dict this config round-trips through.

        Returns:
            ``run.to_dict()``'s keys plus ``num_actors``/``device``/
            ``schema_version``/``runtime`` at the same top level.
        """
        payload = dict(self.run.to_dict())
        payload["num_actors"] = self.num_actors
        payload["device"] = self.device
        payload["schema_version"] = self.schema_version
        payload["runtime"] = self.runtime.to_dict()
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LaunchConfig:
        """Build a :class:`LaunchConfig` from a parsed JSON object.

        Args:
            raw: The parsed config file. ``_``-prefixed keys are ignored
                (forwarded to ``RunConfig.from_dict``, which already ignores
                them at every level).

        Returns:
            The validated, frozen config.

        Raises:
            ValueError: If a launcher-only key is missing, ``schema_version``
                is unsupported, or (via ``RunConfig.from_dict``) the embedded
                protocol has unknown/missing keys or an out-of-range value --
                including an unrecognized stray top-level key, since only the
                four launcher-only keys are stripped before the rest is
                handed to ``RunConfig.from_dict``'s own strict key check.
            TypeError: On a wrong value type, from either layer.
        """
        if not isinstance(raw, Mapping):
            raise TypeError(f"launch config: expected an object, got {type(raw).__name__}")
        present = {k for k in raw if not k.startswith("_")}
        missing_launch = sorted(set(_LAUNCH_ONLY_KEYS) - present)
        if missing_launch:
            raise ValueError(f"launch config: missing config keys {missing_launch}")

        run_raw = {k: v for k, v in raw.items() if k.startswith("_") or k not in _LAUNCH_ONLY_KEYS}
        run_config = RunConfig.from_dict(run_raw)

        where = "launch config"
        schema_version = _int(raw, "schema_version", where)
        if schema_version != LAUNCH_SCHEMA_VERSION:
            raise ValueError(
                f"launch config schema_version mismatch: stored={schema_version!r} "
                f"live={LAUNCH_SCHEMA_VERSION!r}"
            )
        num_actors = _int(raw, "num_actors", where)
        device = _str(raw, "device", where)
        runtime = RuntimeConfig.from_dict(_mapping(raw, "runtime", where))
        return cls(
            run=run_config,
            num_actors=num_actors,
            device=device,
            schema_version=schema_version,
            runtime=runtime,
        )


def load_launch_config(path: Path | str) -> LaunchConfig:
    """Load and validate a launch config from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The validated, frozen :class:`LaunchConfig`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If any object has unknown or missing keys, or any value
            is out of range / not a registered name.
        TypeError: If the document (or any nested object) is not an object,
            or a value has the wrong JSON type.
    """
    return LaunchConfig.from_dict(json.loads(Path(path).read_text()))


# --- config hashing + run id --------------------------------------------------


def compute_config_hash(launch_config: LaunchConfig) -> str:
    """Return a canonical sha256 hex digest of a launch config's full content.

    Args:
        launch_config: The config to hash.

    Returns:
        The sha256 hex digest of ``json.dumps(launch_config.to_dict(),
        sort_keys=True)``.
    """
    payload = json.dumps(launch_config.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_RUN_ID_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def generate_run_id(launch_config: LaunchConfig, *, now: float | None = None) -> str:
    """Derive a filesystem-safe, launch-unique run id.

    ``<safe-name>-<UTC timestamp to the microsecond>-<12 hex chars of the
    config hash>`` -- the name for readability, the timestamp for
    uniqueness across launches (microsecond resolution), and the config-hash
    fragment so two runs launched in the same microsecond from *different*
    configs still never collide (astronomically unlikely to matter at
    microsecond resolution alone, but free to include and makes the id
    self-documenting: two ids sharing their hash fragment came from the exact
    same recorded protocol).

    Args:
        launch_config: The config being launched.
        now: The launch timestamp (``time.time()`` epoch seconds). Defaults
            to the real current time; a test passes an explicit value for
            determinism.

    Returns:
        A run id containing only ``[A-Za-z0-9._-]``.
    """
    ts = now if now is not None else time.time()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(ts))
    micros = int(round((ts - int(ts)) * 1_000_000)) % 1_000_000
    digest = compute_config_hash(launch_config)[:12]
    safe_name = _RUN_ID_UNSAFE.sub("-", launch_config.run.name).strip("-") or "run"
    return f"{safe_name}-{stamp}{micros:06d}-{digest}"


def run_root(launch_config: LaunchConfig, run_id: str) -> Path:
    """Return the concrete on-disk directory one launch under ``run_id`` uses.

    ``RunConfig.run_dir`` names the run *family*'s root (e.g.
    ``"runs/blokus_duo"``, shared by every launch/resume/fork of configs
    naming it); the concrete run directory this module's IO targets is always
    that root's ``run_id`` subdirectory, so two launches from the same config
    family never collide on disk.

    Args:
        launch_config: The config naming the family root (``run.run_dir``).
        run_id: The concrete run's id.

    Returns:
        ``Path(launch_config.run.run_dir) / run_id``.
    """
    return Path(launch_config.run.run_dir) / run_id


# --- diffing -------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigDiff:
    """The classified diff between two :class:`LaunchConfig`\\ s.

    Attributes:
        material: Differing fields that void reproducibility, dotted path ->
            ``(old, new)``.
        non_material: Differing fields that do not, same shape.
    """

    material: dict[str, tuple[Any, Any]]
    non_material: dict[str, tuple[Any, Any]]

    @property
    def is_material(self) -> bool:
        """Whether any material field differs."""
        return bool(self.material)


def diff_launch_configs(old: LaunchConfig, new: LaunchConfig) -> ConfigDiff:
    """Compute the classified field-by-field diff between two launch configs.

    Every leaf field of both configs must appear in
    :data:`FIELD_CLASSIFICATION` (``"device"`` is the one deliberate
    exception, handled by kind -- :func:`device_kind`); an unrecognized leaf
    is a loud :class:`UnclassifiedFieldError`, not a silent non-material
    default.

    Args:
        old: The baseline config (typically the run's stored ``config.json``).
        new: The candidate config (typically freshly passed on the CLI).

    Returns:
        The classified diff.

    Raises:
        UnclassifiedFieldError: If a leaf field of either config has no
            classification entry.
    """
    old_flat = _flatten(old.to_dict())
    new_flat = _flatten(new.to_dict())
    all_paths = set(old_flat) | set(new_flat)

    unclassified = sorted(
        path for path in all_paths if path != _DEVICE_FIELD and path not in FIELD_CLASSIFICATION
    )
    if unclassified:
        raise UnclassifiedFieldError(
            f"config field(s) with no material/non-material classification: {unclassified}"
        )

    material: dict[str, tuple[Any, Any]] = {}
    non_material: dict[str, tuple[Any, Any]] = {}

    old_device, new_device = old_flat.get(_DEVICE_FIELD), new_flat.get(_DEVICE_FIELD)
    if device_kind(old_device) != device_kind(new_device):
        material[_DEVICE_FIELD] = (old_device, new_device)
    elif old_device != new_device:
        non_material[_DEVICE_FIELD] = (old_device, new_device)

    for path in sorted(all_paths - {_DEVICE_FIELD}):
        old_value, new_value = old_flat.get(path), new_flat.get(path)
        if old_value == new_value:
            continue
        bucket = material if FIELD_CLASSIFICATION[path] == MATERIAL else non_material
        bucket[path] = (old_value, new_value)

    return ConfigDiff(material=material, non_material=non_material)


# --- provenance: config.json + run_record.json --------------------------------

CONFIG_FILENAME = "config.json"
RUN_RECORD_FILENAME = "run_record.json"

#: Static references to the M2.5 artifacts this run's entry condition depends
#: on (design constraint): recorded verbatim at launch, never re-verified at
#: runtime -- a NO-GO would have reopened M3 scope doc-first before this CLI
#: was built, not been caught by code here.
ENTRY_CONDITION: dict[str, Any] = {
    "exit_test": {
        "issue": "https://github.com/oojBuffalo/alpha-games/issues/50",
        "status": "PASS",
        "closed_at": "2026-08-18",
        "summary": "M2.5 falsifiable exit test: PASS (issue #50, closed COMPLETED 2026-08-18).",
    },
    "throughput_gate": {
        "issue": "https://github.com/oojBuffalo/alpha-games/issues/66",
        "status": "GO",
        "measured_games_per_hour": 151.2,
        "floor_games_per_hour": 100,
        "device": "RTX 4060 Ti",
        "summary": (
            "M2.5 throughput go/no-go: GO (issue #66) -- projected 151.2 games/hour "
            ">= floor 100 on the RTX 4060 Ti."
        ),
    },
}


@dataclass(frozen=True)
class Lineage:
    """A fork's recorded provenance (design constraint: minimal, fork-as-fresh-start).

    **Judgment call, documented here and in the PR:** this milestone ships
    fork-as-fresh-start-with-lineage only -- a fork never imports the
    parent's checkpoint as its own v0-equivalent. Importing weights would
    mean the fork's learner resumes ``learner_step``/optimizer/scheduler
    state from a checkpoint whose ``run_config`` may not even match the
    fork's own (a fork is explicitly allowed to change material fields), so
    "fast-forward the LR schedule by the parent's learner_step" and "the
    fork's own total_steps stop condition" can disagree in ways a minimal
    implementation cannot safely paper over. :attr:`imported_weights_version`
    exists as the documented seam for that future work -- always ``None`` in
    this milestone.

    Attributes:
        parent_run_id: The parent run's id.
        parent_config_hash: :func:`compute_config_hash` of the parent's
            stored config, at fork time.
        parent_run_dir: The parent's concrete run directory, as a string.
        forked_at: ISO-8601 UTC timestamp of the fork.
        imported_weights_version: The parent model version whose weights were
            imported as the fork's starting point, or ``None`` (this
            milestone: always ``None`` -- see above).
    """

    parent_run_id: str
    parent_config_hash: str
    parent_run_dir: str
    forked_at: str
    imported_weights_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return this lineage as a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Lineage:
        """Build a :class:`Lineage` from a parsed JSON object.

        Args:
            raw: The ``lineage`` object.

        Returns:
            The reconstructed lineage.
        """
        return cls(
            parent_run_id=raw["parent_run_id"],
            parent_config_hash=raw["parent_config_hash"],
            parent_run_dir=raw["parent_run_dir"],
            forked_at=raw["forked_at"],
            imported_weights_version=raw.get("imported_weights_version"),
        )


@dataclass(frozen=True)
class RunRecord:
    """One run's provenance metadata, stored as ``run_record.json``.

    Attributes:
        run_id: This run's id (:func:`generate_run_id`).
        created_at: ISO-8601 UTC timestamp of the launch/fork that created
            this run directory (never updated by a later resume).
        entry_condition: The static M2.5 artifact references
            (:data:`ENTRY_CONDITION`).
        lineage: This run's fork lineage, or ``None`` for a run launched
            fresh (not forked).
    """

    run_id: str
    created_at: str
    entry_condition: Mapping[str, Any]
    lineage: Lineage | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return this record as a plain, JSON-serializable dict."""
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "entry_condition": dict(self.entry_condition),
            "lineage": self.lineage.to_dict() if self.lineage is not None else None,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RunRecord:
        """Build a :class:`RunRecord` from a parsed JSON object.

        Args:
            raw: The parsed ``run_record.json`` content.

        Returns:
            The reconstructed record.
        """
        lineage_raw = raw.get("lineage")
        return cls(
            run_id=raw["run_id"],
            created_at=raw["created_at"],
            entry_condition=raw["entry_condition"],
            lineage=Lineage.from_dict(lineage_raw) if lineage_raw is not None else None,
        )


def iso_now(now: float | None = None) -> str:
    """Return an ISO-8601 UTC timestamp string.

    Args:
        now: Epoch seconds. Defaults to the real current time.

    Returns:
        ``"YYYY-MM-DDTHH:MM:SSZ"``.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now if now is not None else time.time()))


def write_provenance(root: Path | str, launch_config: LaunchConfig, record: RunRecord) -> None:
    """Durably, atomically write a run directory's provenance pair.

    Called exactly once per run identity, at launch or fork time -- never on
    resume, which must never mutate the recorded config (module docstring).

    Args:
        root: The concrete run directory (:func:`run_root`). Created if
            missing.
        launch_config: The config to record verbatim.
        record: The run's provenance metadata.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(root / CONFIG_FILENAME, launch_config.to_dict())
    _atomic_write_json(root / RUN_RECORD_FILENAME, record.to_dict())


def read_stored_config(root: Path | str) -> LaunchConfig:
    """Read a run directory's recorded ``config.json``.

    Args:
        root: The concrete run directory.

    Returns:
        The stored, validated :class:`LaunchConfig`.

    Raises:
        FileNotFoundError: If no ``config.json`` exists at ``root`` (not a
            run directory this module created).
    """
    path = Path(root) / CONFIG_FILENAME
    return LaunchConfig.from_dict(json.loads(path.read_text()))


def read_run_record(root: Path | str) -> RunRecord:
    """Read a run directory's recorded ``run_record.json``.

    Args:
        root: The concrete run directory.

    Returns:
        The stored :class:`RunRecord`.

    Raises:
        FileNotFoundError: If no ``run_record.json`` exists at ``root``.
    """
    path = Path(root) / RUN_RECORD_FILENAME
    return RunRecord.from_dict(json.loads(path.read_text()))


# --- resume / fork resolution --------------------------------------------------


@dataclass(frozen=True)
class ResumeResolution:
    """What :func:`resolve_resume` hands the caller once a resume is accepted.

    Attributes:
        run_id: The (unchanged) run id from the stored record.
        run_root: The (unchanged) concrete run directory.
        effective_config: The config to actually run with -- the freshly
            passed one (its material fields are, by construction of a
            successful resolve, equal to the stored config's; its
            non-material fields are the intentionally-updated ones).
        non_material_diff: Non-material fields that changed, for the caller
            to log.
    """

    run_id: str
    run_root: Path
    effective_config: LaunchConfig
    non_material_diff: dict[str, tuple[Any, Any]]


def resolve_resume(run_dir: Path | str, new_launch_config: LaunchConfig) -> ResumeResolution:
    """Validate a resume request and resolve the run identity to continue under.

    Loads the run directory's stored ``config.json``, computes the classified
    diff against ``new_launch_config``, and refuses on any material
    difference -- **no override flag exists** (design constraint). The stored
    ``config.json`` itself is never rewritten by a resume (module docstring):
    only a fresh launch or fork ever calls :func:`write_provenance`.

    Args:
        run_dir: The existing run directory to resume.
        new_launch_config: The config freshly passed on this invocation.

    Returns:
        The resolution the caller uses to continue the run.

    Raises:
        FileNotFoundError: If ``run_dir`` has no recorded provenance.
        MaterialConfigDiffError: If any material field differs, naming every
            offending field.
        UnclassifiedFieldError: If either config carries an unclassified leaf
            field.
    """
    root = Path(run_dir)
    stored = read_stored_config(root)
    record = read_run_record(root)
    diff = diff_launch_configs(stored, new_launch_config)
    if diff.is_material:
        raise MaterialConfigDiffError(diff.material)
    return ResumeResolution(
        run_id=record.run_id,
        run_root=root,
        effective_config=new_launch_config,
        non_material_diff=diff.non_material,
    )


@dataclass(frozen=True)
class ForkResolution:
    """What :func:`resolve_fork` hands the caller to start a brand-new run identity.

    Attributes:
        run_id: The newly generated run id (never the parent's).
        run_root: The new, not-yet-existing concrete run directory.
        lineage: The recorded parent provenance.
    """

    run_id: str
    run_root: Path
    lineage: Lineage


def resolve_fork(
    parent_run_dir: Path | str, new_launch_config: LaunchConfig, *, now: float | None = None
) -> ForkResolution:
    """Resolve a fork: a genuinely new run identity, with recorded lineage.

    Never diffs against the parent (module docstring) -- forks may freely
    change material fields; that is their purpose. The parent run directory
    is never written to.

    Args:
        parent_run_dir: The existing run directory to fork from.
        new_launch_config: The config for the new run (may differ from the
            parent's in any field, material or not).
        now: The fork timestamp (epoch seconds). Defaults to the real current
            time; a test passes an explicit value for determinism.

    Returns:
        The resolution the caller uses to start the fork.

    Raises:
        FileNotFoundError: If ``parent_run_dir`` has no recorded provenance.
    """
    parent_root = Path(parent_run_dir)
    parent_config = read_stored_config(parent_root)
    parent_record = read_run_record(parent_root)
    new_run_id = generate_run_id(new_launch_config, now=now)
    return ForkResolution(
        run_id=new_run_id,
        run_root=run_root(new_launch_config, new_run_id),
        lineage=Lineage(
            parent_run_id=parent_record.run_id,
            parent_config_hash=compute_config_hash(parent_config),
            parent_run_dir=str(parent_root),
            forked_at=iso_now(now),
            imported_weights_version=None,
        ),
    )
