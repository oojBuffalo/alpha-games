"""The learner checkpoint bundle: schema, atomic IO, and validate-on-load (§6, §12 M3).

Torch lives here (and in ``core/network.py`` / ``core/losses.py`` / ``core/train.py``)
only — the pyproject confinement pin; adapters and the rest of ``core/`` stay
stdlib-pure. This module is deliberately not exported from ``core/__init__`` so
that ``import core`` never pulls torch.

**What a checkpoint bundles** (:class:`CheckpointBundle`, issue #56): the net's
``state_dict``, the optimizer's ``state_dict`` (momentum buffers included), the
AMP ``GradScaler``'s ``state_dict``, the learner-step counter, the full run
config, a learner-owned metrics high-water snapshot (opaque — issue #62 owns
its contents), and the canonical artifact fingerprint
(``core.artifact_fingerprint.build_fingerprint``, orientation hash included).
Every field is a plain ``dict``/``list``/``tuple``/Python-primitive/``torch.Tensor``
structure — never a custom class instance — so the whole bundle round-trips
through ``torch.load(..., weights_only=True)``: the pickle-free loader, not the
"trust anything on disk" one (:func:`_load_raw`'s docstring says why this
module never needs ``weights_only=False``).

**LR-schedule position.** ``core.train.make_lr_lambda`` is a pure function of
the absolute learner-step index (verified against ``core.train.make_lr_scheduler``:
constructing a fresh ``LambdaLR`` and fast-forwarding it with ``learner_step``
calls to ``.step()`` reproduces the exact LR a continuously-stepped scheduler
would have at that step — see ``tests/test_checkpoint.py``'s golden). So this
module stores only the learner-step counter, never a separate scheduler state
blob; a resuming caller rebuilds the schedule from
``(run_config, learner_step)`` and restores only the optimizer's *tensor*
state (momentum buffers) from ``optimizer_state_dict``.

**Two namespaces, one serializer.** :func:`build_bundle` / :func:`_write_bundle`
/ :func:`_load_raw` are the single serializer/deserializer pair; the namespace
split below is purely file naming plus policy on top of that shared pair —
never a second bundle shape.

* **Published** (``ckpt-<version>.pt``, :func:`write_published_checkpoint`):
  immutable — writing an existing version raises — the primary-contrast
  candidate set a later milestone's evaluation harness draws from. A
  ``latest`` pointer file (:func:`write_latest_pointer`) names the newest
  published version by atomic replace; it always names an existing published
  checkpoint (never dangling), though it can go *stale* relative to the
  directory (a later publish that crashes after writing ``ckpt-<v>.pt`` but
  before repointing ``latest`` — :func:`list_published_versions` is
  therefore the ground truth for "what versions exist", never the pointer).
  Version ``0`` is the seeded network initialization: a recorded artifact,
  never a K-member of the evaluation candidate set (mirrors the pinned
  publish/K rules — every publish is a checkpoint; crash-resume snapshots are
  never published K-members; seeded-init v0 is recorded but not a K member).
  This module implements the schema, the IO, and resume-selection only — the
  publish *cadence* (how often ``version`` advances, and K itself) is a
  learner-loop concern outside this module's scope.
* **Resume** (``resume.pt``, :func:`write_resume_snapshot`): a rolling
  snapshot, atomic replace, taken between publish intervals so a crash never
  loses more than the last snapshot interval's training. Never enumerable as
  a published version (a distinct, fixed filename outside the
  ``ckpt-*.pt`` glob) and never counted toward K.

**Resume selection** (:func:`select_resume_bundle`): the snapshot wins when it
is *newer* than the newest publish — "newer" meaning a strictly larger
``learner_step`` recorded inside the bundle, never a filesystem mtime; ties and
an older snapshot fall back to the newest publish; no snapshot falls back to
the newest publish; neither existing is a clean fresh-start signal —
``None``, documented, not an exception (a caller building a brand-new run
tells "start fresh" from "checkpoint IO is broken" by checking for ``None``
vs. an exception).

**Validate-on-load.** :func:`load_checkpoint` and :func:`select_resume_bundle`
both call ``core.artifact_fingerprint.compare_fingerprints`` — the same
fingerprint helper ``core.replay_shard.read_shard`` uses, never a weaker
game-name check — as the very first thing after the bytes are loaded, before
returning anything to the caller. Neither function ever applies a bundle's
state to a live ``net``/``optimizer``/``scaler``: they only return the
:class:`CheckpointBundle`, so "raise before touching net/optimizer" holds by
construction — the caller does the ``load_state_dict`` calls, which never run
if this module raised first.

**Atomicity.** Every write here reuses ``core.replay_shard``'s exact
temp-name-then-``os.replace`` primitives (``_atomic_write`` for the ``.pt``
bundles, ``_atomic_write_json`` for the ``latest`` pointer) — the same
pattern the shard writer and the replay manifest use — so a reader can never
observe a torn file under a checkpoint's final name, and an abandoned
``*.tmp-<uuid>`` file from a crashed write is never picked up by anything
here (the published-version glob matches only ``ckpt-<digits>.pt`` exactly,
and the resume/latest readers target their own fixed filenames directly).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from core.artifact_fingerprint import (
    FingerprintMismatchError,  # noqa: F401  (re-exported: the caller-facing raise type)
    build_fingerprint,
    compare_fingerprints,
)
from core.game import Game
from core.replay_shard import _atomic_write, _atomic_write_json

# This module's own bundle-shape version -- bumped only when a field is added,
# renamed, or reinterpreted (never when an adapter's fingerprint or a run's
# config values merely change -- those are core.artifact_fingerprint.SCHEMA_VERSION's
# and ordinary content's job respectively). Every reader compares against this
# exact value, mirroring core.replay_window.MANIFEST_SCHEMA_VERSION's pattern.
CHECKPOINT_SCHEMA_VERSION = 1

# The seeded-init model version: recorded (a caller may publish it), but never
# a K-member of the evaluation candidate set. Documentation only -- this
# module does not special-case version 0 in any IO path.
SEEDED_INIT_VERSION = 0

RESUME_FILENAME = "resume.pt"
LATEST_FILENAME = "latest"

_PUBLISHED_PATTERN = re.compile(r"^ckpt-(0|[1-9][0-9]*)\.pt$")


class CheckpointFormatError(Exception):
    """Raised when a checkpoint file's bundle shape is malformed or unsupported.

    Covers a ``schema_version`` this module does not recognize (this module's
    own bundle-shape version, distinct from a fingerprint mismatch) and a
    structurally incomplete payload -- never a fingerprint disagreement (that
    is :class:`~core.artifact_fingerprint.FingerprintMismatchError`, raised
    separately by :func:`load_checkpoint` / :func:`select_resume_bundle`).
    Fail-loud, dedicated exception type, never a silent coercion.
    """


@dataclass(frozen=True)
class CheckpointBundle:
    """Everything one checkpoint captures (issue #56; design doc §6, §12 M3).

    Every field is weights-only-safe (plain ``dict``/``list``/``tuple``/
    Python-primitive/``torch.Tensor`` — see the module docstring), so the
    bundle round-trips through ``torch.load(..., weights_only=True)`` without
    unpickling arbitrary objects.

    Attributes:
        schema_version: This bundle's shape version
            (:data:`CHECKPOINT_SCHEMA_VERSION`).
        version: The model-version ordinal for a published checkpoint (0 is
            the seeded init); a resume snapshot's ``version`` is whatever its
            net's version was when the snapshot was taken and plays no part
            in resume selection (:func:`select_resume_bundle` compares
            ``learner_step``, never ``version``).
        learner_step: The learner-step counter — both the training-progress
            count and the LR schedule's entire resumable position (the
            schedule is a pure function of this value; see the module
            docstring).
        run_config: The full run config as nested plain dicts
            (``core.runconfig.RunConfig.to_dict()``).
        fingerprint: The canonical artifact fingerprint
            (``core.artifact_fingerprint.build_fingerprint``) of the adapter
            this checkpoint's net was trained against, orientation hash
            included.
        model_state_dict: The network's ``state_dict``, CPU tensors.
        optimizer_state_dict: The optimizer's ``state_dict`` (momentum
            buffers and the live LR/weight-decay/etc. param-group scalars),
            CPU tensors.
        scaler_state_dict: The AMP ``GradScaler``'s ``state_dict``. Always
            present (the M2 train path always constructs a scaler —
            ``core.train.make_scaler``); on the CPU path the scaler is
            *disabled* and its ``state_dict()`` is the empty dict ``{}``
            (verified: ``torch.amp.GradScaler("cpu", enabled=False).state_dict()
            == {}``), which round-trips through ``load_state_dict`` as a
            genuine no-op — never ``None``, never a fabricated placeholder.
        metrics: The learner-owned metrics high-water snapshot — an opaque
            dict round-tripped verbatim (issue #62 owns its contents; this
            module only preserves it). Must be the same weights-only-safe
            shape as every other field.
    """

    schema_version: int
    version: int
    learner_step: int
    run_config: dict[str, Any]
    fingerprint: dict[str, Any]
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scaler_state_dict: dict[str, Any]
    metrics: dict[str, Any]


_BUNDLE_FIELDS = (
    "schema_version",
    "version",
    "learner_step",
    "run_config",
    "fingerprint",
    "model_state_dict",
    "optimizer_state_dict",
    "scaler_state_dict",
    "metrics",
)


def _to_cpu(obj: Any) -> Any:
    """Recursively move every tensor in a nested dict/list/tuple to an owned CPU copy.

    ``torch.nn.Module.state_dict()`` and ``torch.optim.Optimizer.state_dict()``
    both nest tensors inside plain ``dict``/``list`` containers (the latter
    also has an int-keyed ``state`` sub-dict); this walks either shape
    uniformly. Every tensor is detached, moved to CPU, and cloned — an owned
    copy independent of whatever live training continues to do to the
    original — so a checkpoint written from a GPU run reloads anywhere, and
    a bundle built once is never silently mutated by a later training step.

    Args:
        obj: A state-dict-shaped structure, or any of its nested values.

    Returns:
        The same structure with every ``torch.Tensor`` replaced by an
        independent CPU copy; non-tensor leaves (``int``, ``float``, ``str``,
        ``bool``, ``None``) are returned unchanged (Python's immutable
        scalars need no defensive copy). ``dict``/``OrderedDict`` inputs
        become plain ``dict``\\ s (weights-only-safe; see the module
        docstring).
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu").clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def build_bundle(
    *,
    version: int,
    learner_step: int,
    game: Game,
    run_config: Mapping[str, Any],
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    metrics: Mapping[str, Any],
) -> CheckpointBundle:
    """Assemble a :class:`CheckpointBundle` from live training objects.

    Args:
        version: The model-version ordinal (0: seeded init). Not required to
            be unique here -- namespace policy (:func:`write_published_checkpoint`'s
            immutability check) is what makes a *published* version unique.
        learner_step: The learner-step counter this bundle was built at.
        game: The adapter ``net`` was trained against; fixes the stored
            fingerprint (``core.artifact_fingerprint.build_fingerprint``).
        run_config: The full run config as nested plain dicts
            (``core.runconfig.RunConfig.to_dict()``).
        net: The network to snapshot (``state_dict()``); not mutated.
        optimizer: The optimizer to snapshot (``state_dict()``); not mutated.
        scaler: The AMP scaler to snapshot (``state_dict()``); not mutated.
        metrics: The learner-owned metrics high-water snapshot, verbatim.

    Returns:
        The assembled bundle, every tensor an owned CPU copy
        (:func:`_to_cpu`) independent of the live objects passed in.

    Raises:
        ValueError: If ``version`` or ``learner_step`` is negative.
    """
    if version < 0:
        raise ValueError(f"version must be >= 0, got {version}")
    if learner_step < 0:
        raise ValueError(f"learner_step must be >= 0, got {learner_step}")
    return CheckpointBundle(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        version=version,
        learner_step=learner_step,
        run_config=dict(run_config),
        fingerprint=build_fingerprint(game),
        model_state_dict=_to_cpu(net.state_dict()),
        optimizer_state_dict=_to_cpu(optimizer.state_dict()),
        scaler_state_dict=_to_cpu(scaler.state_dict()),
        metrics=dict(metrics),
    )


def _bundle_to_payload(bundle: CheckpointBundle) -> dict[str, Any]:
    """Return a bundle as the plain dict ``torch.save`` writes.

    Args:
        bundle: The bundle to flatten.

    Returns:
        A plain ``dict`` over exactly :data:`_BUNDLE_FIELDS`.
    """
    return {name: getattr(bundle, name) for name in _BUNDLE_FIELDS}


def _payload_to_bundle(payload: Mapping[str, Any], path: Path) -> CheckpointBundle:
    """Reconstruct a :class:`CheckpointBundle` from a loaded payload.

    Args:
        payload: The dict ``torch.load`` returned.
        path: The file it came from, for error messages only.

    Returns:
        The reconstructed bundle.

    Raises:
        CheckpointFormatError: If a required field is missing, or
            ``schema_version`` disagrees with :data:`CHECKPOINT_SCHEMA_VERSION`.
    """
    missing = [name for name in _BUNDLE_FIELDS if name not in payload]
    if missing:
        raise CheckpointFormatError(f"checkpoint at {path} is missing field(s): {missing}")
    stored_schema = payload["schema_version"]
    if stored_schema != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointFormatError(
            f"checkpoint at {path} has an unsupported schema_version: "
            f"stored={stored_schema!r} live={CHECKPOINT_SCHEMA_VERSION!r}"
        )
    return CheckpointBundle(**{name: payload[name] for name in _BUNDLE_FIELDS})


def _write_bundle(path: Path, bundle: CheckpointBundle) -> None:
    """Write ``bundle`` to ``path`` atomically (temp-name-then-``os.replace``).

    The one write primitive both namespaces call — ``core.replay_shard._atomic_write``
    verbatim, the same pattern the shard writer and the replay manifest use, so a
    reader can never observe a torn file under ``path``.

    Args:
        path: The destination file (a published ``ckpt-<version>.pt`` or the
            rolling ``resume.pt``).
        bundle: The bundle to write.
    """

    def write_body(fh: Any) -> None:
        torch.save(_bundle_to_payload(bundle), fh)

    _atomic_write(path, write_body)


def _load_raw(path: Path) -> CheckpointBundle:
    """Load a bundle from disk without validating its fingerprint.

    Internal only -- every public reader (:func:`load_checkpoint`,
    :func:`select_resume_bundle`) wraps this with a fingerprint check before
    returning anything to its own caller.

    ``weights_only=True`` deliberately, not ``False``: every field this
    module ever writes is a plain ``dict``/``list``/``tuple``/Python-primitive/
    ``torch.Tensor`` structure (:func:`build_bundle`'s ``_to_cpu``, and the
    fingerprint/run-config/metrics contract of plain JSON-safe values) with no
    custom class instances anywhere, so the pickle-free ``weights_only`` loader
    is sufficient -- this module never needs to trust arbitrary unpickled
    objects from a checkpoint file, even though these are trusted local
    artifacts and ``weights_only=False`` would also work.

    Args:
        path: The checkpoint file to load.

    Returns:
        The bundle, unchecked against any adapter's live fingerprint.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        CheckpointFormatError: If the payload is missing a field or carries
            an unsupported ``schema_version``.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return _payload_to_bundle(payload, path)


def load_checkpoint(path: Path | str, game: Game) -> CheckpointBundle:
    """Load one checkpoint file, validating its fingerprint before returning.

    Order: load the bytes, build ``game``'s live fingerprint, compare -- and
    raise on any disagreement -- *before* returning anything. Neither this
    function nor its caller-visible contract ever calls
    ``net.load_state_dict``/``optimizer.load_state_dict``/etc.; a caller that
    applies the returned bundle's state only does so after this function has
    already succeeded, so "raise before touching net/optimizer" holds by
    construction, not by convention.

    Args:
        path: The checkpoint file (a published ``ckpt-<version>.pt`` or the
            rolling ``resume.pt``).
        game: The adapter to validate the stored fingerprint against (its
            live fingerprint, ``core.artifact_fingerprint.build_fingerprint``).

    Returns:
        The validated bundle.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        CheckpointFormatError: If the payload is malformed or its
            ``schema_version`` is unsupported.
        core.artifact_fingerprint.FingerprintMismatchError: If the stored
            fingerprint disagrees with ``game``'s live one on any field,
            naming every mismatched field.
    """
    bundle = _load_raw(Path(path))
    compare_fingerprints(bundle.fingerprint, build_fingerprint(game))
    return bundle


# --- namespace: published (immutable) + latest pointer -----------------------


def published_checkpoint_path(ckpt_dir: Path | str, version: int) -> Path:
    """Return the immutable published-checkpoint path for one version.

    Args:
        ckpt_dir: The run's checkpoint directory.
        version: The model-version ordinal.

    Returns:
        ``ckpt_dir / "ckpt-<version>.pt"``.
    """
    return Path(ckpt_dir) / f"ckpt-{version}.pt"


def latest_pointer_path(ckpt_dir: Path | str) -> Path:
    """Return the ``latest`` pointer file path for one checkpoint directory.

    Args:
        ckpt_dir: The run's checkpoint directory.

    Returns:
        ``ckpt_dir / "latest"``.
    """
    return Path(ckpt_dir) / LATEST_FILENAME


def list_published_versions(ckpt_dir: Path | str) -> tuple[int, ...]:
    """Return every published version present on disk, ascending.

    The ground truth for "what versions exist" -- never re-derived from the
    ``latest`` pointer, which can go stale relative to the directory (a crash
    between publishing ``ckpt-<v>.pt`` and repointing ``latest``).

    Args:
        ckpt_dir: The run's checkpoint directory. Need not exist yet.

    Returns:
        The published versions found (matching ``ckpt-<digits>.pt`` exactly
        -- an in-flight ``*.tmp-<uuid>`` write is never matched), sorted
        ascending.
    """
    directory = Path(ckpt_dir)
    if not directory.exists():
        return ()
    versions = [
        int(m.group(1))
        for p in directory.iterdir()
        if p.is_file() and (m := _PUBLISHED_PATTERN.match(p.name)) is not None
    ]
    return tuple(sorted(versions))


def newest_published_version(ckpt_dir: Path | str) -> int | None:
    """Return the highest published version on disk, or ``None`` if none exist.

    Args:
        ckpt_dir: The run's checkpoint directory.

    Returns:
        The newest version (:func:`list_published_versions`'s last element),
        or ``None`` for a directory with no published checkpoint yet.
    """
    versions = list_published_versions(ckpt_dir)
    return versions[-1] if versions else None


def write_published_checkpoint(ckpt_dir: Path | str, bundle: CheckpointBundle) -> Path:
    """Publish ``bundle`` as ``ckpt-<bundle.version>.pt``, immutably.

    Existence is checked *before* any byte is written -- a failed attempt
    touches the filesystem not at all, so the existing file's content is
    provably unchanged (never a torn write partway through an "immutable"
    file).

    Args:
        ckpt_dir: The run's checkpoint directory. Created if missing.
        bundle: The bundle to publish, under its own ``bundle.version``.

    Returns:
        The published path.

    Raises:
        FileExistsError: If ``ckpt-<bundle.version>.pt`` already exists --
            published checkpoints are immutable; publish a new version
            instead.
    """
    directory = Path(ckpt_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = published_checkpoint_path(directory, bundle.version)
    if path.exists():
        raise FileExistsError(
            f"published checkpoint version {bundle.version} already exists at {path} "
            "-- published checkpoints are immutable; publish a new version instead"
        )
    _write_bundle(path, bundle)
    return path


def write_latest_pointer(ckpt_dir: Path | str, version: int) -> Path:
    """Atomically point ``latest`` at an existing published version.

    Args:
        ckpt_dir: The run's checkpoint directory.
        version: The published version to point at; must already exist
            (:func:`write_published_checkpoint` first).

    Returns:
        The pointer file's path.

    Raises:
        FileNotFoundError: If ``ckpt-<version>.pt`` does not exist --
            ``latest`` must always name an existing published version, never
            a dangling one.
    """
    directory = Path(ckpt_dir)
    if version not in list_published_versions(directory):
        raise FileNotFoundError(
            f"cannot point latest at version {version}: no published checkpoint "
            f"ckpt-{version}.pt exists at {directory}"
        )
    path = latest_pointer_path(directory)
    _atomic_write_json(path, {"version": version})
    return path


def read_latest_pointer(ckpt_dir: Path | str) -> int | None:
    """Read the ``latest`` pointer's version, or ``None`` if never written.

    Args:
        ckpt_dir: The run's checkpoint directory.

    Returns:
        The pointed-at version, or ``None`` if ``latest`` does not exist yet.
        Callers wanting a guaranteed-live version should prefer
        :func:`newest_published_version` (:func:`list_published_versions` is
        the ground truth; the pointer can go stale -- see the module
        docstring).
    """
    path = latest_pointer_path(ckpt_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return int(payload["version"])


# --- namespace: rolling resume snapshot ---------------------------------------


def resume_path(ckpt_dir: Path | str) -> Path:
    """Return the rolling resume-snapshot path for one checkpoint directory.

    Args:
        ckpt_dir: The run's checkpoint directory.

    Returns:
        ``ckpt_dir / "resume.pt"`` -- a fixed filename, never versioned, and
        never matched by :data:`_PUBLISHED_PATTERN` (so it is never
        enumerable as a published version).
    """
    return Path(ckpt_dir) / RESUME_FILENAME


def write_resume_snapshot(ckpt_dir: Path | str, bundle: CheckpointBundle) -> Path:
    """Overwrite the rolling resume snapshot, atomically.

    Unlike :func:`write_published_checkpoint`, this always succeeds over an
    existing snapshot (the whole point of a *rolling* snapshot) -- the
    atomic-replace pattern (:func:`_write_bundle`) still guarantees a reader
    never observes a torn file under ``resume.pt``, including one caught
    mid-replace by a crash: it either still sees the previous snapshot's
    complete content, or the new one, never a mixture.

    Args:
        ckpt_dir: The run's checkpoint directory. Created if missing.
        bundle: The bundle to snapshot.

    Returns:
        The snapshot path.
    """
    directory = Path(ckpt_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = resume_path(directory)
    _write_bundle(path, bundle)
    return path


# --- resume selection ----------------------------------------------------------


def select_resume_bundle(ckpt_dir: Path | str, game: Game) -> CheckpointBundle | None:
    """Select and load the bundle a resuming learner should start from.

    Selection rule, entirely on ``learner_step`` (never a filesystem mtime,
    never ``version``):

    * A resume snapshot exists and is *newer* than the newest publish (its
      ``learner_step`` is strictly greater, or no publish exists at all) ->
      the snapshot.
    * Otherwise, a published checkpoint exists (the snapshot is absent, or no
      newer than the newest publish) -> the newest publish.
    * Neither exists -> ``None`` -- a clean fresh-start signal. Documented,
      not an exception: a caller distinguishes "nothing to resume from yet"
      (``None``) from "checkpoint IO is broken" (a raised exception) this
      way.

    The winning bundle's fingerprint is validated (the same validate-on-load
    contract as :func:`load_checkpoint`) before it is returned; a losing
    candidate's fingerprint is never checked (it is discarded unread).

    Args:
        ckpt_dir: The run's checkpoint directory.
        game: The adapter to validate the winning bundle's fingerprint
            against.

    Returns:
        The selected, fingerprint-validated bundle, or ``None`` for a fresh
        start.

    Raises:
        CheckpointFormatError: If a candidate's payload is malformed or
            carries an unsupported ``schema_version``.
        core.artifact_fingerprint.FingerprintMismatchError: If the winning
            bundle's stored fingerprint disagrees with ``game``'s live one.
    """
    directory = Path(ckpt_dir)
    resume_file = resume_path(directory)
    newest_version = newest_published_version(directory)

    resume_raw = _load_raw(resume_file) if resume_file.exists() else None
    published_raw = (
        _load_raw(published_checkpoint_path(directory, newest_version))
        if newest_version is not None
        else None
    )

    if resume_raw is not None and (
        published_raw is None or resume_raw.learner_step > published_raw.learner_step
    ):
        winner = resume_raw
    elif published_raw is not None:
        winner = published_raw
    else:
        return None

    compare_fingerprints(winner.fingerprint, build_fingerprint(game))
    return winner
