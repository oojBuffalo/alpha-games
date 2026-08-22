"""The on-disk replay-shard artifact (design doc §5, §6.2, §12 M3).

One stored self-play sample (:class:`SampleRecord`) is mover-relative encoded
planes, the D12 sparse policy target as ``(action_id, visit_count)`` pairs,
the D1 primary target ``z``, the game's declared aux tuple (empty when the
game declares none), the mover id, the ``model_version`` the search ran
against (§6.2's version-pinning), the zero-based ply, and a durable game id.
Samples are packed many-to-a-file into a **shard**: a single ``.npz`` array
file plus a header carrying the shard's identity and its
:mod:`core.artifact_fingerprint`.

Invariant 3 (sparse everywhere) governs the policy target only — never the
input planes, which are small dense tensors and always were. π is therefore
stored *ragged*: one flat ``pi_action_ids``/``pi_visit_counts`` pair of arrays
for the whole shard, sliced per sample by a monotone ``pi_offsets`` array
(length ``num_samples + 1``), rather than a per-sample list of arrays (NumPy
has no first-class ragged array type, and a dense ``(num_samples,
max_legal)`` padded layout would waste space and reintroduce exactly the
dense-target problem Invariant 3 exists to avoid).

**Durable shard identity.** A shard's filename is
``shard-<run_id>-<actor_id>-<seq>.npz`` (:func:`shard_filename`). Collision-
proofness rests on three properties the caller/writer jointly guarantee: (1)
``run_id`` is unique per run, (2) at most one live :class:`ShardWriter` acts
under one ``actor_id`` within a run, and (3) ``seq`` is monotonically
issued and never reused, even across a writer restart, because it lives in a
crash-safe **persisted state file** (:class:`WriterState`) the writer advances
and durably writes *before* publishing the shard file that consumes the
sequence number (§12 M3's ordering, made concrete): a crash between those two
steps burns the sequence number (and any game indices reserved alongside it)
forever, but can never produce a duplicate shard name or reissue a game index
already handed to a caller.

Both the shard file and the writer state file are published via
temp-name-then-``os.replace`` (:func:`_atomic_write`) — atomic on the same
filesystem, so a reader never observes a partially-written file under the
final name.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from core.artifact_fingerprint import (
    build_fingerprint,
    canonical_json,
    compare_fingerprints,
)
from core.game import Action, Game, PlayerId

# The shard container's own format version -- how the arrays below are named
# and packed. Folded into the fingerprint's own ``schema_version`` field
# (core.artifact_fingerprint.SCHEMA_VERSION) rather than tracked separately:
# in this module's lifetime the array layout and the fingerprint shape move
# together, and a reader that rejects an unknown fingerprint schema version
# has already rejected an unknown array layout by construction.
GameId = tuple[str, str, int]


class ShardInvariantError(Exception):
    """Raised when a shard's arrays violate one of the pinned invariants.

    Never raised for a fingerprint disagreement (that is
    :class:`~core.artifact_fingerprint.FingerprintMismatchError`) -- this is
    for the array-level contract: action ids in range, non-negative counts
    with a positive per-sample total, monotone ragged offsets, and
    non-decreasing plies within a game. Fail-loud, dedicated exception type,
    never a silent coercion or a warning.
    """


@dataclass(frozen=True)
class PendingSample:
    """One self-play sample before its durable game id is assigned.

    :class:`ShardWriter` is what turns a batch of these (grouped by game) into
    :class:`SampleRecord`\\ s carrying a real ``game_id`` -- callers that
    already know their durable id (e.g. round-trip tests) construct
    :class:`SampleRecord` directly instead.

    Attributes:
        planes: The adapter's ``encode_state`` output for this position
            (nested sequence or array), stacking to
            ``(input_planes, *input_shape)``.
        sparse_pi: The D12 sparse policy target, ``(action_id, visit_count)``
            pairs over the position's full legal set. Zero-count legal
            actions are kept (they shape the legal-set renormalization).
        z: The D1 primary value target, stated from ``mover``'s perspective.
        aux: The declared aux targets in head order, ``mover``-relative; the
            empty tuple for a game declaring no aux heads.
        mover: The player to move at this position.
        model_version: The published weight version the search ran against
            (§6.2).
        ply: Zero-based ply index within the game.
    """

    planes: Any
    sparse_pi: tuple[tuple[Action, int], ...]
    z: float
    aux: tuple[float, ...]
    mover: PlayerId
    model_version: int
    ply: int


@dataclass(frozen=True)
class SampleRecord:
    """One fully-identified stored self-play sample -- the shard's row shape.

    Attributes:
        planes: See :class:`PendingSample`.
        sparse_pi: See :class:`PendingSample`.
        z: See :class:`PendingSample`.
        aux: See :class:`PendingSample`.
        mover: See :class:`PendingSample`.
        model_version: See :class:`PendingSample`.
        ply: See :class:`PendingSample`.
        game_id: ``(run_id, actor_id, game_index)`` -- the durable, globally
            unique id of the game this sample was played in. A shard never
            splits one game across two files, so every sample sharing a
            ``game_id`` within one shard is a contiguous run.
    """

    planes: Any
    sparse_pi: tuple[tuple[Action, int], ...]
    z: float
    aux: tuple[float, ...]
    mover: PlayerId
    model_version: int
    ply: int
    game_id: GameId

    @staticmethod
    def from_pending(pending: PendingSample, game_id: GameId) -> SampleRecord:
        """Attach a durable ``game_id`` to a :class:`PendingSample`.

        Args:
            pending: The sample, still missing its durable identity.
            game_id: The ``(run_id, actor_id, game_index)`` to attach.

        Returns:
            The equivalent :class:`SampleRecord`.
        """
        return SampleRecord(
            planes=pending.planes,
            sparse_pi=pending.sparse_pi,
            z=pending.z,
            aux=pending.aux,
            mover=pending.mover,
            model_version=pending.model_version,
            ply=pending.ply,
            game_id=game_id,
        )


@dataclass(frozen=True)
class ShardData:
    """The result of reading one shard: its identity plus its samples.

    Attributes:
        run_id: The shard's run id (header field, also the first component of
            every sample's ``game_id``).
        actor_id: The shard's actor id (header field, also the second
            component of every sample's ``game_id``).
        seq: The shard's sequence number.
        fingerprint: The stored fingerprint, already checked against the
            reading adapter's live one (:func:`read_shard` raises before
            returning if they disagree).
        records: The shard's samples, in on-disk order.
    """

    run_id: str
    actor_id: str
    seq: int
    fingerprint: dict[str, Any]
    records: tuple[SampleRecord, ...]


def shard_filename(run_id: str, actor_id: str, seq: int) -> str:
    """Return the exact, pinned shard filename for ``(run_id, actor_id, seq)``.

    Args:
        run_id: The run's identity.
        actor_id: The actor's identity within the run.
        seq: The shard's sequence number (per ``(run_id, actor_id)``).

    Returns:
        ``"shard-<run_id>-<actor_id>-<seq>.npz"``.
    """
    return f"shard-{run_id}-{actor_id}-{seq}.npz"


# --- atomic file publishing -------------------------------------------------


def _atomic_write(path: Path, write_body: Callable[[BinaryIO], None]) -> None:
    """Write ``path`` atomically: full contents under a temp name, then rename.

    A reader can never observe a partially-written file under ``path`` -- it
    either doesn't exist yet, or exists complete. The temp name is unique per
    attempt (a uuid4 suffix) so concurrent writers targeting the same final
    path never collide on the temp file itself.

    Args:
        path: The final destination path.
        write_body: Called once with an open, writable binary file handle;
            must write the complete contents to it.

    Raises:
        Exception: Whatever ``write_body`` (or the filesystem) raises; the
            temp file is removed and ``path`` is left untouched -- the final
            name never exists after a failed write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with open(tmp_path, "wb") as fh:
            write_body(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` as human-readable JSON, atomically.

    Args:
        path: The destination path.
        payload: A JSON-serializable dict.
    """
    blob = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
    _atomic_write(path, lambda fh: fh.write(blob))


# --- packing / unpacking arrays ---------------------------------------------


def _num_actions(game: Game) -> int:
    """Return the dense size of ``game``'s policy head (product of its shape)."""
    n = 1
    for dim in game.policy_shape:
        n *= dim
    return n


def _pack_arrays(game: Game, records: Sequence[SampleRecord]) -> dict[str, np.ndarray]:
    """Build the shard's body arrays from fully-identified sample records.

    Args:
        game: The adapter the records belong to (for ``value_targets`` /
            ``input_planes`` / ``input_shape``).
        records: The samples, in the order they will be stored.

    Returns:
        The array dict (without the header fields) -- see the module
        docstring for the layout: ``planes``, ``pi_action_ids``,
        ``pi_visit_counts``, ``pi_offsets``, ``z``, ``movers``,
        ``model_versions``, ``plies``, ``game_indices``, plus ``aux`` iff the
        game declares aux heads.

    Raises:
        ShardInvariantError: If any record's ``aux`` width disagrees with the
            game's declared aux-head count.
    """
    num_aux = len(game.value_targets.aux_names)
    for i, r in enumerate(records):
        if len(r.aux) != num_aux:
            raise ShardInvariantError(
                f"record {i}: aux has {len(r.aux)} value(s), but "
                f"{type(game).__name__} declares {num_aux}"
            )

    planes = np.stack(
        [np.asarray(r.planes, dtype=np.float32) for r in records],
        axis=0,
    )
    action_ids: list[int] = []
    visit_counts: list[int] = []
    offsets = [0]
    for r in records:
        for action_id, count in r.sparse_pi:
            action_ids.append(action_id)
            visit_counts.append(count)
        offsets.append(len(action_ids))

    arrays: dict[str, np.ndarray] = {
        "planes": planes,
        "pi_action_ids": np.asarray(action_ids, dtype=np.int32),
        "pi_visit_counts": np.asarray(visit_counts, dtype=np.int32),
        "pi_offsets": np.asarray(offsets, dtype=np.int64),
        "z": np.asarray([r.z for r in records], dtype=np.float32),
        "movers": np.asarray([r.mover for r in records], dtype=np.int32),
        "model_versions": np.asarray([r.model_version for r in records], dtype=np.int64),
        "plies": np.asarray([r.ply for r in records], dtype=np.int32),
        "game_indices": np.asarray([r.game_id[2] for r in records], dtype=np.int64),
    }
    if num_aux:
        arrays["aux"] = np.asarray([r.aux for r in records], dtype=np.float32)
    return arrays


def _unpack_records(
    game: Game, run_id: str, actor_id: str, arrays: dict[str, np.ndarray]
) -> tuple[SampleRecord, ...]:
    """Reconstruct :class:`SampleRecord`\\ s from a shard's body arrays.

    Args:
        game: The adapter the shard was fingerprinted against (for the
            declared aux-head count).
        run_id: The shard's header ``run_id`` (first component of every
            reconstructed ``game_id``).
        actor_id: The shard's header ``actor_id`` (second component).
        arrays: The body arrays, already invariant-checked
            (:func:`_validate_arrays`).

    Returns:
        The samples, in on-disk order.
    """
    n = len(arrays["z"])
    num_aux = len(game.value_targets.aux_names)
    offsets = arrays["pi_offsets"]
    records = []
    for i in range(n):
        start, end = int(offsets[i]), int(offsets[i + 1])
        sparse_pi = tuple(
            (int(a), int(c))
            for a, c in zip(
                arrays["pi_action_ids"][start:end],
                arrays["pi_visit_counts"][start:end],
                strict=True,
            )
        )
        aux = tuple(float(v) for v in arrays["aux"][i]) if num_aux else ()
        records.append(
            SampleRecord(
                planes=arrays["planes"][i],
                sparse_pi=sparse_pi,
                z=float(arrays["z"][i]),
                aux=aux,
                mover=int(arrays["movers"][i]),
                model_version=int(arrays["model_versions"][i]),
                ply=int(arrays["plies"][i]),
                game_id=(run_id, actor_id, int(arrays["game_indices"][i])),
            )
        )
    return tuple(records)


# --- invariant validation ----------------------------------------------------


def _validate_arrays(game: Game, arrays: dict[str, np.ndarray]) -> None:
    """Check every pinned array invariant, or raise :class:`ShardInvariantError`.

    Checked (design doc §12 M3 / issue #54): sample-count agreement across all
    parallel arrays; ``pi_offsets`` starts at 0, is non-decreasing, and ends at
    the length of the flat π arrays; every action id is in
    ``[0, num_actions)``; every visit count is ``>= 0``; every sample's total
    visit count (``sum`` over its slice) is ``> 0``; ``plies`` is
    non-decreasing *within* each contiguous run of one ``game_indices`` value
    (it may drop back to 0 when the game index changes); every mover is a
    valid player id; and the ``aux`` array is present iff the game declares
    aux heads, with the declared width.

    Args:
        game: The adapter the shard claims to belong to.
        arrays: The body arrays (:func:`_pack_arrays`'s output, or a shard
            freshly loaded from disk).

    Raises:
        ShardInvariantError: On the first violated invariant category; the
            message names what failed.
    """
    n = len(arrays["z"])
    parallel = ("movers", "model_versions", "plies", "game_indices")
    for name in parallel:
        if len(arrays[name]) != n:
            raise ShardInvariantError(
                f"array length mismatch: z has {n} samples, {name} has {len(arrays[name])}"
            )
    if arrays["planes"].shape[0] != n:
        raise ShardInvariantError(
            f"array length mismatch: z has {n} samples, planes has {arrays['planes'].shape[0]}"
        )
    expected_planes_shape = (n, game.input_planes, *game.input_shape)
    if tuple(arrays["planes"].shape) != expected_planes_shape:
        raise ShardInvariantError(
            f"planes shape {tuple(arrays['planes'].shape)} != declared {expected_planes_shape}"
        )

    offsets = arrays["pi_offsets"]
    if len(offsets) != n + 1:
        raise ShardInvariantError(f"pi_offsets has length {len(offsets)}, expected {n + 1}")
    if n and int(offsets[0]) != 0:
        raise ShardInvariantError(f"pi_offsets must start at 0, got {int(offsets[0])}")
    if np.any(np.diff(offsets) < 0):
        raise ShardInvariantError("pi_offsets is not monotone non-decreasing")
    flat_len = len(arrays["pi_action_ids"])
    if len(arrays["pi_visit_counts"]) != flat_len:
        raise ShardInvariantError(
            f"pi_action_ids has {flat_len} entries, pi_visit_counts has "
            f"{len(arrays['pi_visit_counts'])}"
        )
    if n and int(offsets[-1]) != flat_len:
        raise ShardInvariantError(
            f"pi_offsets ends at {int(offsets[-1])}, but the flat pi arrays hold {flat_len}"
        )

    num_actions = _num_actions(game)
    action_ids = arrays["pi_action_ids"]
    if flat_len and (action_ids.min() < 0 or action_ids.max() >= num_actions):
        raise ShardInvariantError(
            f"pi_action_ids out of range [0, {num_actions}): "
            f"min={int(action_ids.min())} max={int(action_ids.max())}"
        )
    counts = arrays["pi_visit_counts"]
    if flat_len and counts.min() < 0:
        raise ShardInvariantError(f"pi_visit_counts has a negative count: min={int(counts.min())}")
    for i in range(n):
        total = int(counts[int(offsets[i]) : int(offsets[i + 1])].sum())
        if total <= 0:
            raise ShardInvariantError(
                f"sample {i}: total visit count sum(N) = {total}, must be > 0"
            )

    movers = arrays["movers"]
    if n and (movers.min() < 0 or movers.max() >= game.num_players):
        raise ShardInvariantError(
            f"movers out of range [0, {game.num_players}): "
            f"min={int(movers.min())} max={int(movers.max())}"
        )

    plies = arrays["plies"]
    game_indices = arrays["game_indices"]
    prev_game: int | None = None
    prev_ply = -1
    for i in range(n):
        gid = int(game_indices[i])
        ply = int(plies[i])
        if gid != prev_game:
            prev_game = gid
            prev_ply = ply
            continue
        if ply < prev_ply:
            raise ShardInvariantError(
                f"sample {i}: ply {ply} < previous ply {prev_ply} within game index {gid} "
                "(plies must be non-decreasing within a game)"
            )
        prev_ply = ply

    num_aux = len(game.value_targets.aux_names)
    if num_aux and "aux" not in arrays:
        raise ShardInvariantError(
            f"{type(game).__name__} declares {num_aux} aux head(s) but the shard has no 'aux' array"
        )
    if not num_aux and "aux" in arrays:
        raise ShardInvariantError(
            f"{type(game).__name__} declares no aux heads but the shard has an 'aux' array "
            "(aux must be materialized only when declared, never zero-filled)"
        )
    if num_aux and arrays["aux"].shape != (n, num_aux):
        raise ShardInvariantError(f"aux shape {arrays['aux'].shape} != expected ({n}, {num_aux})")


# --- shard-level write / read -------------------------------------------------


def write_shard(
    path: Path,
    game: Game,
    records: Sequence[SampleRecord],
    *,
    run_id: str,
    actor_id: str,
    seq: int,
) -> None:
    """Write ``records`` to ``path`` as one shard, atomically.

    Every array invariant (:func:`_validate_arrays`) is checked *before* any
    byte is written, and every record's ``game_id`` must agree with
    ``(run_id, actor_id)`` -- both are fail-loud, not defensive-only-on-read
    checks.

    Args:
        path: The destination shard path (typically
            ``shard_dir / shard_filename(run_id, actor_id, seq)``, but this
            function does not itself enforce the naming convention -- that is
            :class:`ShardWriter`'s job).
        game: The adapter the samples came from; its declared surface fixes
            the fingerprint and the array shapes.
        records: The samples to store, in order. Must be non-empty.
        run_id: The run identity every record's ``game_id`` must carry.
        actor_id: The actor identity every record's ``game_id`` must carry.
        seq: The shard's sequence number, stored in the header.

    Raises:
        ValueError: If ``records`` is empty, or a record's ``game_id`` does
            not start with ``(run_id, actor_id)``.
        ShardInvariantError: If the packed arrays violate a pinned invariant.
    """
    if not records:
        raise ValueError("cannot write an empty shard")
    for i, r in enumerate(records):
        if r.game_id[0] != run_id or r.game_id[1] != actor_id:
            raise ValueError(
                f"record {i}: game_id {r.game_id} does not match shard identity "
                f"(run_id={run_id!r}, actor_id={actor_id!r})"
            )

    arrays = _pack_arrays(game, records)
    _validate_arrays(game, arrays)

    fingerprint = build_fingerprint(game)
    header = {
        "fingerprint_json": np.asarray(canonical_json(fingerprint)),
        "run_id": np.asarray(str(run_id)),
        "actor_id": np.asarray(str(actor_id)),
        "seq": np.asarray(int(seq), dtype=np.int64),
    }

    def write_body(fh: BinaryIO) -> None:
        np.savez(fh, **header, **arrays)

    _atomic_write(path, write_body)


def read_shard(path: Path, game: Game) -> ShardData:
    """Read and fully validate one shard file.

    Order of checks (each fails loudly, never a warning): the stored
    fingerprint is compared against the live one built from ``game`` first
    (a schema-version or identity mismatch is reported before anything else
    is trusted); then every array invariant is checked; only then are
    :class:`SampleRecord`\\ s reconstructed and returned.

    Args:
        path: The shard file to read.
        game: The adapter to validate the shard against (its live fingerprint
            and declared shapes).

    Returns:
        The shard's identity and samples.

    Raises:
        core.artifact_fingerprint.FingerprintMismatchError: If the stored
            fingerprint disagrees with ``game``'s live one on any field,
            including an unsupported schema version.
        ShardInvariantError: If the arrays violate a pinned invariant.
    """
    with np.load(path, allow_pickle=False) as npz:
        stored_fingerprint = json.loads(str(npz["fingerprint_json"]))
        compare_fingerprints(stored_fingerprint, build_fingerprint(game))

        run_id = str(npz["run_id"])
        actor_id = str(npz["actor_id"])
        seq = int(npz["seq"])
        array_keys = (
            "planes",
            "pi_action_ids",
            "pi_visit_counts",
            "pi_offsets",
            "z",
            "movers",
            "model_versions",
            "plies",
            "game_indices",
        )
        arrays = {k: npz[k] for k in array_keys}
        if "aux" in npz.files:
            arrays["aux"] = npz["aux"]

    _validate_arrays(game, arrays)
    records = _unpack_records(game, run_id, actor_id, arrays)
    return ShardData(
        run_id=run_id,
        actor_id=actor_id,
        seq=seq,
        fingerprint=stored_fingerprint,
        records=records,
    )


# --- crash-safe per-actor writer state ---------------------------------------


@dataclass(frozen=True)
class WriterState:
    """A :class:`ShardWriter`'s persisted counters.

    Attributes:
        next_shard_seq: The sequence number the *next* published shard will
            use.
        next_game_index: The durable game index the *next* new game will be
            assigned.
    """

    next_shard_seq: int = 0
    next_game_index: int = 0


def writer_state_path(shard_dir: Path, run_id: str, actor_id: str) -> Path:
    """Return the persisted state-file path for one ``(run_id, actor_id)``.

    Args:
        shard_dir: The directory shards (and their state files) live in.
        run_id: The run identity.
        actor_id: The actor identity.

    Returns:
        ``shard_dir / "writer-state-<run_id>-<actor_id>.json"``.
    """
    return shard_dir / f"writer-state-{run_id}-{actor_id}.json"


def _load_writer_state(path: Path) -> WriterState:
    """Load a persisted :class:`WriterState`, or the zero state if absent.

    Args:
        path: The state file path (:func:`writer_state_path`).

    Returns:
        The persisted state, or ``WriterState()`` if ``path`` does not exist
        yet (a brand-new actor).
    """
    if not path.exists():
        return WriterState()
    payload = json.loads(path.read_text())
    return WriterState(
        next_shard_seq=payload["next_shard_seq"],
        next_game_index=payload["next_game_index"],
    )


class ShardWriter:
    """Crash-safe per-actor shard publisher (design doc §12 M3).

    Owns one actor's durable identity within one run: the persisted
    ``(next_shard_seq, next_game_index)`` counters (:class:`WriterState`) and
    the shard-naming convention. :meth:`write_shard` is the whole public
    surface -- it reserves and durably persists the counters' advance
    *before* publishing the shard file that consumes them
    (:meth:`_reserve` then :meth:`_publish`, exposed separately so the crash
    window between them is directly testable): a process that dies between
    the two has burned a sequence number and a range of game indices, but on
    restart a fresh :class:`ShardWriter` reloads the already-advanced state
    and can never reissue either.

    Args:
        shard_dir: Directory shards and the writer's state file are published
            into. Created if it does not exist.
        game: The adapter every shard is fingerprinted against.
        run_id: This run's identity (constant across every actor).
        actor_id: This actor's identity within the run (constant for the
            lifetime of one :class:`ShardWriter`; at most one writer may be
            live for a given ``(run_id, actor_id)`` at a time).
    """

    def __init__(self, shard_dir: Path, game: Game, run_id: str, actor_id: str):
        self.shard_dir = Path(shard_dir)
        self.game = game
        self.run_id = run_id
        self.actor_id = actor_id
        self.state_path = writer_state_path(self.shard_dir, run_id, actor_id)
        self.state = _load_writer_state(self.state_path)

    def _reserve(self, num_games: int) -> tuple[int, int]:
        """Advance and durably persist the state for one shard, before publish.

        Args:
            num_games: Number of new games this shard's samples belong to.

        Returns:
            ``(seq, game_index_start)``: the sequence number this shard will
            publish under, and the first durable game index it owns (games
            get ``game_index_start .. game_index_start + num_games - 1``).

        Raises:
            ValueError: If ``num_games`` is not positive.
        """
        if num_games <= 0:
            raise ValueError(f"num_games must be positive, got {num_games}")
        seq = self.state.next_shard_seq
        game_index_start = self.state.next_game_index
        new_state = WriterState(
            next_shard_seq=seq + 1,
            next_game_index=game_index_start + num_games,
        )
        _atomic_write_json(self.state_path, asdict(new_state))
        self.state = new_state
        return seq, game_index_start

    def _publish(
        self, seq: int, game_index_start: int, games: Sequence[Sequence[PendingSample]]
    ) -> Path:
        """Assign durable game ids and atomically publish one shard file.

        Args:
            seq: This shard's sequence number (from :meth:`_reserve`).
            game_index_start: The first durable game index this shard owns
                (from :meth:`_reserve`).
            games: One game's ordered pending samples per element, in the
                same order the game indices were reserved for.

        Returns:
            The published shard's path.
        """
        records = [
            SampleRecord.from_pending(pending, (self.run_id, self.actor_id, game_index_start + i))
            for i, game_samples in enumerate(games)
            for pending in game_samples
        ]
        path = self.shard_dir / shard_filename(self.run_id, self.actor_id, seq)
        write_shard(path, self.game, records, run_id=self.run_id, actor_id=self.actor_id, seq=seq)
        return path

    def write_shard(self, games: Sequence[Sequence[PendingSample]]) -> Path:
        """Reserve durable identity for ``games``, then publish them as one shard.

        Args:
            games: One game's ordered pending samples per element -- each
                element is a complete game (a shard never splits a game
                across files).

        Returns:
            The published shard's path.

        Raises:
            ValueError: If ``games`` is empty, or any game contributes zero
                samples.
        """
        if not games:
            raise ValueError("cannot write a shard with zero games")
        if any(not g for g in games):
            raise ValueError("every game in a shard must contribute at least one sample")
        seq, game_index_start = self._reserve(len(games))
        return self._publish(seq, game_index_start, games)
