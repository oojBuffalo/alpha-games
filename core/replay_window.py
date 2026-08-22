"""The learner-side 250k-position replay window (design doc §5/§6, D5, §12 M3).

Sits on top of the on-disk shard artifact (``core.replay_shard``): a ring
window over shard *files*, not over decoded positions. Two ideas make a
learner that crashes and resumes indistinguishable from one that never did.

**A durable ingestion manifest, never filesystem enumeration.** A directory
listing carries no order (``Path.glob``/``os.listdir`` order is unspecified
and platform-dependent) and no memory of what has already been consumed. The
manifest (:class:`ReplayManifest`) is the single source of truth for "what
shards are in the window, in what order, and are they still live": one JSON
file, rewritten wholesale via the same temp-name-then-``os.replace`` pattern
``core.replay_shard.WriterState`` uses (reusing its ``_atomic_write_json``
verbatim), never edited incrementally. A rescan (:meth:`ReplayWindow.rescan`)
is therefore idempotent by construction: it re-lists the directory, keeps
only the filenames absent from the manifest, reads and validates *those* in
sorted filename order (so two rescans racing the same directory always
resolve discovery-batch ties the same way), and commits all of them to the
manifest in one atomic write, or — on any validation failure — commits none.
Ingestion is all-or-nothing per rescan, so a retried rescan after a fixed
corrupt shard starts from exactly the state a first attempt would have left
behind, and never double-counts a shard already in the manifest.

**Fingerprint/invariant validation runs exactly once per shard, at ingest.**
``core.replay_shard.read_shard`` both compares the stored fingerprint
against the adapter's live one and checks every array invariant; running it
again on every later sampling access would re-pay that cost on the hot path
for a file that provably cannot have changed since — shards publish once,
atomically, and are never rewritten (``core.replay_shard``'s own contract).
So ingest is the *only* place the checked reader runs; every later access
goes through :func:`_read_shard_records_unchecked`, which reuses
``core.replay_shard._unpack_records`` on the raw arrays with no fingerprint
or invariant re-check. A live shard whose backing file has since vanished
(filesystem tampering, not a path this module itself takes) is a loud
:class:`MissingShardFileError` at that first post-ingest access — never a
silent skip.

**Position-uniform sampling via a canonical index map, never shard-uniform.**
The live shards, in manifest order, partition ``[0, live_positions)`` into
contiguous ranges sized by each shard's position count
(:meth:`ReplayWindow._partition` / :meth:`ReplayWindow._locate`); drawing an
index uniformly from that range and mapping it back *is* weighting by
position count, with no separate weighted-choice machinery needed. The seed
is the durable ``("learner", step, "replay-sampling")`` stream
(``core.seeding.LearnerRNGs``), so :meth:`ReplayWindow.sample_batch` is a
pure function of ``(run_seed, step, batch_size)`` and the manifest's current
content — independent of which process, or how many prior crashes, produced
that content.

**Bounded memory via a per-shard decode LRU.** Positions are never held
resident for the whole window (250k full-Blokus positions would be tens of
GB — the same blowup Invariant 3 exists to avoid at the per-node level, and
D5's window exists to avoid here). A batch instead resolves each drawn index
to ``(shard, position)``, decodes the owning shard on a cache miss, and
evicts the least-recently-touched *decoded* shard once
``decoded_cache_size`` is exceeded.

**Eviction removes whole oldest live shards, mark-then-delete.** Exceeding
``capacity`` (D5: 250,000 by default) evicts live shards from the front of
manifest order — oldest first — one whole shard at a time, until the live
total is back at or under capacity. Marking evicted shards is a single
atomic manifest rewrite (a crash before it leaves every shard still live,
retried in full on the next rescan); deleting the underlying files happens
only after that mark is durable and is safe to retry on its own — every
rescan sweeps evicted entries with ``Path.unlink(missing_ok=True)``, so a
crash between mark and delete just leaves a dangling file a later rescan
cleans up, never a live shard whose file has already been removed. (The
sweep is memoized in-process so a long-lived instance re-checks each evicted
shard's file at most once, not every rescan; a freshly constructed instance
always re-sweeps everything once, which is exactly the retry a resumed
process needs.)

The manifest's ``positions_stored_total`` counts only ingestion, never
eviction, so it is monotone by construction — the running numerator the D5
replay ratio's "positions stored" side wants (design doc §7). The
denominator, "samples drawn," needs no counter at all: it is
``learner_step * batch_size`` by construction (:func:`samples_drawn`), since
every learner step draws exactly one seeded batch of a fixed size. Both
totals are exposed here; enforcing the [2, 4] band on their ratio is the
learner's job (issue #60), not this module's.
"""

from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from core.game import Game
from core.replay_shard import SampleRecord, _atomic_write_json, _unpack_records, read_shard
from core.seeding import LearnerRNGs

# D5 (§7, §10): "replay ~250k positions". A constructor default, not a
# hardcoded ceiling -- micro-Blokus configs pass a smaller value (M2.5).
DEFAULT_CAPACITY = 250_000

# How many decoded shards :class:`ReplayWindow` keeps resident at once (LRU).
# Small on purpose: the bounded-memory load strategy's whole point is that a
# batch never requires the full window decoded, only the handful of shards
# its drawn indices happen to land in.
DEFAULT_DECODED_CACHE_SIZE = 8

# This module's own manifest-file shape version -- independent of
# core.artifact_fingerprint.SCHEMA_VERSION (that one versions a *shard's*
# fingerprint payload; this one versions the manifest JSON's own field
# layout). Bumped only when the manifest schema itself changes shape.
MANIFEST_SCHEMA_VERSION = 1

STATUS_LIVE = "live"
STATUS_EVICTED = "evicted"

_MANIFEST_FILENAME = "replay-manifest.json"
# Matches core.replay_shard.shard_filename's exact prefix/suffix; excludes
# writer-state JSON files and in-flight ``*.tmp-<uuid>`` publish targets (the
# latter never end in ``.npz`` -- see core.replay_shard._atomic_write).
_SHARD_GLOB = "shard-*.npz"
# The shard header fields (core.replay_shard.write_shard) -- everything else
# in an ``.npz`` is a body array consumed by ``_unpack_records``.
_HEADER_ARRAY_KEYS = frozenset({"fingerprint_json", "run_id", "actor_id", "seq"})


class ReplayManifestError(Exception):
    """Raised when a persisted manifest file is malformed or unsupported.

    Covers an unrecognized ``schema_version`` (this module's own JSON shape,
    distinct from a shard's fingerprint schema version) and a structurally
    broken payload. Never a warning: a manifest this code cannot parse
    correctly can never be trusted to describe the window's true contents.
    """


class MissingShardFileError(Exception):
    """Raised when a manifest-live shard's backing file is gone at read time.

    Distinct from an ingest-time failure (:class:`FingerprintMismatchError`
    at ingest means the shard was never admitted to the manifest at all):
    this fires when a shard the manifest already trusts as live cannot be
    found on disk when a sampled index needs it decoded -- filesystem
    tampering or an out-of-band deletion, not a state this module's own
    mark-then-delete eviction protocol ever produces (an evicted shard is
    removed from the *live* partition before its file is unlinked).
    """


@dataclass(frozen=True)
class ShardEntry:
    """One manifest row: a shard's durable identity, size, and live status.

    Attributes:
        shard_id: The shard's filename (``core.replay_shard.shard_filename``)
            -- the manifest's durable key for this shard, and what
            :meth:`ReplayWindow.rescan` diffs directory listings against.
        run_id: The shard's header ``run_id`` (read from the shard itself at
            ingest, never parsed out of ``shard_id`` -- run/actor ids may
            themselves contain ``-``, so the filename is not safely
            decomposable).
        actor_id: The shard's header ``actor_id``.
        seq: The shard's header sequence number.
        num_positions: The shard's sample count, fixed at ingest.
        status: :data:`STATUS_LIVE` or :data:`STATUS_EVICTED`.
    """

    shard_id: str
    run_id: str
    actor_id: str
    seq: int
    num_positions: int
    status: str


@dataclass(frozen=True)
class ReplayManifest:
    """The durable ingestion manifest (design doc §12 M3 / issue #55).

    Attributes:
        schema_version: This manifest's own JSON-shape version
            (:data:`MANIFEST_SCHEMA_VERSION`).
        shards: Every ever-ingested shard, **in manifest order** -- the order
            entries were appended across every rescan this window (or an
            ancestor process sharing its directory) has ever run. This order
            is what "oldest" means for eviction; it is never re-derived from
            directory listings or timestamps.
        positions_stored_total: Sum of ``num_positions`` over every shard
            ever ingested, live or evicted -- monotone, since eviction never
            touches it.
    """

    schema_version: int
    shards: tuple[ShardEntry, ...]
    positions_stored_total: int


@dataclass(frozen=True)
class RescanResult:
    """One :meth:`ReplayWindow.rescan` call's effect on the manifest.

    Attributes:
        ingested_shard_ids: Newly-ingested shard filenames, in the sorted
            order they were ingested this call (empty if nothing new was on
            disk).
        evicted_shard_ids: Shard filenames evicted *by this call*, in
            manifest (oldest-live-first) order (empty if the window was
            already at or under capacity after ingestion).
    """

    ingested_shard_ids: tuple[str, ...]
    evicted_shard_ids: tuple[str, ...]


def manifest_path(shard_dir: Path) -> Path:
    """Return the persisted manifest path for one shard directory.

    Args:
        shard_dir: The directory shard files (and the manifest) live in.

    Returns:
        ``shard_dir / "replay-manifest.json"``.
    """
    return Path(shard_dir) / _MANIFEST_FILENAME


def samples_drawn(learner_step: int, batch_size: int) -> int:
    """Return the D5 replay-ratio denominator: samples drawn through a step.

    Derived, not separately counted (issue #55): a learner that has
    completed ``learner_step`` steps, each drawing one seeded batch of
    ``batch_size`` positions via :meth:`ReplayWindow.sample_batch`, has drawn
    exactly ``learner_step * batch_size`` samples in total -- no running
    counter can drift from this, because nothing else can change it.

    Args:
        learner_step: The number of learner steps completed so far.
        batch_size: The fixed per-step batch size.

    Returns:
        ``learner_step * batch_size``.

    Raises:
        ValueError: If ``learner_step`` is negative or ``batch_size`` is not
            positive.
    """
    if learner_step < 0:
        raise ValueError(f"learner_step must be >= 0, got {learner_step}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return learner_step * batch_size


def _empty_manifest() -> ReplayManifest:
    """Return the zero-state manifest a brand-new shard directory starts at."""
    return ReplayManifest(
        schema_version=MANIFEST_SCHEMA_VERSION, shards=(), positions_stored_total=0
    )


def _load_manifest(path: Path) -> ReplayManifest:
    """Load a persisted manifest, or the empty one if ``path`` doesn't exist yet.

    Args:
        path: The manifest file path (:func:`manifest_path`).

    Returns:
        The persisted manifest, or :func:`_empty_manifest` for a directory no
        rescan has ever touched.

    Raises:
        ReplayManifestError: If ``path`` exists but its ``schema_version``
            disagrees with :data:`MANIFEST_SCHEMA_VERSION`, or its payload is
            structurally malformed.
    """
    if not path.exists():
        return _empty_manifest()
    payload = json.loads(path.read_text())
    try:
        schema_version = payload["schema_version"]
        positions_stored_total = payload["positions_stored_total"]
        shards = tuple(ShardEntry(**row) for row in payload["shards"])
    except (KeyError, TypeError) as err:
        raise ReplayManifestError(f"malformed replay manifest at {path}: {err}") from err
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ReplayManifestError(
            f"replay manifest schema_version mismatch at {path}: "
            f"stored={schema_version!r} live={MANIFEST_SCHEMA_VERSION!r}"
        )
    return ReplayManifest(
        schema_version=schema_version,
        shards=shards,
        positions_stored_total=positions_stored_total,
    )


def _save_manifest(path: Path, manifest: ReplayManifest) -> None:
    """Persist ``manifest`` to ``path`` as one atomic, human-readable JSON write.

    Reuses ``core.replay_shard._atomic_write_json`` verbatim (the "same
    pattern as the #78 writer state" the M3 design constraint asks for) --
    temp-name-then-``os.replace``, so a reader never observes a
    partially-written manifest.

    Args:
        path: The manifest file path (:func:`manifest_path`).
        manifest: The manifest to persist.
    """
    payload = {
        "schema_version": manifest.schema_version,
        "positions_stored_total": manifest.positions_stored_total,
        "shards": [asdict(e) for e in manifest.shards],
    }
    _atomic_write_json(path, payload)


def _read_shard_records_unchecked(path: Path, game: Game) -> tuple[SampleRecord, ...]:
    """Decode a shard's records without re-running the fingerprint/invariant gate.

    See the module docstring's "validate exactly once per shard" rationale:
    the checked reader (``core.replay_shard.read_shard``) already ran for
    this exact file at ingest, and the file cannot have changed since (shard
    files publish once, atomically, and are never rewritten). Reuses
    ``core.replay_shard._unpack_records`` -- the same reconstruction the
    checked reader uses -- so this can never silently diverge from it.

    Args:
        path: The shard file to decode.
        game: The adapter (for the declared aux-head count).

    Returns:
        The shard's records, in on-disk order.

    Raises:
        FileNotFoundError: If ``path`` does not exist. Callers needing the
            dedicated :class:`MissingShardFileError` message check existence
            first (:meth:`ReplayWindow._get_decoded`).
    """
    with np.load(path, allow_pickle=False) as npz:
        run_id = str(npz["run_id"])
        actor_id = str(npz["actor_id"])
        arrays = {k: npz[k] for k in npz.files if k not in _HEADER_ARRAY_KEYS}
    return _unpack_records(game, run_id, actor_id, arrays)


class ReplayWindow:
    """The learner-side ring window over shard files (design doc §12 M3).

    Args:
        shard_dir: Directory shard files (and the manifest) live in -- shared
            with the actor-side ``core.replay_shard.ShardWriter``s that
            publish into it. Need not exist yet.
        game: The adapter every shard is read against; fixes the fingerprint
            gate and array shapes (``core.replay_shard.read_shard``).
        capacity: Maximum live positions the window holds (D5 default
            250,000; micro-Blokus configs pass a smaller value).
        decoded_cache_size: Maximum decoded shards held resident at once
            (LRU; :meth:`sample_batch`'s bounded-memory load strategy).

    Raises:
        ValueError: If ``capacity`` or ``decoded_cache_size`` is not
            positive.
        ReplayManifestError: If a persisted manifest already exists at
            ``shard_dir`` but this code does not understand its schema
            version.
    """

    def __init__(
        self,
        shard_dir: Path,
        game: Game,
        *,
        capacity: int = DEFAULT_CAPACITY,
        decoded_cache_size: int = DEFAULT_DECODED_CACHE_SIZE,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if decoded_cache_size <= 0:
            raise ValueError(f"decoded_cache_size must be positive, got {decoded_cache_size}")
        self.shard_dir = Path(shard_dir)
        self.game = game
        self.capacity = capacity
        self.decoded_cache_size = decoded_cache_size
        self._manifest_path = manifest_path(self.shard_dir)
        self._manifest = _load_manifest(self._manifest_path)
        self._decoded_cache: OrderedDict[str, tuple[SampleRecord, ...]] = OrderedDict()
        # In-process-only memo of evicted shards already confirmed deleted (or
        # confirmed already gone) -- lets a long-lived instance's later
        # rescans skip re-checking files a prior rescan (this instance) has
        # already handled, while a freshly constructed instance (the
        # crash-resume case) always re-sweeps every evicted entry once.
        self._deleted_shard_ids: set[str] = set()

    # --- manifest read-only views --------------------------------------------

    @property
    def shard_entries(self) -> tuple[ShardEntry, ...]:
        """Every ingested shard (live and evicted), in manifest order."""
        return self._manifest.shards

    @property
    def positions_stored(self) -> int:
        """Total positions ever ingested (monotone; unaffected by eviction)."""
        return self._manifest.positions_stored_total

    @property
    def live_positions(self) -> int:
        """Current live-window position count (drops when a shard is evicted)."""
        return sum(e.num_positions for e in self._manifest.shards if e.status == STATUS_LIVE)

    # --- discovery + eviction --------------------------------------------------

    def rescan(self) -> RescanResult:
        """Idempotently discover new shards and enforce ``capacity``, atomically.

        Discovery: lists ``shard_dir`` for ``shard-*.npz``, keeps only
        filenames absent from the manifest, and ingests them in sorted
        filename order. Every newly discovered shard is read with the
        checked reader (fingerprint + invariants) *before* any manifest
        write; only if every one of them validates does the manifest gain a
        single new atomic write appending all of them plus the summed
        position count. A validation failure partway through therefore
        commits nothing -- the next rescan re-attempts the exact same set of
        not-yet-ingested filenames.

        Eviction: after ingestion (whether or not anything new was found),
        evicts whole live shards from the front of manifest order until the
        live total is at or under ``capacity``, then sweeps every evicted
        entry's file with an idempotent, memoized delete.

        Returns:
            The shards ingested and evicted by this call.

        Raises:
            core.artifact_fingerprint.FingerprintMismatchError: If a newly
                discovered shard's stored fingerprint disagrees with
                ``game``'s live one (including an unsupported schema
                version) -- propagated uncaught from
                ``core.replay_shard.read_shard``.
            core.replay_shard.ShardInvariantError: Likewise, for an
                invariant violation in a newly discovered shard.
        """
        known_ids = {e.shard_id for e in self._manifest.shards}
        on_disk = sorted(p.name for p in self.shard_dir.glob(_SHARD_GLOB))
        new_ids = tuple(name for name in on_disk if name not in known_ids)

        if new_ids:
            new_entries = []
            added_positions = 0
            for shard_id in new_ids:
                data = read_shard(self.shard_dir / shard_id, self.game)
                num_positions = len(data.records)
                new_entries.append(
                    ShardEntry(
                        shard_id=shard_id,
                        run_id=data.run_id,
                        actor_id=data.actor_id,
                        seq=data.seq,
                        num_positions=num_positions,
                        status=STATUS_LIVE,
                    )
                )
                added_positions += num_positions
            self._manifest = ReplayManifest(
                schema_version=self._manifest.schema_version,
                shards=self._manifest.shards + tuple(new_entries),
                positions_stored_total=self._manifest.positions_stored_total + added_positions,
            )
            _save_manifest(self._manifest_path, self._manifest)

        evicted_ids = self._evict_to_capacity()
        return RescanResult(ingested_shard_ids=new_ids, evicted_shard_ids=evicted_ids)

    def _evict_to_capacity(self) -> tuple[str, ...]:
        """Evict whole oldest live shards until ``live_positions <= capacity``.

        Marking is one atomic manifest rewrite covering every shard evicted
        by this call; file deletion is swept separately afterward
        (:meth:`_sweep_evicted_files`), covering both the shards just marked
        and any earlier mark whose delete step never completed.

        Returns:
            Shard ids evicted by this call, oldest first (empty if the
            window was already at or under capacity).
        """
        live_total = self.live_positions
        evicted_now: list[str] = []
        if live_total > self.capacity:
            shards = list(self._manifest.shards)
            for i, entry in enumerate(shards):
                if live_total <= self.capacity:
                    break
                if entry.status != STATUS_LIVE:
                    continue
                shards[i] = replace(entry, status=STATUS_EVICTED)
                live_total -= entry.num_positions
                evicted_now.append(entry.shard_id)
            self._manifest = replace(self._manifest, shards=tuple(shards))
            _save_manifest(self._manifest_path, self._manifest)

        self._sweep_evicted_files()
        return tuple(evicted_now)

    def _sweep_evicted_files(self) -> None:
        """Delete every evicted entry's file, idempotently and at most once each.

        ``Path.unlink(missing_ok=True)`` makes a retry after a crash between
        mark-evicted and delete safe by construction: whether this call
        finds the file present (a prior process died before deleting it) or
        already gone (a prior process finished the delete, or this shard was
        never evicted by a process that crashed), the outcome converges to
        "gone" either way.
        """
        for entry in self._manifest.shards:
            if entry.status != STATUS_EVICTED or entry.shard_id in self._deleted_shard_ids:
                continue
            (self.shard_dir / entry.shard_id).unlink(missing_ok=True)
            self._deleted_shard_ids.add(entry.shard_id)
            self._decoded_cache.pop(entry.shard_id, None)

    # --- position-uniform sampling ----------------------------------------------

    def _partition(self) -> tuple[tuple[ShardEntry, ...], tuple[int, ...]]:
        """Return live shards (manifest order) and their cumulative start offsets.

        Returns:
            ``(live_entries, offsets)``: ``offsets[i]`` is the first
            canonical index owned by ``live_entries[i]``. Both empty if the
            window holds no live shards.
        """
        live = tuple(e for e in self._manifest.shards if e.status == STATUS_LIVE)
        offsets = []
        total = 0
        for entry in live:
            offsets.append(total)
            total += entry.num_positions
        return live, tuple(offsets)

    @staticmethod
    def _locate_at(
        live: tuple[ShardEntry, ...], offsets: tuple[int, ...], index: int
    ) -> tuple[ShardEntry, int]:
        """Map a canonical index to its owning shard and in-shard position.

        The live shards, in manifest order, partition ``[0,
        live_positions)`` into contiguous ranges sized by ``num_positions`` --
        this *is* the position-uniform, weighted-by-position-count sampling
        contract (never shard-uniform): the weighting is entirely implicit in
        each range's width.

        Args:
            live: Live shard entries in manifest order (:meth:`_partition`).
            offsets: Parallel cumulative start offsets (:meth:`_partition`).
            index: A candidate canonical index.

        Returns:
            ``(shard_entry, position_within_shard)``.

        Raises:
            IndexError: If ``index`` is negative, or outside
                ``[0, live_positions)``.
        """
        if index < 0 or not live:
            raise IndexError(f"index {index} out of range for an empty live window")
        i = bisect.bisect_right(offsets, index) - 1
        if i < 0 or index >= offsets[i] + live[i].num_positions:
            total = offsets[-1] + live[-1].num_positions
            raise IndexError(f"index {index} out of range for live_positions={total}")
        return live[i], index - offsets[i]

    def _locate(self, index: int) -> tuple[ShardEntry, int]:
        """Single-index convenience wrapper over :meth:`_partition`/:meth:`_locate_at`.

        Args:
            index: A candidate canonical index.

        Returns:
            ``(shard_entry, position_within_shard)``.

        Raises:
            IndexError: See :meth:`_locate_at`.
        """
        live, offsets = self._partition()
        return self._locate_at(live, offsets, index)

    def _get_decoded(self, shard_id: str) -> tuple[SampleRecord, ...]:
        """Return a shard's decoded records, via the LRU cache on a hit.

        Args:
            shard_id: The shard's manifest key / filename.

        Returns:
            The shard's records, in on-disk order.

        Raises:
            MissingShardFileError: If the shard's file is not present at
                ``shard_dir / shard_id``.
        """
        cached = self._decoded_cache.get(shard_id)
        if cached is not None:
            self._decoded_cache.move_to_end(shard_id)
            return cached
        path = self.shard_dir / shard_id
        if not path.exists():
            raise MissingShardFileError(
                f"live shard {shard_id!r} has no backing file at {path} "
                "(manifest says live, but the file is missing)"
            )
        records = _read_shard_records_unchecked(path, self.game)
        self._decoded_cache[shard_id] = records
        self._decoded_cache.move_to_end(shard_id)
        if len(self._decoded_cache) > self.decoded_cache_size:
            self._decoded_cache.popitem(last=False)
        return records

    def sample_batch(self, run_seed: int, step: int, batch_size: int) -> tuple[SampleRecord, ...]:
        """Draw one learner step's position-uniform-with-replacement batch.

        Seeded via ``core.seeding.LearnerRNGs.for_step(run_seed,
        step).window_sampling`` -- the durable ``("learner", step,
        "replay-sampling")`` stream -- so a resumed learner redraws exactly
        the batches an uninterrupted run would have for every step it
        repeats, and different steps draw independent batches.

        Args:
            run_seed: The run's recorded root seed.
            step: The learner step number (durable; the learner checkpoints
                it).
            batch_size: Number of positions to draw.

        Returns:
            ``batch_size`` records, **in draw order** (element *i* is the
            record the *i*-th drawn index resolved to) -- never grouped by
            shard, so the returned sequence itself is what a resumed
            learner's batch is compared against, element for element.

        Raises:
            ValueError: If ``batch_size`` is not positive.
            IndexError: If the live window is empty.
            MissingShardFileError: If a drawn index resolves to a live shard
                whose file is missing.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        live, offsets = self._partition()
        total = offsets[-1] + live[-1].num_positions if live else 0
        if total == 0:
            raise IndexError("cannot sample from an empty replay window")

        rng = LearnerRNGs.for_step(run_seed, step).window_sampling
        drawn_indices = [rng.randrange(total) for _ in range(batch_size)]

        records: list[SampleRecord] = []
        for index in drawn_indices:
            entry, position = self._locate_at(live, offsets, index)
            shard_records = self._get_decoded(entry.shard_id)
            records.append(shard_records[position])
        return tuple(records)
