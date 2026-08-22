"""The learner-side replay window: ``core/replay_window.py`` (§12 M3, issue #55).

Builds shard files directly with ``core.replay_shard.write_shard`` (no real
self-play), so every test controls exact per-shard position counts. Exercises
idempotent discovery, the durable ingestion manifest, position-uniform
sampling and its canonical index map, capacity-driven eviction, the
fingerprint/missing-file failure modes, seeded determinism, crash-resume
equivalence, and the two D5 replay-ratio totals.
"""

from __future__ import annotations

import json

import pytest

from core.artifact_fingerprint import FingerprintMismatchError, build_fingerprint, canonical_json
from core.replay_shard import SampleRecord, shard_filename, write_shard
from core.replay_window import (
    MANIFEST_SCHEMA_VERSION,
    STATUS_EVICTED,
    STATUS_LIVE,
    MissingShardFileError,
    ReplayManifestError,
    ReplayWindow,
    manifest_path,
    samples_drawn,
)
from core.seeding import LearnerRNGs
from games.tictactoe import TicTacToe

TTT = TicTacToe()


def _synthetic_records(run_id, actor_id, game_index, num_positions, model_version=1):
    """Build ``num_positions`` trivially-valid, cheaply-comparable TTT records.

    All samples belong to one synthetic game (``game_index``), with strictly
    increasing plies ``0..num_positions - 1`` -- the only structural
    requirement ``write_shard``'s invariants impose across a game. Real board
    reachability does not matter here: these tests exercise the window's
    manifest/sampling machinery, never Blokus/TTT rules.

    Args:
        run_id: The synthetic game's run id.
        actor_id: The synthetic game's actor id.
        game_index: The synthetic game's durable index.
        num_positions: Number of records to build.
        model_version: Stamped on every record -- a convenient, otherwise
            game-meaningless tag tests use to identify which shard a sampled
            record came from.

    Returns:
        A tuple of ``num_positions`` ``SampleRecord``s.
    """
    state = TTT.initial_state()
    planes = TTT.encode_state(state)
    legal_action = next(iter(TTT.legal_moves(state)))
    return tuple(
        SampleRecord(
            planes=planes,
            sparse_pi=((legal_action, 1),),
            z=0.0,
            aux=(),
            mover=0,
            model_version=model_version,
            ply=ply,
            game_id=(run_id, actor_id, game_index),
        )
        for ply in range(num_positions)
    )


def _write_shard(shard_dir, run_id, actor_id, seq, num_positions, model_version=1):
    """Write one synthetic shard of exactly ``num_positions`` records to disk.

    Returns:
        The shard's filename (``shard_id``).
    """
    records = _synthetic_records(run_id, actor_id, seq, num_positions, model_version)
    shard_id = shard_filename(run_id, actor_id, seq)
    write_shard(shard_dir / shard_id, TTT, records, run_id=run_id, actor_id=actor_id, seq=seq)
    return shard_id


# --- idempotent rescan / durable manifest -------------------------------------


def test_rescan_ingests_new_shards_in_sorted_filename_order(tmp_path):
    # Written out of alphabetical order; discovery must not depend on that.
    id_z = _write_shard(tmp_path, "run", "z-actor", 0, 3)
    id_a = _write_shard(tmp_path, "run", "a-actor", 0, 5)

    window = ReplayWindow(tmp_path, TTT, capacity=1000)
    result = window.rescan()

    assert result.ingested_shard_ids == (id_a, id_z)
    assert [e.shard_id for e in window.shard_entries] == [id_a, id_z]
    assert window.positions_stored == 8


def test_rescan_twice_with_no_new_shards_is_a_no_op(tmp_path):
    _write_shard(tmp_path, "run", "actor", 0, 4)
    window = ReplayWindow(tmp_path, TTT, capacity=1000)

    first = window.rescan()
    manifest_after_first = window.shard_entries
    second = window.rescan()

    assert first.ingested_shard_ids != ()
    assert second.ingested_shard_ids == ()
    assert second.evicted_shard_ids == ()
    assert window.shard_entries == manifest_after_first
    assert window.positions_stored == 4


def test_rescan_ingests_only_shards_absent_from_the_manifest(tmp_path):
    id_1 = _write_shard(tmp_path, "run", "actor", 0, 4)
    window = ReplayWindow(tmp_path, TTT, capacity=1000)
    window.rescan()

    id_2 = _write_shard(tmp_path, "run", "actor", 1, 2)
    result = window.rescan()

    assert result.ingested_shard_ids == (id_2,)
    assert [e.shard_id for e in window.shard_entries] == [id_1, id_2]
    assert window.positions_stored == 6


def test_manifest_persists_across_fresh_instances(tmp_path):
    _write_shard(tmp_path, "run", "actor", 0, 4)
    _write_shard(tmp_path, "run", "actor", 1, 3)
    first = ReplayWindow(tmp_path, TTT, capacity=1000)
    first.rescan()

    reloaded = ReplayWindow(tmp_path, TTT, capacity=1000)
    assert reloaded.shard_entries == first.shard_entries
    assert reloaded.positions_stored == first.positions_stored == 7


def test_manifest_file_is_human_readable_json(tmp_path):
    shard_id = _write_shard(tmp_path, "run", "actor", 0, 4)
    window = ReplayWindow(tmp_path, TTT, capacity=1000)
    window.rescan()

    payload = json.loads(manifest_path(tmp_path).read_text())
    assert payload["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert payload["positions_stored_total"] == 4
    assert payload["shards"] == [
        {
            "shard_id": shard_id,
            "run_id": "run",
            "actor_id": "actor",
            "seq": 0,
            "num_positions": 4,
            "status": STATUS_LIVE,
        }
    ]


def test_manifest_schema_version_mismatch_rejected(tmp_path):
    _write_shard(tmp_path, "run", "actor", 0, 4)
    window = ReplayWindow(tmp_path, TTT, capacity=1000)
    window.rescan()

    payload = json.loads(manifest_path(tmp_path).read_text())
    payload["schema_version"] = 999
    manifest_path(tmp_path).write_text(json.dumps(payload))

    with pytest.raises(ReplayManifestError, match="schema_version"):
        ReplayWindow(tmp_path, TTT, capacity=1000)


# --- position-uniform sampling / canonical index map --------------------------


def test_locate_maps_indices_to_shard_and_position_exactly(tmp_path):
    id_a = _write_shard(tmp_path, "run", "a", 0, 2)
    id_b = _write_shard(tmp_path, "run", "b", 0, 6)
    id_c = _write_shard(tmp_path, "run", "c", 0, 12)
    window = ReplayWindow(tmp_path, TTT, capacity=1000)
    window.rescan()

    expected = (
        [(id_a, i) for i in range(2)]
        + [(id_b, i) for i in range(6)]
        + [(id_c, i) for i in range(12)]
    )
    for index, (shard_id, position) in enumerate(expected):
        entry, resolved_position = window._locate(index)
        assert entry.shard_id == shard_id
        assert resolved_position == position

    with pytest.raises(IndexError):
        window._locate(20)
    with pytest.raises(IndexError):
        window._locate(-1)


def test_sample_batch_draws_are_position_weighted_not_shard_uniform(tmp_path):
    _write_shard(tmp_path, "run", "small", 0, 2, model_version=100)
    _write_shard(tmp_path, "run", "medium", 0, 6, model_version=200)
    _write_shard(tmp_path, "run", "large", 0, 12, model_version=300)
    window = ReplayWindow(tmp_path, TTT, capacity=1000)
    window.rescan()

    n = 20_000
    batch = window.sample_batch(run_seed=7, step=0, batch_size=n)
    counts = {100: 0, 200: 0, 300: 0}
    for record in batch:
        counts[record.model_version] += 1

    # Position-uniform over 20 total positions (2/6/12): expected shares are
    # 0.1 / 0.3 / 0.6. Shard-uniform (the bug this guards against) would give
    # ~0.333 each -- a 2-percentage-point tolerance cleanly separates the two
    # while staying far outside sampling noise at n=20,000 (sigma <= ~0.35pp).
    for model_version, expected_share in ((100, 0.1), (200, 0.3), (300, 0.6)):
        observed_share = counts[model_version] / n
        assert abs(observed_share - expected_share) < 0.02, (model_version, counts)


def test_sample_batch_returns_records_in_draw_order_not_shard_order(tmp_path):
    # Interleaving-friendly sizes: many small, equal-sized shards, so a batch
    # is very likely to touch shards out of manifest order across its draws.
    model_version_by_shard = {}
    for i in range(5):
        shard_id = _write_shard(tmp_path, "run", f"actor{i}", 0, 3, model_version=i)
        model_version_by_shard[shard_id] = i
    window = ReplayWindow(tmp_path, TTT, capacity=1000)
    window.rescan()

    batch = window.sample_batch(run_seed=99, step=0, batch_size=50)

    live, offsets = window._partition()
    total = offsets[-1] + live[-1].num_positions
    rng = LearnerRNGs.for_step(99, 0).window_sampling
    drawn_indices = [rng.randrange(total) for _ in range(50)]
    expected_model_versions = [
        model_version_by_shard[window._locate_at(live, offsets, idx)[0].shard_id]
        for idx in drawn_indices
    ]

    assert [r.model_version for r in batch] == expected_model_versions
    # Genuinely interleaved, not grouped by shard -- guards against an
    # implementation that decodes shard-by-shard and concatenates.
    assert expected_model_versions != sorted(expected_model_versions)


def test_sample_batch_empty_window_raises(tmp_path):
    window = ReplayWindow(tmp_path, TTT, capacity=1000)
    window.rescan()
    with pytest.raises(IndexError, match="empty"):
        window.sample_batch(run_seed=1, step=0, batch_size=4)


def test_sample_batch_rejects_non_positive_batch_size(tmp_path):
    _write_shard(tmp_path, "run", "actor", 0, 4)
    window = ReplayWindow(tmp_path, TTT, capacity=1000)
    window.rescan()
    with pytest.raises(ValueError, match="batch_size"):
        window.sample_batch(run_seed=1, step=0, batch_size=0)


# --- eviction -------------------------------------------------------------------


def test_eviction_removes_whole_oldest_live_shards_until_at_capacity(tmp_path):
    window = ReplayWindow(tmp_path, TTT, capacity=10)

    id_a = _write_shard(tmp_path, "run", "a", 0, 4)
    r1 = window.rescan()
    assert r1.evicted_shard_ids == ()
    assert window.live_positions == 4

    id_b = _write_shard(tmp_path, "run", "b", 0, 4)
    r2 = window.rescan()
    assert r2.evicted_shard_ids == ()
    assert window.live_positions == 8

    _write_shard(tmp_path, "run", "c", 0, 4)
    r3 = window.rescan()
    assert r3.evicted_shard_ids == (id_a,)
    assert window.live_positions == 8
    assert window.positions_stored == 12

    statuses = {e.shard_id: e.status for e in window.shard_entries}
    assert statuses[id_a] == STATUS_EVICTED
    assert statuses[id_b] == STATUS_LIVE
    assert not (tmp_path / id_a).exists()
    assert (tmp_path / id_b).exists()

    _write_shard(tmp_path, "run", "d", 0, 3)
    r4 = window.rescan()
    assert r4.evicted_shard_ids == (id_b,)
    assert window.live_positions == 7
    assert window.positions_stored == 15


def test_eviction_of_multiple_shards_in_a_single_rescan(tmp_path):
    _write_shard(tmp_path, "run", "a", 0, 3)
    _write_shard(tmp_path, "run", "b", 0, 3)
    id_c = _write_shard(tmp_path, "run", "c", 0, 3)
    window = ReplayWindow(tmp_path, TTT, capacity=5)

    result = window.rescan()

    # Ingested a,b,c (9 positions) in one call, over capacity 5: evicts a
    # (9-3=6, still over), then b (6-3=3, at/under capacity); c survives.
    assert result.evicted_shard_ids == (
        shard_filename("run", "a", 0),
        shard_filename("run", "b", 0),
    )
    assert window.live_positions == 3
    assert [e.shard_id for e in window.shard_entries if e.status == STATUS_LIVE] == [id_c]


def test_eviction_can_empty_the_window_when_a_single_shard_exceeds_capacity(tmp_path):
    id_a = _write_shard(tmp_path, "run", "a", 0, 20)
    window = ReplayWindow(tmp_path, TTT, capacity=5)

    result = window.rescan()

    assert result.evicted_shard_ids == (id_a,)
    assert window.live_positions == 0
    assert window.positions_stored == 20


def test_positions_stored_never_decreases_across_evictions(tmp_path):
    window = ReplayWindow(tmp_path, TTT, capacity=6)
    totals = []
    for i in range(6):
        _write_shard(tmp_path, "run", f"actor{i}", 0, 3)
        window.rescan()
        totals.append(window.positions_stored)

    assert totals == sorted(totals)
    assert totals[-1] == 18


def test_eviction_sweep_is_retry_safe_across_fresh_instances(tmp_path):
    id_a = _write_shard(tmp_path, "run", "a", 0, 4)
    window = ReplayWindow(tmp_path, TTT, capacity=4)
    window.rescan()

    _write_shard(tmp_path, "run", "b", 0, 4)
    window.rescan()  # evicts a; deletes its file

    assert not (tmp_path / id_a).exists()

    # Simulate "crash before delete": recreate the evicted file out of band,
    # as if a prior process's mark-evicted commit landed but the unlink did
    # not. A fresh instance's rescan must retry the delete.
    (tmp_path / id_a).write_bytes(b"stale-bytes-should-be-removed")
    fresh = ReplayWindow(tmp_path, TTT, capacity=4)
    fresh.rescan()

    assert not (tmp_path / id_a).exists()


# --- fingerprint / missing-file failure modes ------------------------------------


def test_rescan_rejects_unsupported_shard_fingerprint_schema_version(tmp_path):
    import numpy as np

    shard_id = _write_shard(tmp_path, "run", "actor", 0, 3)
    path = tmp_path / shard_id
    with np.load(path) as npz:
        arrays = dict(npz.items())
    fingerprint = build_fingerprint(TTT)
    fingerprint["schema_version"] = 999
    arrays["fingerprint_json"] = np.asarray(canonical_json(fingerprint))
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)

    window = ReplayWindow(tmp_path, TTT, capacity=1000)
    with pytest.raises(FingerprintMismatchError, match="schema_version"):
        window.rescan()

    # All-or-nothing: the bad shard never entered the manifest.
    assert window.shard_entries == ()
    assert window.positions_stored == 0
    assert not manifest_path(tmp_path).exists()


def test_rescan_rejects_wrong_adapter_fingerprint(tmp_path):
    from games.blokus_duo import BlokusDuo
    from games.blokus_duo.config import MICRO_CONFIG

    micro = BlokusDuo(config=MICRO_CONFIG)
    state = micro.initial_state()
    planes = micro.encode_state(state)
    legal_action = next(iter(micro.legal_moves(state)))
    record = SampleRecord(
        planes=planes,
        sparse_pi=((legal_action, 1),),
        z=0.0,
        aux=(0.0,),
        mover=0,
        model_version=1,
        ply=0,
        game_id=("run", "actor", 0),
    )
    shard_id = shard_filename("run", "actor", 0)
    write_shard(tmp_path / shard_id, micro, [record], run_id="run", actor_id="actor", seq=0)

    window = ReplayWindow(tmp_path, TTT, capacity=1000)  # wrong adapter on purpose
    with pytest.raises(FingerprintMismatchError):
        window.rescan()
    assert window.shard_entries == ()


def test_fingerprint_and_invariant_validation_runs_only_at_ingest(tmp_path, monkeypatch):
    """The checked reader (fingerprint + invariants) never re-runs post-ingest.

    Forces cache churn (``decoded_cache_size=1`` against two live shards) so
    every sampled batch below is guaranteed to decode a shard at least once
    from disk; if sampling silently re-validated on each such reload, the
    patched ``read_shard`` would be called and the assertion below would
    fail loudly.
    """
    import core.replay_window as replay_window_module

    _write_shard(tmp_path, "run", "a", 0, 3)
    _write_shard(tmp_path, "run", "b", 0, 3)
    window = ReplayWindow(tmp_path, TTT, capacity=1000, decoded_cache_size=1)
    window.rescan()  # the one and only place read_shard should ever run

    call_count = {"n": 0}

    def counting_read_shard(*args, **kwargs):
        call_count["n"] += 1
        raise AssertionError("read_shard (checked reader) must not run after ingest")

    monkeypatch.setattr(replay_window_module, "read_shard", counting_read_shard)

    for step in range(6):
        window.sample_batch(run_seed=1, step=step, batch_size=4)

    assert call_count["n"] == 0


def test_sample_batch_raises_on_a_live_shard_with_a_missing_file(tmp_path):
    shard_id = _write_shard(tmp_path, "run", "actor", 0, 4)
    window = ReplayWindow(tmp_path, TTT, capacity=1000)
    window.rescan()

    (tmp_path / shard_id).unlink()  # simulate out-of-band tampering

    with pytest.raises(MissingShardFileError, match=shard_id):
        window.sample_batch(run_seed=1, step=0, batch_size=1)


# --- determinism ------------------------------------------------------------------


def test_sample_batch_is_deterministic_given_the_same_run_seed_step_and_manifest(tmp_path):
    _write_shard(tmp_path, "run", "a", 0, 5)
    _write_shard(tmp_path, "run", "b", 0, 7)
    window_1 = ReplayWindow(tmp_path, TTT, capacity=1000)
    window_1.rescan()
    window_2 = ReplayWindow(tmp_path, TTT, capacity=1000)
    window_2.rescan()

    batch_1 = window_1.sample_batch(run_seed=123, step=4, batch_size=9)
    batch_2 = window_2.sample_batch(run_seed=123, step=4, batch_size=9)

    assert [(r.game_id, r.ply, r.model_version) for r in batch_1] == [
        (r.game_id, r.ply, r.model_version) for r in batch_2
    ]


def test_sample_batch_differs_across_steps(tmp_path):
    _write_shard(tmp_path, "run", "a", 0, 5)
    _write_shard(tmp_path, "run", "b", 0, 7)
    window = ReplayWindow(tmp_path, TTT, capacity=1000)
    window.rescan()

    batch_step_0 = window.sample_batch(run_seed=123, step=0, batch_size=10)
    batch_step_1 = window.sample_batch(run_seed=123, step=1, batch_size=10)

    key = lambda batch: [(r.game_id, r.ply) for r in batch]  # noqa: E731
    assert key(batch_step_0) != key(batch_step_1)


# --- crash-resume equivalence -----------------------------------------------------


def _timeline_step(shard_dir, window, step, new_shard_size, run_seed, batch_size):
    """Write one new shard, rescan, and draw one seeded batch for ``step``.

    Returns:
        ``(evicted_shard_ids, batch_key)`` -- ``batch_key`` is a
        content-comparable projection of the drawn batch (avoids comparing
        ``SampleRecord`` instances directly, since ``planes`` is decoded as
        an ndarray and dataclass equality on an ndarray field raises).
    """
    _write_shard(shard_dir, "run", f"actor{step}", 0, new_shard_size)
    result = window.rescan()
    batch = window.sample_batch(run_seed=run_seed, step=step, batch_size=batch_size)
    batch_key = tuple((r.game_id, r.ply, r.model_version) for r in batch)
    return result.evicted_shard_ids, batch_key


def test_crash_resume_equivalence_evictions_and_batches(tmp_path):
    run_seed = 4242
    batch_size = 4
    capacity = 10
    sizes = [4, 4, 4, 5]  # forces eviction from step 2 onward

    dir_a = tmp_path / "continuous"
    dir_a.mkdir()
    window_a = ReplayWindow(dir_a, TTT, capacity=capacity)
    continuous = [
        _timeline_step(dir_a, window_a, step, size, run_seed, batch_size)
        for step, size in enumerate(sizes)
    ]

    dir_b = tmp_path / "resumed"
    dir_b.mkdir()
    window_b = ReplayWindow(dir_b, TTT, capacity=capacity)
    resumed = [
        _timeline_step(dir_b, window_b, step, size, run_seed, batch_size)
        for step, size in enumerate(sizes[:2])
    ]
    # "Crash": drop window_b, resume from the persisted manifest.
    window_b_resumed = ReplayWindow(dir_b, TTT, capacity=capacity)
    resumed += [
        _timeline_step(dir_b, window_b_resumed, step, size, run_seed, batch_size)
        for step, size in zip(range(2, 4), sizes[2:], strict=True)
    ]

    assert resumed == continuous
    assert window_b_resumed.positions_stored == window_a.positions_stored
    assert window_b_resumed.live_positions == window_a.live_positions
    assert [e.status for e in window_b_resumed.shard_entries] == [
        e.status for e in window_a.shard_entries
    ]


# --- totals -------------------------------------------------------------------------


def test_positions_stored_reflects_the_manifest_running_total(tmp_path):
    _write_shard(tmp_path, "run", "a", 0, 4)
    _write_shard(tmp_path, "run", "b", 0, 6)
    window = ReplayWindow(tmp_path, TTT, capacity=1000)
    window.rescan()
    assert window.positions_stored == 10


def test_samples_drawn_is_step_times_batch_size():
    assert samples_drawn(0, 256) == 0
    assert samples_drawn(1, 256) == 256
    assert samples_drawn(37, 256) == 37 * 256


def test_samples_drawn_rejects_bad_inputs():
    with pytest.raises(ValueError, match="learner_step"):
        samples_drawn(-1, 256)
    with pytest.raises(ValueError, match="batch_size"):
        samples_drawn(1, 0)


# --- constructor validation -----------------------------------------------------------


def test_replay_window_rejects_non_positive_capacity(tmp_path):
    with pytest.raises(ValueError, match="capacity"):
        ReplayWindow(tmp_path, TTT, capacity=0)


def test_replay_window_rejects_non_positive_decoded_cache_size(tmp_path):
    with pytest.raises(ValueError, match="decoded_cache_size"):
        ReplayWindow(tmp_path, TTT, decoded_cache_size=0)
