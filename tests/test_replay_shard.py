"""The on-disk replay-shard artifact: ``core/replay_shard.py`` (§12 M3, issue #54).

Round-trips real per-adapter samples (Blokus micro/full, whose aux head is
declared, and Tic-Tac-Toe, whose aux head is not) through the shard file
format, then exercises every pinned invariant, the fingerprint gate, the
crash-safe durable-identity scheme, and atomic publishing.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from core.artifact_fingerprint import FingerprintMismatchError, build_fingerprint, canonical_json
from core.replay_shard import (
    PendingSample,
    SampleRecord,
    ShardInvariantError,
    ShardWriter,
    _atomic_write,
    read_shard,
    shard_filename,
    write_shard,
    writer_state_path,
)
from games.blokus_duo import BlokusDuo
from games.blokus_duo.config import FULL_CONFIG, MICRO_CONFIG
from games.tictactoe import TicTacToe

MICRO = BlokusDuo(config=MICRO_CONFIG)
FULL = BlokusDuo(config=FULL_CONFIG)
TTT = TicTacToe()

# Binary-exact-in-float32 values, so a round trip never loses a bit -- the
# "exact equality of all fields" round-trip tests below can compare
# read-back values against these literals directly.
_CLEAN_FLOATS = (0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.125, -0.125)


def _pending_samples(game, num_plies, seed, model_version):
    """Play ``num_plies`` random-legal moves, returning one ``PendingSample`` each.

    Args:
        game: The adapter to play.
        num_plies: Number of plies to record (stops early on a terminal).
        seed: Seed for move choice and the synthetic sparse-π counts.
        model_version: Stamped onto every sample.

    Returns:
        The recorded samples, in play order.
    """
    rng = random.Random(seed)
    state = game.initial_state()
    num_aux = len(game.value_targets.aux_names)
    samples = []
    for ply in range(num_plies):
        if game.is_terminal(state):
            break
        legal = list(game.legal_moves(state))
        counts = rng.sample(range(1, len(legal) + 1), len(legal))
        sparse_pi = tuple(zip(legal, counts, strict=True))
        z = _CLEAN_FLOATS[(seed + ply) % len(_CLEAN_FLOATS)]
        aux = tuple(
            _CLEAN_FLOATS[(seed + ply + h + 1) % len(_CLEAN_FLOATS)] for h in range(num_aux)
        )
        samples.append(
            PendingSample(
                planes=game.encode_state(state),
                sparse_pi=sparse_pi,
                z=z,
                aux=aux,
                mover=game.current_player(state),
                model_version=model_version,
                ply=ply,
            )
        )
        state = game.apply(state, rng.choice(legal))
    return samples


def _to_records(game_id, pending):
    return tuple(SampleRecord.from_pending(p, game_id) for p in pending)


# --- round trip --------------------------------------------------------------


@pytest.mark.parametrize(
    "game,game_ids",
    [
        (MICRO, (("run-a", "actor-0", 0), ("run-a", "actor-0", 1))),
        (FULL, (("run-b", "actor-1", 0), ("run-b", "actor-1", 1))),
        (TTT, (("run-c", "actor-2", 0), ("run-c", "actor-2", 1))),
    ],
    ids=["blokus-micro-with-aux", "blokus-full-with-aux", "tictactoe-without-aux"],
)
def test_round_trip_exact_equality(tmp_path, game, game_ids):
    game1 = _pending_samples(game, 4, seed=1, model_version=7)
    game2 = _pending_samples(game, 3, seed=2, model_version=7)
    records = _to_records(game_ids[0], game1) + _to_records(game_ids[1], game2)

    run_id, actor_id, _ = game_ids[0]
    path = tmp_path / shard_filename(run_id, actor_id, 0)
    write_shard(path, game, records, run_id=run_id, actor_id=actor_id, seq=0)
    data = read_shard(path, game)

    assert data.run_id == run_id
    assert data.actor_id == actor_id
    assert data.seq == 0
    assert data.fingerprint == build_fingerprint(game)
    assert len(data.records) == len(records)

    for original, read_back in zip(records, data.records, strict=True):
        assert np.array_equal(
            np.asarray(original.planes, dtype=np.float32), np.asarray(read_back.planes)
        )
        assert read_back.sparse_pi == original.sparse_pi
        assert read_back.z == original.z
        assert read_back.aux == original.aux
        assert read_back.mover == original.mover
        assert read_back.model_version == original.model_version
        assert read_back.ply == original.ply
        assert read_back.game_id == original.game_id


def test_round_trip_preserves_ply_order_and_game_grouping(tmp_path):
    game1 = _pending_samples(MICRO, 5, seed=3, model_version=1)
    game2 = _pending_samples(MICRO, 5, seed=4, model_version=1)
    records = _to_records(("r", "a", 0), game1) + _to_records(("r", "a", 1), game2)
    path = tmp_path / shard_filename("r", "a", 0)
    write_shard(path, MICRO, records, run_id="r", actor_id="a", seq=0)
    data = read_shard(path, MICRO)

    game0_plies = [r.ply for r in data.records if r.game_id == ("r", "a", 0)]
    game1_plies = [r.ply for r in data.records if r.game_id == ("r", "a", 1)]
    assert game0_plies == sorted(game0_plies)
    assert game1_plies == sorted(game1_plies)


# --- aux materialization -------------------------------------------------------


def test_aux_absent_from_npz_when_the_game_declares_none(tmp_path):
    records = _to_records(("r", "a", 0), _pending_samples(TTT, 3, seed=5, model_version=1))
    path = tmp_path / shard_filename("r", "a", 0)
    write_shard(path, TTT, records, run_id="r", actor_id="a", seq=0)
    with np.load(path) as npz:
        assert "aux" not in npz.files


def test_aux_present_and_shaped_when_the_game_declares_one(tmp_path):
    records = _to_records(("r", "a", 0), _pending_samples(MICRO, 3, seed=6, model_version=1))
    path = tmp_path / shard_filename("r", "a", 0)
    write_shard(path, MICRO, records, run_id="r", actor_id="a", seq=0)
    with np.load(path) as npz:
        assert "aux" in npz.files
        assert npz["aux"].shape == (len(records), 1)


def test_aux_width_mismatch_with_declared_heads_raises(tmp_path):
    bad = SampleRecord(
        planes=TTT.encode_state(TTT.initial_state()),
        sparse_pi=((0, 1),),
        z=0.0,
        aux=(0.5,),  # TTT declares zero aux heads
        mover=0,
        model_version=1,
        ply=0,
        game_id=("r", "a", 0),
    )
    with pytest.raises(ShardInvariantError, match="aux"):
        write_shard(tmp_path / "bad.npz", TTT, [bad], run_id="r", actor_id="a", seq=0)


# --- array invariants ----------------------------------------------------------


def _one_record(game, **overrides):
    state = game.initial_state()
    legal = list(game.legal_moves(state))
    num_aux = len(game.value_targets.aux_names)
    base = dict(
        planes=game.encode_state(state),
        sparse_pi=tuple((a, 1) for a in legal[:2]),
        z=0.0,
        aux=tuple(0.0 for _ in range(num_aux)),
        mover=game.current_player(state),
        model_version=1,
        ply=0,
        game_id=("r", "a", 0),
    )
    base.update(overrides)
    return SampleRecord(**base)


def test_out_of_range_action_id_raises(tmp_path):
    num_actions = 1
    for d in MICRO.policy_shape:
        num_actions *= d
    rec = _one_record(MICRO, sparse_pi=((num_actions, 1),))
    with pytest.raises(ShardInvariantError, match="out of range"):
        write_shard(tmp_path / "bad.npz", MICRO, [rec], run_id="r", actor_id="a", seq=0)


def test_negative_visit_count_raises(tmp_path):
    rec = _one_record(MICRO, sparse_pi=((0, -1),))
    with pytest.raises(ShardInvariantError, match="negative count"):
        write_shard(tmp_path / "bad.npz", MICRO, [rec], run_id="r", actor_id="a", seq=0)


def test_zero_total_visit_count_raises(tmp_path):
    rec = _one_record(MICRO, sparse_pi=((0, 0), (1, 0)))
    with pytest.raises(ShardInvariantError, match="sum\\(N\\)"):
        write_shard(tmp_path / "bad.npz", MICRO, [rec], run_id="r", actor_id="a", seq=0)


def test_empty_sparse_pi_raises(tmp_path):
    rec = _one_record(MICRO, sparse_pi=())
    with pytest.raises(ShardInvariantError, match="sum\\(N\\)"):
        write_shard(tmp_path / "bad.npz", MICRO, [rec], run_id="r", actor_id="a", seq=0)


def test_plies_decreasing_within_a_game_raises(tmp_path):
    r0 = _one_record(MICRO, ply=1)
    r1 = _one_record(MICRO, ply=0)  # same game_id, ply goes backwards
    with pytest.raises(ShardInvariantError, match="non-decreasing"):
        write_shard(tmp_path / "bad.npz", MICRO, [r0, r1], run_id="r", actor_id="a", seq=0)


def test_plies_may_reset_across_a_game_boundary(tmp_path):
    r0 = _one_record(MICRO, ply=3, game_id=("r", "a", 0))
    r1 = _one_record(MICRO, ply=0, game_id=("r", "a", 1))  # new game: ply resets, fine
    path = tmp_path / "ok.npz"
    write_shard(path, MICRO, [r0, r1], run_id="r", actor_id="a", seq=0)
    data = read_shard(path, MICRO)
    assert [r.ply for r in data.records] == [3, 0]


def test_non_monotone_pi_offsets_raises_via_validate_arrays():
    # write_shard always builds monotone offsets by construction; broken
    # offsets can only arise from a hand-corrupted array, so this reaches the
    # private validator directly (as the task note allows).
    from core.replay_shard import _pack_arrays, _validate_arrays

    rec = _one_record(MICRO)
    arrays = _pack_arrays(MICRO, [rec, rec])
    arrays["pi_offsets"] = np.array([0, 5, 2], dtype=np.int64)
    with pytest.raises(ShardInvariantError, match="monotone"):
        _validate_arrays(MICRO, arrays)


def test_game_id_not_matching_run_actor_raises(tmp_path):
    rec = _one_record(MICRO, game_id=("other-run", "a", 0))
    with pytest.raises(ValueError, match="game_id"):
        write_shard(tmp_path / "bad.npz", MICRO, [rec], run_id="r", actor_id="a", seq=0)


def test_empty_shard_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_shard(tmp_path / "bad.npz", MICRO, [], run_id="r", actor_id="a", seq=0)


# --- fingerprint gate on read --------------------------------------------------


def test_read_shard_rejects_wrong_adapter_config(tmp_path):
    rec = _one_record(MICRO)
    path = tmp_path / "s.npz"
    write_shard(path, MICRO, [rec], run_id="r", actor_id="a", seq=0)
    with pytest.raises(FingerprintMismatchError, match="orientation_hash"):
        read_shard(path, FULL)


def test_read_shard_rejects_unsupported_schema_version(tmp_path):
    rec = _one_record(MICRO)
    path = tmp_path / "s.npz"
    write_shard(path, MICRO, [rec], run_id="r", actor_id="a", seq=0)

    with np.load(path) as npz:
        arrays = dict(npz.items())
    fingerprint = build_fingerprint(MICRO)
    fingerprint["schema_version"] = 999
    arrays["fingerprint_json"] = np.asarray(canonical_json(fingerprint))
    corrupted = tmp_path / "corrupted.npz"
    with open(corrupted, "wb") as fh:
        np.savez(fh, **arrays)

    with pytest.raises(FingerprintMismatchError, match="schema_version"):
        read_shard(corrupted, MICRO)


# --- shard filename + durable identity -----------------------------------------


def test_shard_filename_format_is_exact():
    assert shard_filename("run7", "actor3", 42) == "shard-run7-actor3-42.npz"


def test_shard_writer_persists_state_across_restarts(tmp_path):
    writer = ShardWriter(tmp_path, MICRO, run_id="run", actor_id="actor")
    g1 = [_pending_samples(MICRO, 2, seed=10, model_version=1)]
    writer.write_shard(g1)
    assert writer.state.next_shard_seq == 1
    assert writer.state.next_game_index == 1

    g2 = [_pending_samples(MICRO, 2, seed=11, model_version=1)]
    writer.write_shard(g2)
    assert writer.state.next_shard_seq == 2
    assert writer.state.next_game_index == 2

    reloaded = ShardWriter(tmp_path, MICRO, run_id="run", actor_id="actor")
    assert reloaded.state == writer.state

    g3 = [_pending_samples(MICRO, 2, seed=12, model_version=1)]
    path3 = reloaded.write_shard(g3)
    assert path3.name == shard_filename("run", "actor", 2)


def test_shard_writer_assigns_durable_game_ids_in_batch_order(tmp_path):
    writer = ShardWriter(tmp_path, MICRO, run_id="run", actor_id="actor")
    games = [
        _pending_samples(MICRO, 2, seed=20, model_version=1),
        _pending_samples(MICRO, 2, seed=21, model_version=1),
        _pending_samples(MICRO, 2, seed=22, model_version=1),
    ]
    path = writer.write_shard(games)
    data = read_shard(path, MICRO)
    game_indices = sorted({r.game_id[2] for r in data.records})
    assert game_indices == [0, 1, 2]
    for r in data.records:
        assert r.game_id[0] == "run"
        assert r.game_id[1] == "actor"


def test_crash_window_burns_seq_and_game_index_without_reissue(tmp_path):
    writer = ShardWriter(tmp_path, MICRO, run_id="run", actor_id="actor")
    published = writer.write_shard([_pending_samples(MICRO, 2, seed=30, model_version=1)])
    assert published.exists()

    # Reserve state for a second shard, then simulate a crash: never publish.
    burned_seq, burned_game_index = writer._reserve(1)
    burned_path = tmp_path / shard_filename("run", "actor", burned_seq)
    assert not burned_path.exists()

    # "Restart": a fresh writer over the same directory reloads persisted state.
    restarted = ShardWriter(tmp_path, MICRO, run_id="run", actor_id="actor")
    assert restarted.state.next_shard_seq == burned_seq + 1
    assert restarted.state.next_game_index == burned_game_index + 1

    # The burned sequence number is never reused...
    next_path = restarted.write_shard([_pending_samples(MICRO, 2, seed=31, model_version=1)])
    assert next_path.name == shard_filename("run", "actor", burned_seq + 1)
    assert not burned_path.exists()

    # ...and the burned game index is never reissued to a real sample.
    data = read_shard(next_path, MICRO)
    game_indices = {r.game_id[2] for r in data.records}
    assert burned_game_index not in game_indices
    assert game_indices == {burned_game_index + 1}


def test_writer_state_file_is_human_readable_json(tmp_path):
    writer = ShardWriter(tmp_path, MICRO, run_id="run", actor_id="actor")
    writer.write_shard([_pending_samples(MICRO, 2, seed=40, model_version=1)])
    state_path = writer_state_path(tmp_path, "run", "actor")
    import json

    payload = json.loads(state_path.read_text())
    assert payload == {"next_shard_seq": 1, "next_game_index": 1}


# --- atomic publish -------------------------------------------------------------


def test_atomic_write_leaves_no_temp_files_after_success(tmp_path):
    path = tmp_path / "s.npz"
    rec = _one_record(MICRO)
    write_shard(path, MICRO, [rec], run_id="r", actor_id="a", seq=0)
    assert path.exists()
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_atomic_write_never_leaves_the_final_name_on_failure(tmp_path):
    path = tmp_path / "s.npz"

    def boom(fh):
        fh.write(b"partial-bytes")
        raise RuntimeError("simulated crash mid-write")

    with pytest.raises(RuntimeError, match="simulated crash"):
        _atomic_write(path, boom)

    assert not path.exists()
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_write_shard_failure_never_publishes_the_final_name(tmp_path, monkeypatch):
    import core.replay_shard as replay_shard

    def boom(*args, **kwargs):
        raise RuntimeError("simulated numpy failure")

    monkeypatch.setattr(replay_shard.np, "savez", boom)
    path = tmp_path / "s.npz"
    rec = _one_record(MICRO)
    with pytest.raises(RuntimeError, match="simulated numpy failure"):
        write_shard(path, MICRO, [rec], run_id="r", actor_id="a", seq=0)
    assert not path.exists()
    assert list(tmp_path.glob("*.tmp-*")) == []
