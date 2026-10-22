"""Tests for the M4 eval record store (design doc §9; tasks/m4/005, subtasks 5.2/5.3).

Covers: the protocol registry's self-enforcing fingerprint, cell-id canonicalization
bijectivity, header/record round-trips, the ``PairResult -> record`` and
``records -> Match`` conversions, the crash-resume torn-tail golden, manifest
idempotence and its illegal-transition guards, and the snapshot reader's race-free
semantics.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import threading
import time
from pathlib import Path

import pytest

import core.eval_protocol as eval_protocol
from core import RandomAgent
from core.eval_store import (
    CellHeader,
    CellId,
    ConfigMismatchError,
    CorruptedCellError,
    GameRecordSnapshot,
    ManifestError,
    PairRecord,
    ProtocolMismatchError,
    SchemaVersionError,
    append_pair_record,
    build_cell_id,
    build_header,
    cell_path,
    complete_cell,
    is_cell_complete,
    iter_cells,
    load_snapshot,
    manifest_path,
    open_cell_for_resume,
    open_cell_for_write,
    pair_result_to_record,
    parse_cell_id,
    read_cell,
    records_to_match,
    register_member,
    write_header,
)
from core.runner import GameRecord, PairResult, play_pairs
from core.seeding import derive_seed
from games.tictactoe import TicTacToe

PAIRS_PER_CELL = eval_protocol.PAIRS_PER_CELL


def _make_header(
    *,
    run_id: str = "run-1",
    candidate_version: int = 12,
    rung: int = 7,
    opponent_id: str = "random",
    eval_config: dict | None = None,
    candidate_fingerprint: dict | None = None,
    cell_seed: int = 0,
) -> CellHeader:
    return build_header(
        run_id=run_id,
        cell_seed=cell_seed,
        cell_id=CellId(candidate_version, rung, opponent_id),
        candidate_identity=f"rung{rung}-v1-{candidate_version}",
        opponent_identity=opponent_id,
        eval_config=(
            eval_config if eval_config is not None else eval_protocol.eval_config_snapshot()
        ),
        candidate_fingerprint=candidate_fingerprint or {"orientation_table_hash": "abc123"},
    )


def _flat_record(pair_index: int, score_a: float = 1.0) -> PairRecord:
    return PairRecord(
        pair_index=pair_index,
        pair_seed=derive_seed(0, "pair", pair_index),
        score_a=score_a,
        games=(GameRecordSnapshot(plies=1, opening=0), GameRecordSnapshot(plies=1, opening=0)),
    )


def _fill_cell(run_dir, header: CellHeader, n_pairs: int):
    """Open (or resume) ``header``'s cell and append ``n_pairs`` flat records."""
    next_index = open_cell_for_write(run_dir, header)
    path = cell_path(run_dir, header.cell_id.to_string())
    for i in range(next_index, next_index + n_pairs):
        append_pair_record(path, _flat_record(i))
    return path


def _write_and_complete_cell(run_dir, header: CellHeader, n_pairs: int = PAIRS_PER_CELL):
    cid = header.cell_id.to_string()
    register_member(run_dir, header.cell_id.candidate_version, [cid])
    _fill_cell(run_dir, header, n_pairs)
    complete_cell(run_dir, cid)
    return cid


# ---------------------------------------------------------------------------------
# Protocol registry.
# ---------------------------------------------------------------------------------


def test_protocol_fingerprint_changes_when_the_registry_changes(monkeypatch):
    before = eval_protocol.protocol_fingerprint()
    monkeypatch.setitem(eval_protocol.REGISTRY, "a_new_convention_constant", 1)
    after = eval_protocol.protocol_fingerprint()
    assert before != after


def test_protocol_fingerprint_is_stable_for_unchanged_registry_content():
    assert eval_protocol.protocol_fingerprint() == eval_protocol.protocol_fingerprint()
    # Reordering-insensitive: rebuild an equal-content dict in a different order.
    reordered = dict(reversed(list(eval_protocol.REGISTRY.items())))
    canonical = json.dumps(reordered, sort_keys=True, separators=(",", ":"))
    import hashlib

    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert expected == eval_protocol.protocol_fingerprint()


def test_eval_sims_registry_constant_matches_the_eval_agents_code_mirror():
    from core.eval_agents import EVAL_SIMS as agents_eval_sims

    assert eval_protocol.EVAL_SIMS == agents_eval_sims


def test_rung8_lag_divisor_and_earliest_version_match_the_eval_agents_code_mirror():
    """RUNG8_LAG_DIVISOR/RUNG8_EARLIEST_VERSION mirror core.eval_agents.historical_opponents's
    hardcoded ``math.ceil(k_total / 4)`` / ``{..., 1}`` rule (the registry documents the
    shape of that rule rather than being imported by it -- same EVAL_SIMS-style mirror
    above). Exercised across several (candidate, k_total) pairs, including one where the
    lag term and the earliest-version term coincide, so a future change to the hardcoded
    divisor or earliest version drifts loudly here instead of silently.
    """
    from core.eval_agents import historical_opponents

    cases = [
        (2, 4),  # k_total=4: lag = ceil(4/4) = 1 -> candidate-1 == candidate-lag
        (5, 12),  # k_total=12: lag = ceil(12/4) = 3
        (7, 8),  # k_total=8: lag = ceil(8/4) = 2
        (30, 30),  # candidate == k_total
        (2, 2),  # lag term lands exactly on the earliest version
    ]
    for candidate, k_total in cases:
        versions = list(range(1, k_total + 1))
        lag = math.ceil(k_total / eval_protocol.RUNG8_LAG_DIVISOR)
        wanted = {candidate - 1, candidate - lag, eval_protocol.RUNG8_EARLIEST_VERSION}
        expected = sorted(u for u in wanted if 1 <= u < candidate and u in set(versions))
        assert historical_opponents(versions, candidate, k_total=k_total) == expected


def test_seed_label_constants_match_play_pairs_actual_derivation():
    """SEED_LABEL_PAIR/SEED_LABEL_SEAT_A/SEED_LABEL_SEAT_B mirror core.runner.play_pairs's
    hardcoded ``"pair"``/``"a"``/``"b"`` labels (intentional layering -- ``core.runner``
    must not import the M4-only ``core.eval_protocol``, see that module's docstring). This
    structurally ties all three registry labels to the actual per-seat seeds play_pairs
    derives (not just the per-pair seed the incidental round-trip test above already
    covers), by capturing the seeds handed to each agent factory and comparing them
    against seeds derived straight from the registry's own labels.
    """
    game = TicTacToe()
    seed = 909
    for pair_index in range(4):
        seen_a: set[int] = set()
        seen_b: set[int] = set()

        def factory_a(s, seen=seen_a):
            seen.add(s)
            return RandomAgent(s)

        def factory_b(s, seen=seen_b):
            seen.add(s)
            return RandomAgent(s)

        [result] = play_pairs(
            game, factory_a, factory_b, n_pairs=1, seed=seed, start_pair_index=pair_index
        )

        expected_pair_seed = derive_seed(seed, eval_protocol.SEED_LABEL_PAIR, pair_index)
        expected_seed_a = derive_seed(expected_pair_seed, eval_protocol.SEED_LABEL_SEAT_A)
        expected_seed_b = derive_seed(expected_pair_seed, eval_protocol.SEED_LABEL_SEAT_B)

        assert result.pair_seed == expected_pair_seed
        assert seen_a == {expected_seed_a}
        assert seen_b == {expected_seed_b}


def test_build_header_stamps_the_current_protocol_registry():
    header = _make_header()
    assert header.schema_version == eval_protocol.SCHEMA_VERSION
    assert header.protocol_version == eval_protocol.PROTOCOL_VERSION
    assert header.protocol_fingerprint == eval_protocol.protocol_fingerprint()


# ---------------------------------------------------------------------------------
# Cell-id canonicalization: bijective, filesystem-safe.
# ---------------------------------------------------------------------------------


def test_cell_id_canonicalization_is_bijective_over_random_triples():
    rng = random.Random(20260829)
    charset = list(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        "-_./ %:!@#$^&*()[]{}|\\'\"~+=<>,;\t"
    )
    for _ in range(500):
        candidate_version = rng.randint(-1000, 1000)
        rung = rng.randint(-10, 10)
        length = rng.randint(0, 24)
        opponent_id = "".join(rng.choice(charset) for _ in range(length))
        if rng.random() < 0.3:
            # Stress non-ASCII too, avoiding the surrogate range.
            opponent_id += "".join(chr(rng.randint(0x80, 0x2FFF)) for _ in range(rng.randint(0, 3)))
        triple = CellId(candidate_version, rung, opponent_id)
        cid = build_cell_id(candidate_version, rung, opponent_id)
        assert "/" not in cid
        assert "\n" not in cid
        assert parse_cell_id(cid) == triple
        assert triple.to_string() == cid


def test_parse_cell_id_rejects_malformed_input():
    with pytest.raises(ValueError):
        parse_cell_id("only.two")
    with pytest.raises(ValueError):
        parse_cell_id("notanint.7.opponent")
    with pytest.raises(ValueError):
        parse_cell_id("7.notanint.opponent")


def test_candidate_identity_matches_eval_agents_naming_convention():
    cell = CellId(candidate_version=9, rung=6, opponent_id="mobility")
    assert cell.candidate_identity == "rung6-v1-9"


# ---------------------------------------------------------------------------------
# Header write -> read round trip.
# ---------------------------------------------------------------------------------


def test_header_round_trip_preserves_every_field(tmp_path):
    header = _make_header(eval_config={"pairs_per_cell": 24, "eval_sims": 512})
    path = tmp_path / "cell.jsonl"
    write_header(path, header)
    read_header, records = read_cell(path)
    assert read_header == header
    assert records == []


def test_write_header_refuses_to_overwrite_an_existing_cell_file(tmp_path):
    header = _make_header()
    path = tmp_path / "cell.jsonl"
    write_header(path, header)
    with pytest.raises(FileExistsError):
        write_header(path, header)


def test_write_read_round_trip_preserves_pair_records(tmp_path):
    header = _make_header()
    path = _fill_cell(tmp_path, header, 4)
    read_header, records = read_cell(path)
    assert read_header == header
    assert [r.pair_index for r in records] == [0, 1, 2, 3]
    assert records[2] == _flat_record(2)


# ---------------------------------------------------------------------------------
# PairResult -> record, and records -> Match.
# ---------------------------------------------------------------------------------


def test_pair_result_to_record_agrees_with_a_hand_built_pair_including_draws():
    fwd = GameRecord(utilities=(1.0, -1.0), plies=5, opening=0)
    rev = GameRecord(utilities=(0.0, 0.0), plies=9, opening=1)  # a drawn second game
    result = PairResult(pair_index=2, score_a=1.5, score_b=0.5, games=(fwd, rev), pair_seed=777)

    record = pair_result_to_record(result)

    assert record.pair_index == 2
    assert record.pair_seed == 777
    assert record.score_a == 1.5
    assert "score_b" not in record.to_dict()
    assert result.score_b == 2.0 - record.score_a  # implicit relationship holds
    assert record.games == (
        GameRecordSnapshot(plies=5, opening=0),
        GameRecordSnapshot(plies=9, opening=1),
    )


def test_records_to_match_aggregates_like_core_elo_matches_from_pairs():
    records = [_flat_record(i, score_a=1.0) for i in range(3)]
    match = records_to_match("candidate", "opponent", records)
    assert match == ("candidate", "opponent", 3.0, 6)


def test_pair_seed_recomputes_from_cell_seed_and_pair_index():
    game = TicTacToe()
    seed = 555
    results = play_pairs(
        game, lambda s: RandomAgent(s), lambda s: RandomAgent(s), n_pairs=5, seed=seed
    )
    for result in results:
        expected = derive_seed(seed, eval_protocol.SEED_LABEL_PAIR, result.pair_index)
        assert result.pair_seed == expected
        assert pair_result_to_record(result).pair_seed == expected


# ---------------------------------------------------------------------------------
# Resumption golden: torn-tail crash recovery is byte-identical to an uninterrupted run.
# ---------------------------------------------------------------------------------


def _play_and_append(path, game, seed, start, n_pairs):
    for i in range(start, start + n_pairs):
        [result] = play_pairs(
            game,
            lambda s: RandomAgent(s),
            lambda s: RandomAgent(s),
            n_pairs=1,
            seed=seed,
            start_pair_index=i,
        )
        append_pair_record(path, pair_result_to_record(result))


def test_resumption_golden_byte_identical_after_a_torn_tail(tmp_path):
    game = TicTacToe()
    seed = 4242
    n_pairs = 6
    kill_at = 3
    header = _make_header(candidate_version=1, cell_seed=seed)

    ref_dir = tmp_path / "uninterrupted"
    ref_next = open_cell_for_write(ref_dir, header)
    assert ref_next == 0
    ref_path = cell_path(ref_dir, header.cell_id.to_string())
    _play_and_append(ref_path, game, seed, 0, n_pairs)
    reference_bytes = ref_path.read_bytes()

    live_dir = tmp_path / "interrupted"
    open_cell_for_write(live_dir, header)
    live_path = cell_path(live_dir, header.cell_id.to_string())
    _play_and_append(live_path, game, seed, 0, kill_at)
    size_before_crash = live_path.stat().st_size

    # Simulate a crash mid-append: a torn fragment with no trailing newline.
    with open(live_path, "ab") as fh:
        fh.write(b'{"pair_index": 3, "pair_se')

    resumed_index = open_cell_for_resume(live_path, header)
    assert resumed_index == kill_at
    assert live_path.stat().st_size == size_before_crash  # torn fragment discarded

    _play_and_append(live_path, game, seed, kill_at, n_pairs - kill_at)
    assert live_path.read_bytes() == reference_bytes


def test_open_cell_for_write_resumes_transparently(tmp_path):
    game = TicTacToe()
    seed = 99
    header = _make_header(candidate_version=1, cell_seed=seed)
    next_index = open_cell_for_write(tmp_path, header)
    assert next_index == 0
    path = cell_path(tmp_path, header.cell_id.to_string())
    _play_and_append(path, game, seed, 0, 2)

    resumed = open_cell_for_write(tmp_path, header)
    assert resumed == 2


def test_open_cell_for_resume_raises_on_a_header_only_file_with_no_records(tmp_path):
    header = _make_header()
    open_cell_for_write(tmp_path, header)
    path = cell_path(tmp_path, header.cell_id.to_string())
    assert open_cell_for_resume(path, header) == 0


def test_open_cell_for_resume_missing_file_raises(tmp_path):
    header = _make_header()
    with pytest.raises(FileNotFoundError):
        open_cell_for_resume(cell_path(tmp_path, header.cell_id.to_string()), header)


# ---------------------------------------------------------------------------------
# Real corruption is not a crash-torn tail: it must fail loudly, never truncate.
# ---------------------------------------------------------------------------------


def test_open_cell_for_resume_raises_on_mid_stream_corruption_without_truncating(tmp_path):
    """A newline-terminated line that fails to parse is real corruption (bit rot, a
    bug, a stray second writer) -- not the crash-torn, no-trailing-newline tail
    ``_truncate_torn_tail`` exists to discard. It must raise loudly and must never
    truncate the file: silently truncating here would discard the corrupted line
    *and* every already-durable, valid line recorded after it.
    """
    game = TicTacToe()
    seed = 777
    header = _make_header(candidate_version=1, cell_seed=seed)
    open_cell_for_write(tmp_path, header)
    path = cell_path(tmp_path, header.cell_id.to_string())
    _play_and_append(path, game, seed, 0, 3)  # durable pair_index 0, 1, 2

    raw_lines = path.read_bytes().splitlines(keepends=True)
    assert len(raw_lines) == 4  # header + 3 pair records
    assert json.loads(raw_lines[2])["pair_index"] == 1

    # Corrupt pair_index 1's JSON body in place -- preserving its trailing newline,
    # simulating bit-rot rather than a crash mid-write.
    raw_lines[2] = b"not valid json at all\n"
    corrupted_bytes = b"".join(raw_lines)
    path.write_bytes(corrupted_bytes)

    with pytest.raises(CorruptedCellError):
        open_cell_for_resume(path, header)

    # Nothing was truncated: the corrupted line and the valid pair_index 2 record
    # stored after it are both still on disk, byte-for-byte.
    assert path.read_bytes() == corrupted_bytes


# ---------------------------------------------------------------------------------
# Loud failures: unknown schema, config drift, protocol/registry drift.
# ---------------------------------------------------------------------------------


def test_read_cell_rejects_unknown_schema_version(tmp_path):
    header = _make_header()
    path = tmp_path / "cell.jsonl"
    write_header(path, header)
    payload = json.loads(path.read_text().splitlines()[0])
    payload["schema_version"] = 999
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(SchemaVersionError):
        read_cell(path)


def test_open_cell_for_resume_rejects_unknown_schema_version(tmp_path):
    header = _make_header()
    open_cell_for_write(tmp_path, header)
    path = cell_path(tmp_path, header.cell_id.to_string())
    payload = json.loads(path.read_text().splitlines()[0])
    payload["schema_version"] = 999
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(SchemaVersionError):
        open_cell_for_resume(path, header)


def test_open_cell_for_resume_rejects_a_config_mismatch(tmp_path):
    header = _make_header()
    open_cell_for_write(tmp_path, header)
    path = cell_path(tmp_path, header.cell_id.to_string())

    drifted_config = dict(header.eval_config)
    drifted_config["pairs_per_cell"] = drifted_config["pairs_per_cell"] + 1
    drifted_header = dataclasses.replace(header, eval_config=drifted_config)

    with pytest.raises(ConfigMismatchError):
        open_cell_for_resume(path, drifted_header)


def test_open_cell_for_resume_rejects_a_protocol_registry_mismatch(tmp_path):
    header = _make_header()
    open_cell_for_write(tmp_path, header)
    path = cell_path(tmp_path, header.cell_id.to_string())

    drifted_header = dataclasses.replace(header, protocol_fingerprint="0" * 64)

    with pytest.raises(ProtocolMismatchError):
        open_cell_for_resume(path, drifted_header)


def test_open_cell_for_resume_rejects_a_changed_run_id(tmp_path):
    header = _make_header(run_id="run-1")
    open_cell_for_write(tmp_path, header)
    path = cell_path(tmp_path, header.cell_id.to_string())
    other_run_header = dataclasses.replace(header, run_id="run-2")
    with pytest.raises(ConfigMismatchError):
        open_cell_for_resume(path, other_run_header)


# ---------------------------------------------------------------------------------
# Manifest: idempotence and illegal transitions.
# ---------------------------------------------------------------------------------


def test_register_member_is_idempotent_on_the_identical_set(tmp_path):
    cids = [build_cell_id(1, 7, "random"), build_cell_id(1, 7, "mobility")]
    register_member(tmp_path, 1, cids)
    register_member(tmp_path, 1, list(reversed(cids)))  # same set, different order
    payload = json.loads(manifest_path(tmp_path).read_text())
    assert sorted(payload["members"]["1"]["required_cells"]) == sorted(cids)


def test_register_member_rejects_a_changed_required_set(tmp_path):
    register_member(tmp_path, 1, [build_cell_id(1, 7, "random"), build_cell_id(1, 7, "mobility")])
    with pytest.raises(ManifestError):
        register_member(tmp_path, 1, [build_cell_id(1, 7, "random")])


def test_register_member_rejects_a_foreign_candidate_cell_id(tmp_path):
    with pytest.raises(ValueError):
        register_member(tmp_path, 1, [build_cell_id(2, 7, "random")])


def test_register_member_rejects_a_non_member_version(tmp_path):
    with pytest.raises(ValueError):
        register_member(tmp_path, 0, [build_cell_id(0, 7, "random")])


def test_complete_cell_requires_prior_scheduling(tmp_path):
    with pytest.raises(ManifestError):
        complete_cell(tmp_path, build_cell_id(1, 7, "random"))


def test_complete_cell_is_idempotent_and_never_reopens(tmp_path):
    cid = build_cell_id(1, 7, "random")
    register_member(tmp_path, 1, [cid])
    assert not is_cell_complete(tmp_path, cid)
    _fill_cell(tmp_path, _make_header(candidate_version=1), PAIRS_PER_CELL)
    complete_cell(tmp_path, cid)
    assert is_cell_complete(tmp_path, cid)
    before = json.loads(manifest_path(tmp_path).read_text())

    complete_cell(tmp_path, cid)  # idempotent no-op

    after = json.loads(manifest_path(tmp_path).read_text())
    assert before == after  # completed_at untouched, cell never reopened


# ---------------------------------------------------------------------------------
# Snapshot reader.
# ---------------------------------------------------------------------------------


def test_snapshot_excludes_a_partial_cell(tmp_path):
    header = _make_header(candidate_version=1)
    cid = header.cell_id.to_string()
    register_member(tmp_path, 1, [cid])
    _fill_cell(tmp_path, header, PAIRS_PER_CELL)  # fully written, but never completed

    snap = load_snapshot(tmp_path)

    assert snap.completed_cell_ids == frozenset()
    assert snap.member_prefix == 0


def test_snapshot_prefix_truncates_at_a_hole_in_the_member_series(tmp_path):
    cids = {}
    for v in (1, 2, 3):
        header = _make_header(candidate_version=v)
        cids[v] = header.cell_id.to_string()
        register_member(tmp_path, v, [cids[v]])
        _fill_cell(tmp_path, header, PAIRS_PER_CELL)
        if v != 2:
            complete_cell(tmp_path, cids[v])

    snap = load_snapshot(tmp_path)

    assert snap.member_prefix == 1
    assert cids[1] in snap.completed_cell_ids
    assert cids[2] not in snap.completed_cell_ids
    # Member 3 is complete despite the hole at 2 -- still valid evidence for
    # per-checkpoint reporting, just outside the contiguous authoritative prefix.
    assert cids[3] in snap.completed_cell_ids


def test_snapshot_fingerprint_stable_while_a_writer_appends_to_an_incomplete_cell(tmp_path):
    header1 = _make_header(candidate_version=1)
    _write_and_complete_cell(tmp_path, header1)
    snap_before = load_snapshot(tmp_path)

    header2 = _make_header(candidate_version=2, opponent_id="mobility")
    register_member(tmp_path, 2, [header2.cell_id.to_string()])
    _fill_cell(tmp_path, header2, 1)  # incomplete: still "scheduled", not complete

    snap_after = load_snapshot(tmp_path)

    assert snap_after.snapshot_fingerprint == snap_before.snapshot_fingerprint
    assert snap_after.completed_cell_ids == snap_before.completed_cell_ids
    assert snap_after.member_prefix == snap_before.member_prefix


def test_snapshot_fingerprint_changes_when_a_completed_cells_content_changes(tmp_path):
    def build_run(run_dir, score_a):
        header = _make_header(candidate_version=1)
        cid = header.cell_id.to_string()
        register_member(run_dir, 1, [cid])
        path = None
        open_cell_for_write(run_dir, header)
        path = cell_path(run_dir, cid)
        for i in range(PAIRS_PER_CELL):
            append_pair_record(path, _flat_record(i, score_a=score_a))
        complete_cell(run_dir, cid)

    run_a, run_b = tmp_path / "a", tmp_path / "b"
    build_run(run_a, 1.0)
    build_run(run_b, 1.5)

    assert load_snapshot(run_a).snapshot_fingerprint != load_snapshot(run_b).snapshot_fingerprint


def test_snapshot_fingerprint_is_invariant_to_manifest_timestamps(tmp_path):
    header = _make_header(candidate_version=1)
    _write_and_complete_cell(tmp_path, header)
    snap1 = load_snapshot(tmp_path)

    payload = json.loads(manifest_path(tmp_path).read_text())
    for entry in payload["cells"].values():
        entry["scheduled_at"] += 1000.0
        if entry["completed_at"] is not None:
            entry["completed_at"] += 1000.0
    manifest_path(tmp_path).write_text(json.dumps(payload))

    snap2 = load_snapshot(tmp_path)
    assert snap2.snapshot_fingerprint == snap1.snapshot_fingerprint
    assert snap2.member_prefix == snap1.member_prefix
    assert snap2.completed_cell_ids == snap1.completed_cell_ids


def test_snapshot_rejects_manifest_marking_a_missing_cell_complete(tmp_path):
    cid = build_cell_id(1, 7, "random")
    register_member(tmp_path, 1, [cid])
    with pytest.raises(ManifestError):
        complete_cell(tmp_path, cid)
    assert not is_cell_complete(tmp_path, cid)


def test_snapshot_rejects_a_cell_short_of_its_pinned_pair_count(tmp_path):
    header = _make_header(candidate_version=1)
    cid = header.cell_id.to_string()
    register_member(tmp_path, 1, [cid])
    _fill_cell(tmp_path, header, 2)  # short of PAIRS_PER_CELL
    with pytest.raises(ManifestError):
        complete_cell(tmp_path, cid)
    assert not is_cell_complete(tmp_path, cid)


def test_snapshot_rejects_a_cell_whose_header_disagrees_with_its_own_filename(tmp_path):
    header = _make_header(candidate_version=1, opponent_id="random")
    wrong_cid = build_cell_id(1, 7, "mobility")
    register_member(tmp_path, 1, [wrong_cid])
    path = cell_path(tmp_path, wrong_cid)
    write_header(path, header)  # header says "random"; filed under "mobility"
    for i in range(PAIRS_PER_CELL):
        append_pair_record(path, _flat_record(i))
    with pytest.raises(ManifestError):
        complete_cell(tmp_path, wrong_cid)


def test_snapshot_rejects_unknown_manifest_schema_version(tmp_path):
    register_member(tmp_path, 1, [build_cell_id(1, 7, "random")])
    payload = json.loads(manifest_path(tmp_path).read_text())
    payload["schema_version"] = 999
    manifest_path(tmp_path).write_text(json.dumps(payload))
    with pytest.raises(SchemaVersionError):
        load_snapshot(tmp_path)


def test_iter_cells_yields_every_completed_cell_in_sorted_order(tmp_path):
    header_a = _make_header(candidate_version=1, opponent_id="mobility")
    header_b = _make_header(candidate_version=1, opponent_id="random")
    register_member(tmp_path, 1, [header_a.cell_id.to_string(), header_b.cell_id.to_string()])
    for header in (header_a, header_b):
        _fill_cell(tmp_path, header, PAIRS_PER_CELL)
        complete_cell(tmp_path, header.cell_id.to_string())

    snap = load_snapshot(tmp_path)
    paths = list(iter_cells(snap))

    assert paths == sorted(paths)
    assert {p.name for p in paths} == {
        f"{header_a.cell_id.to_string()}.jsonl",
        f"{header_b.cell_id.to_string()}.jsonl",
    }
    for path in paths:
        read_header, records = read_cell(path)
        assert len(records) == PAIRS_PER_CELL
        assert read_header.schema_version == eval_protocol.SCHEMA_VERSION


def test_form_versions_and_search_budgets_have_distinct_cell_paths():
    cells = [
        CellId(12, 7, "random"),
        CellId(12, 7, "random", full_candidate_identity="rung7-v2-12"),
        CellId(12, 7, "random", full_candidate_identity="rung7-v1-s64-12"),
    ]
    assert len({cell.to_string() for cell in cells}) == 3
    for cell in cells:
        assert parse_cell_id(cell.to_string()) == cell


@pytest.mark.parametrize("field", ["candidate_identity", "opponent_identity"])
def test_header_duplicate_identity_must_agree(field):
    with pytest.raises(ConfigMismatchError):
        dataclasses.replace(
            _make_header(), **{field: "mobility" if field == "opponent_identity" else "rung7-v2-12"}
        )


@pytest.mark.parametrize("fragment", [b"", b'{"candidate_fingerprint":{"h"'])
def test_legacy_torn_header_is_recovered(tmp_path, fragment):
    header = _make_header()
    path = cell_path(tmp_path, header.cell_id.to_string())
    path.parent.mkdir(parents=True)
    path.write_bytes(fragment)
    assert open_cell_for_write(tmp_path, header) == 0
    assert read_cell(path) == (header, [])


def test_failed_header_publication_never_exposes_partial_file(tmp_path, monkeypatch):
    import core.eval_store as store

    path = tmp_path / "cell.jsonl"

    def crash(*args):
        raise OSError("crash before publication")

    with monkeypatch.context() as m:
        m.setattr(store.os, "link", crash)
        with pytest.raises(OSError):
            write_header(path, _make_header())
    assert not path.exists()
    write_header(path, _make_header())
    assert read_cell(path)[0] == _make_header()


def test_concurrent_header_publication_has_one_winner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "cell.jsonl"

    def publish(_):
        try:
            write_header(path, _make_header())
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(publish, range(8))) == 1
    assert read_cell(path) == (_make_header(), [])


def test_completed_cell_rejects_all_write_surfaces_and_retained_path(tmp_path):
    header = _make_header(candidate_version=1)
    cid = _write_and_complete_cell(tmp_path, header)
    path = cell_path(tmp_path, cid)
    original = path.read_bytes()
    for operation in [
        lambda: open_cell_for_write(tmp_path, header),
        lambda: open_cell_for_resume(path, header),
        lambda: append_pair_record(path, _flat_record(PAIRS_PER_CELL)),
    ]:
        with pytest.raises(ManifestError):
            operation()
        assert path.read_bytes() == original


@pytest.mark.parametrize("indices", [[1], [0, 0], [0, 2], [1, 0]])
def test_resume_and_read_reject_noncontiguous_pairs(tmp_path, indices):
    header = _make_header()
    path = cell_path(tmp_path, header.cell_id.to_string())
    open_cell_for_write(tmp_path, header)
    with path.open("a") as fh:
        for index in indices:
            fh.write(json.dumps(_flat_record(index).to_dict()) + "\n")
    for operation in [lambda: open_cell_for_resume(path, header), lambda: read_cell(path)]:
        with pytest.raises(CorruptedCellError):
            operation()


def test_changed_seed_and_wrong_pair_seed_are_rejected(tmp_path):
    header = _make_header()
    path = _fill_cell(tmp_path, header, 1)
    with pytest.raises(ConfigMismatchError):
        open_cell_for_resume(path, dataclasses.replace(header, cell_seed=42))
    with pytest.raises(ConfigMismatchError):
        append_pair_record(path, dataclasses.replace(_flat_record(1), pair_seed=999))
    assert len(read_cell(path)[1]) == 1


def test_full_scheduled_cell_resumes_at_pair_count_then_completes(tmp_path):
    # Crash between the final append and complete_cell: the relaunch must see
    # "nothing left to play" and be able to complete, never wedge the cell.
    header = _make_header()
    cid = header.cell_id.to_string()
    register_member(tmp_path, header.cell_id.candidate_version, [cid])
    path = _fill_cell(tmp_path, header, PAIRS_PER_CELL)
    assert open_cell_for_write(tmp_path, header) == PAIRS_PER_CELL
    assert open_cell_for_resume(path, header) == PAIRS_PER_CELL
    with pytest.raises(ManifestError):
        append_pair_record(path, _flat_record(PAIRS_PER_CELL))
    complete_cell(tmp_path, cid)
    assert is_cell_complete(tmp_path, cid)
    with pytest.raises(ManifestError):
        open_cell_for_write(tmp_path, header)


def test_snapshot_refuses_evidence_cells_with_mixed_run_ids(tmp_path):
    _write_and_complete_cell(tmp_path, _make_header(candidate_version=1, run_id="run-1"))
    _write_and_complete_cell(tmp_path, _make_header(candidate_version=2, run_id="other-run"))
    with pytest.raises(ConfigMismatchError, match="run_id"):
        load_snapshot(tmp_path)


def test_snapshot_refuses_evidence_cells_stamped_by_another_protocol(tmp_path, monkeypatch):
    _write_and_complete_cell(tmp_path, _make_header(candidate_version=1))
    monkeypatch.setitem(eval_protocol.REGISTRY, "a_new_convention_constant", 1)
    with pytest.raises(ProtocolMismatchError):
        load_snapshot(tmp_path)


def test_snapshot_stamp_check_ignores_post_hole_provisional_cells(tmp_path):
    # Only the evidence prefix must agree; a complete cell after a hole is
    # provisional and its stamps do not gate the analyzed prefix.
    _write_and_complete_cell(tmp_path, _make_header(candidate_version=1))
    register_member(tmp_path, 2, [build_cell_id(2, 7, "random")])
    _write_and_complete_cell(tmp_path, _make_header(candidate_version=3, run_id="other-run"))
    snapshot = load_snapshot(tmp_path)
    assert snapshot.member_prefix == 1


def test_future_header_schema_checked_before_renamed_fields(tmp_path):
    header = _make_header()
    path = _fill_cell(tmp_path, header, 0)
    path.write_text(json.dumps({"schema_version": 999, "future_run": "foo"}) + "\n")
    for operation in [lambda: read_cell(path), lambda: open_cell_for_resume(path, header)]:
        with pytest.raises(SchemaVersionError):
            operation()


def test_concurrent_manifest_updates_preserve_every_registration_and_completion(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    first = _make_header(candidate_version=1)
    cid = first.cell_id.to_string()
    register_member(tmp_path, 1, [cid])
    _fill_cell(tmp_path, first, PAIRS_PER_CELL)

    def update(version):
        if version == 1:
            complete_cell(tmp_path, cid)
        else:
            register_member(tmp_path, version, [build_cell_id(version, 7, "random")])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(update, range(1, 25)))
    manifest = json.loads(manifest_path(tmp_path).read_text())
    assert len(manifest["members"]) == 24
    assert is_cell_complete(tmp_path, cid)


def test_post_hole_completion_leaves_prefix_fingerprint_and_iterator_unchanged(tmp_path):
    _write_and_complete_cell(tmp_path, _make_header(candidate_version=1))
    register_member(tmp_path, 2, [build_cell_id(2, 7, "random")])
    before = load_snapshot(tmp_path)
    _write_and_complete_cell(tmp_path, _make_header(candidate_version=3))
    after = load_snapshot(tmp_path)
    assert after.completed_cell_ids != before.completed_cell_ids
    assert after.snapshot_fingerprint == before.snapshot_fingerprint
    assert list(iter_cells(after)) == list(iter_cells(before))


def test_legacy_schema_rejected_before_missing_seed_parse(tmp_path):
    header = _make_header()
    path = _fill_cell(tmp_path, header, 0)
    payload = header.to_dict()
    payload["schema_version"] = 1
    del payload["cell_seed"]
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(SchemaVersionError):
        read_cell(path)
    with pytest.raises(SchemaVersionError):
        open_cell_for_resume(path, header)


def test_new_store_directories_are_persisted_in_their_parents_before_use(tmp_path, monkeypatch):
    """Each directory the store creates is fsynced into its parent before contents.

    ``mkdir`` alone does not survive a power loss until the parent is fsynced, so
    ``eval/`` must be persisted in ``run_dir`` and ``cells/`` in ``eval/`` before
    any durable write inside them is acknowledged.
    """
    import core.eval_store as store

    events: list[tuple[str, str]] = []
    real_fsync_dir = store._fsync_directory
    real_fsync = store.os.fsync

    def record_dir(path):
        events.append(("dir", Path(path).resolve().relative_to(tmp_path.resolve()).as_posix()))
        real_fsync_dir(path)

    def record_file(fd):
        events.append(("file", ""))
        real_fsync(fd)

    header = _make_header(eval_config={**eval_protocol.eval_config_snapshot(), "pairs_per_cell": 1})
    cid = header.cell_id.to_string()
    with monkeypatch.context() as m:
        m.setattr(store, "_fsync_directory", record_dir)
        m.setattr(store.os, "fsync", record_file)
        register_member(tmp_path, header.cell_id.candidate_version, [cid])
        assert open_cell_for_write(tmp_path, header) == 0
        append_pair_record(cell_path(tmp_path, cid), _flat_record(0))
        complete_cell(tmp_path, cid)
    assert is_cell_complete(tmp_path, cid)

    dir_syncs = [name for kind, name in events if kind == "dir"]
    # eval/ is created on first registration: run_dir is synced before anything else.
    assert events[0] == ("dir", ".")
    # cells/ is created by the first header: eval/ is synced after cells/ exists and
    # before the header file's own fsync is acknowledged by its directory sync.
    cells_sync = dir_syncs.index("eval/cells")
    assert "eval" in dir_syncs[1:cells_sync]
    # Store-owned levels are re-persisted on every use (existence is not durability),
    # but directories outside the store (run_dir's own parent) are never synced.
    assert ".." not in dir_syncs and all(not name.startswith("..") for name in dir_syncs)


def _record_dir_syncs(monkeypatch, root, events, *, block_first_root_sync=None):
    import core.eval_store as store

    real_fsync_dir = store._fsync_directory
    blocked = {"done": False}

    def record_dir(path):
        name = Path(path).resolve().relative_to(root.resolve()).as_posix()
        if block_first_root_sync is not None and name == "." and not blocked["done"]:
            blocked["done"] = True
            block_first_root_sync.wait(timeout=10)
        events.append((threading.get_ident(), name))
        real_fsync_dir(path)

    monkeypatch.setattr(store, "_fsync_directory", record_dir)


def test_preexisting_unsynced_store_dirs_are_persisted_before_acknowledgement(
    tmp_path, monkeypatch
):
    """A visible ``eval/``/``cells/`` left by a dead predecessor is still persisted.

    Simulates a writer that died between ``mkdir`` and its parent fsync: both
    store directories exist but were never synced. The next writer must fsync
    ``run_dir`` (for ``eval/``) and ``eval/`` (for ``cells/``) itself before its
    evidence is acknowledged.
    """
    (tmp_path / "eval" / "cells").mkdir(parents=True)
    events: list[tuple[int, str]] = []
    header = _make_header(eval_config={**eval_protocol.eval_config_snapshot(), "pairs_per_cell": 1})
    cid = header.cell_id.to_string()
    _record_dir_syncs(monkeypatch, tmp_path, events)
    register_member(tmp_path, header.cell_id.candidate_version, [cid])
    assert (threading.get_ident(), ".") in events  # eval/ persisted in run_dir
    events.clear()
    assert open_cell_for_write(tmp_path, header) == 0
    names = [name for _, name in events]
    assert "." in names and "eval" in names  # cells/ persisted in eval/ too
    append_pair_record(cell_path(tmp_path, cid), _flat_record(0))
    complete_cell(tmp_path, cid)
    assert is_cell_complete(tmp_path, cid)


def test_concurrent_initializer_cannot_ack_before_eval_dir_is_durable(tmp_path, monkeypatch):
    """Writer B must not rely on writer A's still-pending fsync of ``run_dir``.

    A creates ``eval/`` during ``register_member`` and stalls before fsyncing
    ``run_dir``. B then registers, writes, appends and completes a cell; B must
    have fsynced ``run_dir`` itself before ``complete_cell`` returns.
    """
    gate = threading.Event()
    events: list[tuple[int, str]] = []
    _record_dir_syncs(monkeypatch, tmp_path, events, block_first_root_sync=gate)
    header_a = _make_header()
    header_b = _make_header(
        candidate_version=header_a.cell_id.candidate_version + 1,
        eval_config={**eval_protocol.eval_config_snapshot(), "pairs_per_cell": 1},
    )
    errors: list[BaseException] = []

    def writer_a():
        try:
            register_member(
                tmp_path, header_a.cell_id.candidate_version, [header_a.cell_id.to_string()]
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread_a = threading.Thread(target=writer_a)
    thread_a.start()
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "eval").exists():
            assert time.monotonic() < deadline, "writer A never created eval/"
            time.sleep(0.001)
        me = threading.get_ident()
        cid = header_b.cell_id.to_string()
        register_member(tmp_path, header_b.cell_id.candidate_version, [cid])
        assert open_cell_for_write(tmp_path, header_b) == 0
        append_pair_record(cell_path(tmp_path, cid), _flat_record(0))
        complete_cell(tmp_path, cid)
        assert is_cell_complete(tmp_path, cid)
        assert not gate.is_set()  # A is still stalled before its run_dir sync
        assert (me, ".") in events, "B acknowledged evidence without persisting eval/"
    finally:
        gate.set()
        thread_a.join(timeout=10)
    assert not errors


def test_complete_cell_syncs_recovered_cell_before_committing_manifest(tmp_path, monkeypatch):
    """A final append that died before its fsync is made durable before completion.

    The writer flushes the last record and dies before ``os.fsync`` returns, so the
    complete record sits only in the page cache: recovery sees no torn tail and
    never syncs. ``complete_cell`` must fsync the cell file and its directory
    before the manifest marks the cell complete.
    """
    import os

    import core.eval_store as store

    header = _make_header(eval_config={**eval_protocol.eval_config_snapshot(), "pairs_per_cell": 1})
    cid = header.cell_id.to_string()
    path = cell_path(tmp_path, cid)
    register_member(tmp_path, header.cell_id.candidate_version, [cid])
    assert open_cell_for_write(tmp_path, header) == 0
    cell_ino = path.stat().st_ino
    real_fsync = store.os.fsync

    def dying_fsync(fd):
        if os.fstat(fd).st_ino == cell_ino:
            raise OSError("simulated crash before the final append's fsync")
        real_fsync(fd)

    with monkeypatch.context() as m:
        m.setattr(store.os, "fsync", dying_fsync)
        with pytest.raises(OSError, match="simulated crash"):
            append_pair_record(path, _flat_record(0))

    events: list[str] = []
    cells_ino = path.parent.stat().st_ino
    real_write_manifest = store._atomic_write_manifest

    def tracking_fsync(fd):
        ino = os.fstat(fd).st_ino
        if ino == cell_ino:
            events.append("cell")
        elif ino == cells_ino:
            events.append("cells_dir")
        real_fsync(fd)

    def tracking_manifest(run_dir, payload):
        if any(e["status"] == "complete" for e in payload["cells"].values()):
            events.append("manifest_complete")
        real_write_manifest(run_dir, payload)

    monkeypatch.setattr(store.os, "fsync", tracking_fsync)
    monkeypatch.setattr(store, "_atomic_write_manifest", tracking_manifest)
    assert open_cell_for_write(tmp_path, header) == 1  # full record survived in cache
    events.clear()
    complete_cell(tmp_path, cid)
    assert is_cell_complete(tmp_path, cid)
    commit = events.index("manifest_complete")
    assert "cell" in events[:commit]
    assert "cells_dir" in events[:commit]
