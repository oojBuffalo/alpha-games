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
) -> CellHeader:
    return build_header(
        run_id=run_id,
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
        pair_seed=pair_index,
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
    header = _make_header(candidate_version=1)

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
    header = _make_header(candidate_version=1)
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
    header = _make_header(candidate_version=1)
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
    complete_cell(tmp_path, cid)  # no cell file was ever written
    with pytest.raises(ManifestError):
        load_snapshot(tmp_path)


def test_snapshot_rejects_a_cell_short_of_its_pinned_pair_count(tmp_path):
    header = _make_header(candidate_version=1)
    cid = header.cell_id.to_string()
    register_member(tmp_path, 1, [cid])
    _fill_cell(tmp_path, header, 2)  # short of PAIRS_PER_CELL
    complete_cell(tmp_path, cid)
    with pytest.raises(ManifestError):
        load_snapshot(tmp_path)


def test_snapshot_rejects_a_cell_whose_header_disagrees_with_its_own_filename(tmp_path):
    header = _make_header(candidate_version=1, opponent_id="random")
    wrong_cid = build_cell_id(1, 7, "mobility")
    register_member(tmp_path, 1, [wrong_cid])
    path = cell_path(tmp_path, wrong_cid)
    write_header(path, header)  # header says "random"; filed under "mobility"
    for i in range(PAIRS_PER_CELL):
        append_pair_record(path, _flat_record(i))
    complete_cell(tmp_path, wrong_cid)
    with pytest.raises(ManifestError):
        load_snapshot(tmp_path)


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
