"""Perft fixtures: perft(1)/(2) by opening, Klein-4 orbit constancy, perft(3) golden.

The perft(2) reply counts are oracle-generated (exhaustive move-gen per
opening); perft(3) is bitboard-generated with Klein-4 orbit reduction —
provenance is recorded in the fixture, and the oracle differentially
spot-checks sampled perft(2)-frontier states here [F2]. The [F5] shortcut
(P2's reply must cover the other start square) is asserted equivalent to the
exhaustive oracle both in the generator and on a sample below.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from games.blokus_duo.actions import OPENING_ACTIONS, START_SQUARES, action_cells
from games.blokus_duo.bitboard import BitboardEngine
from games.blokus_duo.oracle import OracleEngine
from games.blokus_duo.pieces import orientation_table_hash

FIXTURES = Path(__file__).parent / "fixtures" / "blokus"

ORACLE = OracleEngine()
BITBOARD = BitboardEngine()
ALL_OPENINGS = sorted(set(OPENING_ACTIONS[(4, 4)]) | set(OPENING_ACTIONS[(9, 9)]))


@pytest.fixture(scope="module")
def perft():
    data = json.loads((FIXTURES / "perft.json").read_text())
    assert data["orientation_hash"] == orientation_table_hash()
    assert data["conventions"]["flatten"] == "(r*14+c)*91+o"
    return data


@pytest.fixture(scope="module")
def symmetry_maps():
    data = json.loads((FIXTURES / "symmetry_table.json").read_text())
    actions = data["actions"]
    return {g: dict(zip(actions, m, strict=True)) for g, m in data["maps"].items()}


def _reply_counts(perft):
    return {int(a): n for a, n in perft["reply_counts"].items()}


def _n2(perft):
    return {int(a): n for a, n in perft["perft3_by_opening"].items()}


def _place_opening(engine, a):
    return engine.place(engine.initial_state(), a)


def _other_square(a):
    covered = set(action_cells(a)) & set(START_SQUARES)
    (sq,) = covered
    return START_SQUARES[1 - START_SQUARES.index(sq)]


def test_perft1_is_828(perft):
    assert perft["perft1"] == 828
    assert sorted(_reply_counts(perft)) == ALL_OPENINGS
    assert len(BITBOARD.legal_actions(BITBOARD.initial_state(), 0)) == 828


def test_perft2_total_and_orbit_constancy(perft, symmetry_maps):
    counts = _reply_counts(perft)
    assert sum(counts.values()) == perft["perft2_total"]
    for g, table in symmetry_maps.items():
        for a in ALL_OPENINGS:
            assert counts[a] == counts[table[a]], (g, a)


def test_perft3_total_and_orbit_constancy(perft, symmetry_maps):
    n2 = _n2(perft)
    assert sorted(n2) == ALL_OPENINGS
    assert sum(n2.values()) == perft["perft3_total"]
    for g, table in symmetry_maps.items():
        for a in ALL_OPENINGS:
            assert n2[a] == n2[table[a]], (g, a)


def test_oracle_reply_counts_sample_vs_fixture(perft):
    counts = _reply_counts(perft)
    for a in random.Random(5).sample(ALL_OPENINGS, 4):
        s = _place_opening(ORACLE, a)
        assert len(ORACLE.legal_actions(s, 1)) == counts[a]


def test_bitboard_reply_counts_sample_vs_fixture(perft):
    counts = _reply_counts(perft)
    for a in random.Random(6).sample(ALL_OPENINGS, 24):
        s = _place_opening(BITBOARD, a)
        assert len(BITBOARD.legal_actions(s, 1)) == counts[a]


@pytest.mark.slow
def test_bitboard_reply_counts_all_828(perft):
    counts = _reply_counts(perft)
    for a in ALL_OPENINGS:
        s = _place_opening(BITBOARD, a)
        assert len(BITBOARD.legal_actions(s, 1)) == counts[a]


def test_shortcut_equals_exhaustive_oracle_on_sample():
    # [F5] The 414-candidate shortcut (reply must cover the other start
    # square, minus overlaps) must equal the oracle's exhaustive move-gen.
    for a in random.Random(7).sample(ALL_OPENINGS, 3):
        cells_a = set(action_cells(a))
        candidates = sorted(
            b for b in OPENING_ACTIONS[_other_square(a)] if not cells_a & set(action_cells(b))
        )
        s = _place_opening(ORACLE, a)
        assert ORACLE.legal_actions(s, 1) == candidates


def test_oracle_spot_checks_perft3_frontier(perft):
    # [F2] Differential oracle recount of P1's moves at sampled perft(2)-
    # frontier states (the bitboard generated perft(3); the oracle audits it).
    rng = random.Random(8)
    for a in rng.sample(ALL_OPENINGS, 2):
        so = ORACLE.place(ORACLE.initial_state(), a)
        sb = BITBOARD.place(BITBOARD.initial_state(), a)
        replies = ORACLE.legal_actions(so, 1)
        for b in rng.sample(replies, 2):
            so2 = ORACLE.place((*so[:6], 1, False), b)
            sb2 = BITBOARD.place((*sb[:6], 1, False), b)
            assert ORACLE.legal_actions(so2, 0) == BITBOARD.legal_actions(sb2, 0)


@pytest.mark.slow
def test_perft3_recompute_orbit_representative(perft):
    # Recompute N2 for one orbit representative from scratch via the bitboard
    # and compare against the fixture (guards against fixture drift without
    # regenerating the whole table).
    n2 = _n2(perft)
    a = ALL_OPENINGS[0]
    sb = _place_opening(BITBOARD, a)
    total = 0
    for b in BITBOARD.legal_actions(sb, 1):
        sb2 = BITBOARD.place((*sb[:6], 1, False), b)
        total += len(BITBOARD.legal_actions(sb2, 0))
    assert total == n2[a]
