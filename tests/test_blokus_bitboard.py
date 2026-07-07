"""Bitboard engine: goldens, hand-position differentials, and fuzz vs. the oracle.

The bitboard engine is the production move generator; the oracle is the
exhaustive reference. They share only piece data and the action encoding, so
agreement on legal-id *sets* at every ply of random games, plus apply results,
terminal flags, and exact scores, is the load-bearing correctness argument
(design doc §12 M1). [F2]: |score_diff| <= 109 is asserted at every fuzz
terminal — the range every training target flows through.
"""

from __future__ import annotations

import random

import pytest

from games.blokus_duo.actions import OPENING_ACTIONS, encode
from games.blokus_duo.bitboard import BitboardEngine, cells_to_bb
from games.blokus_duo.game import BlokusDuo
from games.blokus_duo.oracle import OracleEngine
from tests.test_blokus_oracle import make_state

ORACLE = OracleEngine()
BITBOARD = BitboardEngine()
MONO, DOMINO = 0, 1


def bb_state(oracle_state):
    """Convert an oracle (frozenset-based) state to the bitboard layout."""
    occ0, occ1, *rest = oracle_state
    return (cells_to_bb(occ0), cells_to_bb(occ1), *rest)


def test_initial_legal_actions_are_the_828_openings():
    legal = BITBOARD.legal_actions(BITBOARD.initial_state(), 0)
    assert len(legal) == 828
    assert set(legal) == set(OPENING_ACTIONS[(4, 4)]) | set(OPENING_ACTIONS[(9, 9)])


@pytest.mark.parametrize(
    "oracle_state",
    [
        make_state(occ0=[(0, 0)], inv0=[MONO]),
        make_state(occ0=[(0, 0)], inv0=[]),
        make_state(occ0=[(0, 0)], occ1=[(1, 2)], inv0=[DOMINO]),
        make_state(occ0=[(0, 0)], inv0=[DOMINO]),
        make_state(occ0=[(0, 0)], occ1=[(1, 1)], inv0=[MONO]),
        make_state(occ0=[(4, 4)], inv0=range(1, 21), to_play=1),  # P2 opening
        make_state(occ0=[(5, 5), (6, 6)], occ1=[(8, 8)], inv0=range(3, 21), inv1=range(1, 21)),
    ],
)
def test_hand_positions_match_oracle(oracle_state):
    # Differential on the hand-built legality positions: every clause the
    # oracle tests pin must agree bit-for-bit here, for both players.
    s = bb_state(oracle_state)
    for player in (0, 1):
        assert BITBOARD.legal_actions(s, player) == ORACLE.legal_actions(oracle_state, player)


def test_place_matches_oracle_semantics():
    so = make_state(occ0=[(0, 0)], inv0=[MONO])
    sb = bb_state(so)
    a = encode(1, 1, 0)
    no, nb = ORACLE.place(so, a), BITBOARD.place(sb, a)
    assert nb[0] == cells_to_bb(no[0])
    assert no[2:] == nb[2:]  # inventories, flags, to_play, terminal identical
    assert ORACLE.scores(no) == BITBOARD.scores(nb) == (20, -89)


def _fuzz(n_games, seed):
    game_o = BlokusDuo(ORACLE)
    game_b = BlokusDuo(BITBOARD)
    rng = random.Random(seed)
    for _ in range(n_games):
        so, sb = game_o.initial_state(), game_b.initial_state()
        while True:
            assert game_o.is_terminal(so) == game_b.is_terminal(sb)
            assert so[2:] == sb[2:]  # inv/flags/to_play/terminal identical every ply
            if game_o.is_terminal(so):
                break
            legal_o = list(game_o.legal_moves(so))
            legal_b = list(game_b.legal_moves(sb))
            assert legal_o == legal_b  # sorted legal-id sets agree at every ply
            a = rng.choice(legal_o)
            so, sb = game_o.apply(so, a), game_b.apply(sb, a)
        scores_o, scores_b = ORACLE.scores(so), BITBOARD.scores(sb)
        assert scores_o == scores_b
        assert abs(scores_o[0] - scores_o[1]) <= 109  # [F2] on every fuzz terminal


def test_differential_fuzz_random_playouts():
    _fuzz(3, seed=11)


@pytest.mark.slow
def test_differential_fuzz_random_playouts_slow():
    _fuzz(25, seed=13)
