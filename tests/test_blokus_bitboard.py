"""Bitboard engine: goldens, hand-position differentials, and fuzz vs. the oracle.

The bitboard engine is the production move generator; the oracle is the
exhaustive reference. They share only piece data and the action encoding, so
agreement on legal-id *sets* at every ply of random games, plus apply results,
terminal flags, and exact scores, is the load-bearing correctness argument
(design doc §12 M1). [F2]: ``|score_diff| <= max_score_diff(config)`` is
asserted at every fuzz terminal — the range every training target flows through
(109 full, 29 micro).

M2.5 task 3 puts the fuzz on a config axis: the identical differential runs for
:data:`FULL_CONFIG` and the §5.3 micro instance, so a micro-only legality or
scoring bug cannot slip through the reduced pipeline. Full-game coverage is
unchanged (same game counts, same seeds); micro rides along with a much larger
game budget because the board is cheap. The exhaustive micro counterpart — both
engines in lockstep over *every* reachable micro position — lives in
``tests/test_blokus_perft.py``.
"""

from __future__ import annotations

import random

import pytest

from games.blokus_duo.actions import OPENING_ACTIONS, action_codec, encode
from games.blokus_duo.bitboard import BitboardEngine, cells_to_bb
from games.blokus_duo.config import FULL_CONFIG, MICRO_CONFIG
from games.blokus_duo.game import BlokusDuo
from games.blokus_duo.oracle import OracleEngine
from games.blokus_duo.pieces import build_pieces
from games.blokus_duo.targets import max_score_diff
from tests.test_blokus_oracle import make_state, micro_state

ORACLE = OracleEngine()
BITBOARD = BitboardEngine()
MICRO_CODEC = action_codec(MICRO_CONFIG)
MONO, DOMINO = 0, 1
MICRO_V3 = 3

# Fuzz budgets per instance: the full game keeps its M1 game counts exactly;
# micro is orders of magnitude cheaper per game, so it gets a wider net.
FUZZ_FAST = [pytest.param(FULL_CONFIG, 3, id="full"), pytest.param(MICRO_CONFIG, 40, id="micro")]
FUZZ_SLOW = [pytest.param(FULL_CONFIG, 25, id="full"), pytest.param(MICRO_CONFIG, 2000, id="micro")]


def bb_state(oracle_state, board_size: int = FULL_CONFIG.board_size):
    """Convert an oracle (frozenset-based) state to the bitboard layout.

    Args:
        oracle_state: State tuple with occupancies as frozensets of cells.
        board_size: Row stride of the bit layout (14 full, 5 micro).

    Returns:
        The same state with occupancies packed as ints.
    """
    occ0, occ1, *rest = oracle_state
    return (cells_to_bb(occ0, board_size), cells_to_bb(occ1, board_size), *rest)


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


@pytest.mark.parametrize(
    "oracle_state",
    [
        micro_state(occ0=[(0, 0)], inv0=[MONO]),
        micro_state(occ0=[(0, 0)], inv0=[]),
        micro_state(occ0=[(0, 0)], occ1=[(1, 2)], inv0=[DOMINO]),
        micro_state(occ0=[(0, 0)], inv0=[DOMINO]),
        micro_state(occ0=[(0, 0)], occ1=[(1, 1)], inv0=[MONO]),
        micro_state(occ0=[(1, 1)], inv0=range(1, 4), to_play=1),  # P2 opening
        micro_state(occ0=[(0, 0)], inv0=[MICRO_V3]),  # the only 4-orientation piece
        micro_state(occ0=[(1, 1), (2, 2)], occ1=[(3, 3)], inv0=[2, 3], inv1=range(1, 4)),
    ],
)
def test_micro_hand_positions_match_oracle(oracle_state):
    # The same clause-by-clause differential on the §5.3 instance, where the
    # board edge clips halos that never clip at 14×14.
    oracle, bitboard = OracleEngine(MICRO_CONFIG), BitboardEngine(MICRO_CONFIG)
    s = bb_state(oracle_state, MICRO_CONFIG.board_size)
    for player in (0, 1):
        assert bitboard.legal_actions(s, player) == oracle.legal_actions(oracle_state, player)


def test_place_matches_oracle_semantics():
    so = make_state(occ0=[(0, 0)], inv0=[MONO])
    sb = bb_state(so)
    a = encode(1, 1, 0)
    no, nb = ORACLE.place(so, a), BITBOARD.place(sb, a)
    assert nb[0] == cells_to_bb(no[0])
    assert no[2:] == nb[2:]  # inventories, flags, to_play, terminal identical
    assert ORACLE.scores(no) == BITBOARD.scores(nb) == (20, -89)


def test_micro_place_matches_oracle_semantics():
    oracle, bitboard = OracleEngine(MICRO_CONFIG), BitboardEngine(MICRO_CONFIG)
    so = micro_state(occ0=[(0, 0)], inv0=[MONO])
    sb = bb_state(so, MICRO_CONFIG.board_size)
    a = MICRO_CODEC.encode(1, 1, 0)
    no, nb = oracle.place(so, a), bitboard.place(sb, a)
    assert nb[0] == cells_to_bb(no[0], MICRO_CONFIG.board_size)
    assert no[2:] == nb[2:]
    assert oracle.scores(no) == bitboard.scores(nb) == (20, -9)


def _fuzz(config, n_games, seed):
    """Play seeded random games through both engines in lockstep.

    Asserts agreement on move generation (identical sorted legal-id lists at
    every ply), ``apply`` (identical inventories, flags, ``to_play``), terminal
    detection, and final scores — plus the [F2] score-range invariant this
    config's aux divisor rests on.

    Args:
        config: The Blokus instance to fuzz.
        n_games: Number of random playouts.
        seed: Seed for the move-choice RNG stream.
    """
    oracle, bitboard = OracleEngine(config), BitboardEngine(config)
    game_o, game_b = BlokusDuo(oracle), BlokusDuo(bitboard)
    squares = sum(len(p) for p in build_pieces(config)[0])
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
        scores_o, scores_b = oracle.scores(so), bitboard.scores(sb)
        assert scores_o == scores_b
        assert all(-squares <= s <= 20 for s in scores_o)  # 89 / 9 squares per set
        assert abs(scores_o[0] - scores_o[1]) <= max_score_diff(config)  # [F2]


@pytest.mark.parametrize("config,n_games", FUZZ_FAST)
def test_differential_fuzz_random_playouts(config, n_games):
    _fuzz(config, n_games, seed=11)


@pytest.mark.slow
@pytest.mark.parametrize("config,n_games", FUZZ_SLOW)
def test_differential_fuzz_random_playouts_slow(config, n_games):
    _fuzz(config, n_games, seed=13)
