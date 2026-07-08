"""Blokus-specific ladder pieces (M1.6): rung 2 and the start-square balancer.

Rung 2 (largest-piece/coverage, §12 M1.6 pin): argmax placed-cell count through
the adapter surface only (``len(decode_action(a))``), uniform-random among ties.
The balancer restricts a pair's second-game opener to openings covering the same
start square the first game's opener covered.
"""

from __future__ import annotations

from core import RandomAgent
from core.runner import play_game, play_pairs
from games.blokus_duo import BlokusDuo
from games.blokus_duo.baselines import LargestPieceAgent, start_square_balancer

GAME = BlokusDuo()  # bitboard-backed production engine

_STARTS = ((4, 4), (9, 9))


def _covered_start(game, action):
    cells = set(game.decode_action(action))
    (sq,) = [s for s in _STARTS if s in cells]
    return sq


def test_largest_piece_agent_opens_with_a_pentomino():
    agent = LargestPieceAgent(seed=0)
    s0 = GAME.initial_state()
    for _ in range(5):
        a = agent.select_action(GAME, s0)
        assert a in list(GAME.legal_moves(s0))
        assert len(GAME.decode_action(a)) == 5  # largest order available


def test_largest_piece_agent_always_plays_a_maximal_piece():
    agent = LargestPieceAgent(seed=3)
    s = GAME.initial_state()
    for _ in range(6):  # first plies: maximal size must be picked at each step
        moves = list(GAME.legal_moves(s))
        best = max(len(GAME.decode_action(a)) for a in moves)
        a = agent.select_action(GAME, s)
        assert len(GAME.decode_action(a)) == best
        s = GAME.apply(s, a)


def test_largest_piece_vs_random_full_game_is_legal():
    rec = play_game(GAME, (LargestPieceAgent(seed=1), RandomAgent(seed=2)))
    assert sum(rec.utilities) == 0.0
    assert rec.plies >= 4


def test_start_square_balancer_matches_start_squares_within_pairs():
    results = play_pairs(
        GAME,
        lambda s: RandomAgent(s),
        lambda s: RandomAgent(s),
        n_pairs=3,
        seed=11,
        opening_balancer=start_square_balancer,
    )
    for pair in results:
        fwd, rev = pair.games
        assert _covered_start(GAME, fwd.opening) == _covered_start(GAME, rev.opening)
