"""Blokus-specific ladder pieces (M1.6): rung 2 and the start-square balancer.

Rung 2 (largest-piece/coverage, §12 M1.6 pin): argmax placed-cell count through
the adapter surface only (``len(decode_action(a))``), uniform-random among ties.
The balancer restricts a pair's second-game opener to openings covering the same
start square the first game's opener covered — keyed on the *configured* start
squares (§12 M2.5 evaluates the same hook on the micro instance), so both
instances are exercised here.
"""

from __future__ import annotations

import pytest

from core import RandomAgent
from core.runner import play_game, play_pairs
from games.blokus_duo import BlokusDuo
from games.blokus_duo.baselines import LargestPieceAgent, start_square_balancer
from games.blokus_duo.config import MICRO_CONFIG
from games.tictactoe import TicTacToe

GAME = BlokusDuo()  # bitboard-backed production engine
MICRO = BlokusDuo(config=MICRO_CONFIG)  # the §5.3 instance: a config, not a fork


def _covered_start(game, action):
    cells = set(game.decode_action(action))
    (sq,) = [s for s in game.config.start_squares if s in cells]
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


def test_start_square_balancer_matches_start_squares_on_the_micro_instance():
    """The §12 M2.5 generalization: the hook keys on the *configured* squares."""
    assert MICRO.config.start_squares == ((1, 1), (3, 3)) != GAME.config.start_squares
    results = play_pairs(
        MICRO,
        lambda s: RandomAgent(s),
        lambda s: RandomAgent(s),
        n_pairs=5,
        seed=11,
        opening_balancer=start_square_balancer,
    )
    seen = set()
    for pair in results:
        fwd, rev = pair.games
        square = _covered_start(MICRO, fwd.opening)
        assert square in MICRO.config.start_squares
        assert _covered_start(MICRO, rev.opening) == square
        seen.add(square)
    # Balancing must not collapse the pair set onto one start square.
    assert seen == set(MICRO.config.start_squares)


def test_start_square_balancer_predicate_accepts_exactly_the_matching_openings():
    """The returned predicate is the same-square restriction, on either instance."""
    for game in (GAME, MICRO):
        openings = list(game.legal_moves(game.initial_state()))
        first = openings[0]
        square = _covered_start(game, first)
        predicate = start_square_balancer(game, first)
        for opening in openings:
            assert predicate(opening) == (square in set(game.decode_action(opening)))
        assert predicate(first)


def test_start_square_balancer_rejects_a_non_blokus_game():
    """The balancer is game-specific; a foreign game must not be silently balanced."""
    with pytest.raises(TypeError, match="no Blokus instance config"):
        start_square_balancer(TicTacToe(), 0)


def test_start_square_balancer_rejects_a_non_opening_action():
    """An action covering no start square would unbalance the pair silently."""
    state = GAME.initial_state()
    for _ in range(2):  # both openings are played out: ply 2 covers neither square
        state = GAME.apply(state, list(GAME.legal_moves(state))[0])
    later = list(GAME.legal_moves(state))[0]
    assert not set(GAME.decode_action(later)) & set(GAME.config.start_squares)
    with pytest.raises(ValueError, match="covers 0 start square"):
        start_square_balancer(GAME, later)
