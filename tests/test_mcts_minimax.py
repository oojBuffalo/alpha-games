"""MCTS-vs-minimax correctness oracle (design doc §12 M0, §13).

The search analogue of "oracle first": high-sim MCTS must recover the game-theoretic
value on the fully-solved reference games. The non-flaky property checked is *MCTS plays
optimally* — its chosen move preserves the mover's optimal value (it never throws away
the game-theoretic outcome) — verified against the independent max-n solver.

Solver-grounded sweeps run on small games (TTT, 3x3 Connect-3). On the standard Connect
Four board the solver is infeasible, so those cases assert the concrete tactic directly.
"""

from __future__ import annotations

import pytest

from core import MCTS
from games.connect4 import Connect4
from games.tictactoe import TicTacToe
from tests.reference.minimax import (
    optimal_values,
    reachable_states,
    subtree_size,
)


def _empties(state) -> int:
    return state[0].count(-1)


def _mcts_move(game, state, sims):
    m = MCTS(game)
    m.run(sims, root_state=state)
    return m.best_action()


def _assert_plays_optimally(game, state, sims, value_cache):
    mover = game.current_player(state)
    target = optimal_values(game, state, value_cache)[mover]
    action = _mcts_move(game, state, sims)
    achieved = optimal_values(game, game.apply(state, action), value_cache)[mover]
    assert achieved >= target - 1e-9, (
        f"MCTS blundered on {state}: chose {action} (value {achieved}) < optimal {target}"
    )


# --- solver-grounded optimality sweeps on small games ------------------------


def test_mcts_plays_optimally_on_ttt_endgames():
    """Every TTT position with <= 3 plies remaining: MCTS must not blunder."""
    game = TicTacToe()
    value_cache: dict = {}
    size_cache: dict = {}
    tested = 0
    for state in reachable_states(game):
        if game.is_terminal(state) or _empties(state) > 3:
            continue
        sz = subtree_size(game, state, size_cache)
        sims = min(2000, max(240, 30 * sz))
        _assert_plays_optimally(game, state, sims, value_cache)
        tested += 1
    assert tested > 100  # sanity: many distinct endgames actually exercised


def test_mcts_plays_optimally_on_small_connect4_endgames():
    """Every 3x3 Connect-3 position with <= 3 plies remaining: MCTS must not blunder."""
    game = Connect4(3, 3, 3)
    value_cache: dict = {}
    size_cache: dict = {}
    tested = 0
    for state in reachable_states(game):
        if game.is_terminal(state) or _empties(state) > 3:
            continue
        sz = subtree_size(game, state, size_cache)
        sims = min(2000, max(240, 30 * sz))
        _assert_plays_optimally(game, state, sims, value_cache)
        tested += 1
    assert tested > 30


@pytest.mark.slow
def test_mcts_plays_optimally_on_ttt_deep():
    """Deeper sweep: every TTT position with <= 5 plies remaining (slow)."""
    game = TicTacToe()
    value_cache: dict = {}
    size_cache: dict = {}
    for state in reachable_states(game):
        if game.is_terminal(state) or _empties(state) > 5:
            continue
        sz = subtree_size(game, state, size_cache)
        sims = min(3000, max(300, 16 * sz))
        _assert_plays_optimally(game, state, sims, value_cache)


# --- direct tactical assertions on the standard board (solver infeasible) -----


def test_mcts_takes_immediate_vertical_win_connect4():
    game = Connect4()
    state = game.from_moves([0, 1, 0, 1, 0, 1])  # player 0, three stacked in col 0
    assert _mcts_move(game, state, 1500) == 0  # complete the vertical four


def test_mcts_blocks_immediate_vertical_loss_connect4():
    game = Connect4()
    # Player 1 has three stacked in col 1; player 0 (no win of its own) must block col 1.
    state = game.from_moves([0, 1, 2, 1, 0, 1])
    assert _mcts_move(game, state, 2000) == 1
