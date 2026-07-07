"""Evaluation game runner (M1.6): single games, then mirrored pairs.

The runner drives seeded agents through the public ``Game`` interface and
reports exact terminal utilities; the §9 protocol details (seat swap, per-pair
seeds, draws 0.5, opening balancing) are layered on top of single games.
"""

from __future__ import annotations

from core import RandomAgent
from core.runner import play_game
from games.tictactoe import TicTacToe

GAME = TicTacToe()


def test_play_game_reaches_a_zero_sum_terminal():
    rec = play_game(GAME, (RandomAgent(seed=1), RandomAgent(seed=2)))
    assert sum(rec.utilities) == 0.0
    assert all(u in (-1.0, 0.0, 1.0) for u in rec.utilities)
    assert 5 <= rec.plies <= 9


def test_play_game_is_deterministic_given_seeded_agents():
    rec_a = play_game(GAME, (RandomAgent(seed=3), RandomAgent(seed=4)))
    rec_b = play_game(GAME, (RandomAgent(seed=3), RandomAgent(seed=4)))
    assert rec_a == rec_b


def test_play_game_seats_map_to_player_ids():
    # Agent at index p moves exactly when current_player == p: an always-first
    # scripted agent on seat 0 must produce a game whose first move is its pick.
    class Scripted(RandomAgent):
        def __init__(self):
            super().__init__(seed=0)
            self.seen_players = []

        def select_action(self, game, state):
            self.seen_players.append(game.current_player(state))
            return super().select_action(game, state)

    a0, a1 = Scripted(), Scripted()
    play_game(GAME, (a0, a1))
    assert set(a0.seen_players) == {0}
    assert set(a1.seen_players) == {1}
