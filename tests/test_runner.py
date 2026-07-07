"""Evaluation game runner (M1.6): single games, then mirrored pairs.

The runner drives seeded agents through the public ``Game`` interface and
reports exact terminal utilities; the §9 protocol details (seat swap, per-pair
seeds, draws 0.5, opening balancing) are layered on top of single games.
"""

from __future__ import annotations

from core import RandomAgent
from core.agents import Agent
from core.runner import play_game, play_pairs
from games.tictactoe import TicTacToe

GAME = TicTacToe()


class CenterMinAgent(Agent):
    """Deterministic seat-agnostic rule: take the center if free, else the
    lowest legal cell. Self-play with this rule draws TTT from either seat."""

    @property
    def name(self) -> str:
        return "center-min"

    def select_action(self, game, state):
        moves = list(game.legal_moves(state))
        return 4 if 4 in moves else min(moves)


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


# --- mirrored pairs (§9 protocol) ---------------------------------------------


def test_play_pairs_scores_are_complementary_and_bounded():
    results = play_pairs(
        GAME,
        lambda seed: RandomAgent(seed),
        lambda seed: RandomAgent(seed),
        n_pairs=8,
        seed=123,
    )
    assert len(results) == 8
    for i, pair in enumerate(results):
        assert pair.pair_index == i
        # score_a + score_b == 2 per pair (each game contributes 1 total).
        assert pair.score_a + pair.score_b == 2.0
        assert 0.0 <= pair.score_a <= 2.0
        assert len(pair.games) == 2


def test_play_pairs_is_deterministic_given_master_seed():
    args = (GAME, lambda s: RandomAgent(s), lambda s: RandomAgent(s))
    assert play_pairs(*args, n_pairs=5, seed=9) == play_pairs(*args, n_pairs=5, seed=9)
    assert play_pairs(*args, n_pairs=5, seed=9) != play_pairs(*args, n_pairs=5, seed=10)


def test_play_pairs_swaps_seats_and_reuses_the_pair_seed():
    calls_a, calls_b = [], []

    def factory_a(seed):
        calls_a.append(seed)
        return RandomAgent(seed)

    def factory_b(seed):
        calls_b.append(seed)
        return RandomAgent(seed)

    play_pairs(GAME, factory_a, factory_b, n_pairs=3, seed=0)
    # Each factory is invoked once per game (two per pair), with the same
    # per-pair seed in both games of a pair, and different seeds across pairs.
    assert len(calls_a) == len(calls_b) == 6
    for i in range(3):
        assert calls_a[2 * i] == calls_a[2 * i + 1]
        assert calls_b[2 * i] == calls_b[2 * i + 1]
    assert len(set(calls_a)) == 3


def test_draws_score_half_per_game():
    # Center-then-min self-play draws TTT from either seat: both games of the
    # pair end 0/0 and each must contribute exactly 0.5.
    results = play_pairs(
        GAME, lambda s: CenterMinAgent(), lambda s: CenterMinAgent(), n_pairs=1, seed=1
    )
    (pair,) = results
    assert all(rec.utilities == (0.0, 0.0) for rec in pair.games)
    assert pair.score_a == pair.score_b == 1.0  # 0.5 + 0.5 each
