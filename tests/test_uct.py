"""Ladder rung 4: UCT (UCB1) with uniform-random rollouts (design doc §9, §12 M3).

Rung 4 is a standalone opponent, not a PUCT configuration — uniform priors do
not turn PUCT into UCB1's selection rule (``core/mcts.py``'s module docstring).
This battery covers the UCB1 selection formula and tie-break in isolation, the
player-aware backup on the synthetic pass-game fixtures (mirroring
``tests/test_pass_backup.py``), the rollout's exact terminal-utility read,
determinism, the frozen identity golden, a strength-sanity check against rung
1, and scheduling parity through the existing paired runner.
"""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from core import RandomAgent, UCTAgent
from core.agents import Agent
from core.elo import fit_elo, matches_from_pairs
from core.game import EnvelopeError, assert_v1_envelope
from core.runner import play_pairs
from core.uct import DEFAULT_C, UCTSearcher
from games.connect4 import Connect4
from games.tictactoe import TicTacToe
from tests.fixtures.bad_adapters import StochasticGame
from tests.fixtures.pass_game import PassGame, Scenario, consecutive_trap_game, consecutive_win_game

TTT = TicTacToe()


def _searcher(game=TTT, num_simulations=1, seed=0, c=DEFAULT_C):
    return UCTSearcher(game, num_simulations=num_simulations, rng=random.Random(seed), c=c)


# --- UCB1 selection: exact formula, untried-first, tie-break ------------------


def test_select_edge_visits_untried_children_first_in_lowest_id_order():
    # actions 5 and 9 are untried (N=0); the tried action (2) must not be picked
    # even though it sits between them in the actions list.
    node = SimpleNamespace(actions=[5, 2, 9], N=[0, 3, 0], Q=[0.0, 0.9, 0.0])
    s = _searcher()
    assert s._select_edge(node) == 0  # action id 5: lower than action id 9


def test_select_edge_untried_order_is_by_action_id_not_list_position():
    node = SimpleNamespace(actions=[9, 3, 7], N=[0, 0, 5], Q=[0.0, 0.0, 0.4])
    s = _searcher()
    assert s._select_edge(node) == 1  # action id 3 is untried and lowest


def test_select_edge_exact_ucb1_formula_once_all_visited():
    # Hand-computed: c = sqrt(2), N = [4, 9], Q = [0.5, 0.6].
    node = SimpleNamespace(actions=[0, 1], N=[4, 9], Q=[0.5, 0.6])
    c = math.sqrt(2.0)
    log_total = math.log(13)
    score0 = 0.5 + c * math.sqrt(log_total / 4)
    score1 = 0.6 + c * math.sqrt(log_total / 9)
    assert score0 > score1  # sanity: action 0 should win on these numbers
    s = _searcher(c=c)
    assert s._select_edge(node) == 0


def test_select_edge_breaks_exact_ties_by_lowest_action_id():
    # Identical N and Q for both actions -> identical UCB1 scores exactly.
    node = SimpleNamespace(actions=[9, 3], N=[5, 5], Q=[0.2, 0.2])
    s = _searcher()
    assert s._select_edge(node) == 1  # action id 3, even though it is list index 1


# --- player-aware backup (mirrors tests/test_pass_backup.py) ------------------


def test_uct_backup_signs_on_consecutive_win_game():
    g = consecutive_win_game()
    s = _searcher(g, num_simulations=3000, seed=1)
    assert s.search(g.initial_state()) == 0

    q = s.action_values()
    # Sign is measured in the mover's (player 0's) perspective: the winning branch
    # is positive, the losing branch negative. A sign bug would invert this.
    assert q[0] > 0.0 > q[1]

    # The consecutive (player-0-again) node must itself play the winning move.
    consecutive_state = g.apply(g.initial_state(), 0)
    s2 = _searcher(g, num_simulations=3000, seed=2)
    assert s2.search(consecutive_state) == 0
    q2 = s2.action_values()
    assert q2[0] > 0.0 > q2[1]


def test_uct_backup_avoids_self_loss_on_trap_game():
    g = consecutive_trap_game()
    consecutive_state = g.apply(g.initial_state(), 0)  # forced single move
    s = _searcher(g, num_simulations=3000, seed=3)

    # At the consecutive node, player 0 must avoid the trap (action 1 self-destructs).
    assert s.search(consecutive_state) == 0
    q = s.action_values()
    assert q[0] > 0.0 > q[1]


# --- rollout: exact terminal utility, zero-sum symmetry ------------------------


def _single_path_game() -> PassGame:
    """A branch-free two-ply scenario: every state has exactly one legal move.

    Deterministic regardless of the rollout RNG's draws — isolates the "read the
    terminal utility, never negate blindly through plies" behavior from any
    randomness in move choice.
    """
    return PassGame(
        Scenario(
            start=0,
            to_play={0: 0, 1: 1},
            edges={0: [(0, 1)], 1: [(0, 2)]},
            terminal={2: (1.0, -1.0)},
        )
    )


def test_rollout_returns_exact_terminal_utility_per_player():
    g = _single_path_game()
    s = _searcher(g, seed=123)
    assert s._rollout(0, 0) == 1.0
    assert s._rollout(0, 1) == -1.0


def test_rollout_utilities_are_zero_sum_symmetric():
    g = _single_path_game()
    s = _searcher(g, seed=456)
    assert s._rollout(0, 0) + s._rollout(0, 1) == 0.0


# --- determinism ----------------------------------------------------------------


def _play_full_game(game, agent):
    s = game.initial_state()
    moves = []
    while not game.is_terminal(s):
        a = agent.select_action(game, s)
        moves.append(a)
        s = game.apply(s, a)
    return moves


def test_uct_agent_is_deterministic_per_seed():
    moves_a = _play_full_game(TTT, UCTAgent(seed=99))
    moves_b = _play_full_game(TTT, UCTAgent(seed=99))
    assert moves_a == moves_b


def test_uct_agent_different_seeds_can_diverge():
    # Not a hard guarantee for every seed pair, but this pair is pinned to diverge
    # so the determinism test above cannot be vacuously true (e.g. a bug that
    # ignores the RNG entirely and always plays the same game).
    moves_a = _play_full_game(TTT, UCTAgent(seed=1))
    moves_b = _play_full_game(TTT, UCTAgent(seed=2))
    assert moves_a != moves_b


# --- frozen identity golden -----------------------------------------------------


def test_uct_rung4_identity_is_frozen():
    assert UCTAgent.RUNG_ID == "uct-rollout-v1"
    assert UCTAgent.NUM_SIMULATIONS == 1000
    assert UCTAgent.C == math.sqrt(2.0)
    assert UCTAgent(seed=0).name == "uct-rollout-v1"
    assert isinstance(UCTAgent(seed=0), Agent)


# --- envelope rejection (parity with MCTS's construction-time check) ------------


def test_uct_searcher_construction_rejects_out_of_envelope_adapters():
    with pytest.raises(EnvelopeError):
        UCTSearcher(StochasticGame(), num_simulations=1, rng=random.Random(0))


def test_uct_searcher_accepts_valid_games():
    assert_v1_envelope(TTT)  # sanity: must not raise
    UCTSearcher(TTT, num_simulations=1, rng=random.Random(0))  # must not raise


# --- strength sanity: rung 4 beats rung 1 decisively ----------------------------


class _FastUCTAgent(Agent):
    """Test-only UCT wrapper at reduced sims, for a cheap strength-sanity check.

    Not the frozen rung-4 identity: the registered ladder rung (``UCTAgent``)
    stays pinned at 1,000 simulations, exercised elsewhere in this module. This
    class exists only so the strength assertion below runs in a reasonable test
    budget, via the same parameterizable :class:`~core.uct.UCTSearcher`.
    """

    def __init__(self, seed: int, num_simulations: int = 150):
        self._rng = random.Random(seed)
        self._num_simulations = num_simulations

    @property
    def name(self) -> str:
        return "uct-fast-test-only"

    def select_action(self, game, state):
        searcher = UCTSearcher(game, num_simulations=self._num_simulations, rng=self._rng)
        return searcher.search(state)


def test_uct_beats_random_with_positive_anchored_elo_on_tictactoe():
    pairs = play_pairs(
        TTT,
        lambda seed: _FastUCTAgent(seed),
        lambda seed: RandomAgent(seed),
        n_pairs=15,
        seed=808,
    )
    score = sum(p.score_a for p in pairs)
    assert score > 27.0  # decisively above an even split of the 30 games
    ratings = fit_elo(matches_from_pairs("uct", "random", pairs), anchor="random")
    assert ratings["random"] == 0.0
    assert ratings["uct"] > 100.0


@pytest.mark.slow
def test_uct_beats_random_with_positive_anchored_elo_on_connect4():
    pairs = play_pairs(
        Connect4(),
        lambda seed: _FastUCTAgent(seed, num_simulations=200),
        lambda seed: RandomAgent(seed),
        n_pairs=10,
        seed=909,
    )
    score = sum(p.score_a for p in pairs)
    assert score > 18.0  # decisively above an even split of the 20 games
    ratings = fit_elo(matches_from_pairs("uct", "random", pairs), anchor="random")
    assert ratings["uct"] > 100.0


# --- scheduling parity: the real frozen rung through the existing runner --------


def test_uct_rung4_schedules_through_the_paired_runner_like_earlier_rungs():
    pairs = play_pairs(
        TTT,
        lambda seed: UCTAgent(seed),
        lambda seed: RandomAgent(seed),
        n_pairs=1,
        seed=555,
    )
    assert len(pairs) == 1
    assert pairs[0].score_a + pairs[0].score_b == 2.0
    ratings = fit_elo(matches_from_pairs("uct-rollout-v1", "random", pairs), anchor="random")
    assert ratings["random"] == 0.0
