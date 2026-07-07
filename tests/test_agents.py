"""Network-free baseline agents: rung 1 (uniform random) and the agent seam.

M1.6 (design doc §12, §9): the frozen ladder's absolute-strength anchor. Agents
act through the public ``Game`` interface only — ``select_action`` must return
a legal action at any nonterminal state, and identical seeds must reproduce
identical choices (the pair runner drives determinism from per-pair seeds).
"""

from __future__ import annotations

from core import RandomAgent
from games.tictactoe import TicTacToe

GAME = TicTacToe()


def test_random_agent_plays_legal_full_game():
    agent = RandomAgent(seed=0)
    s = GAME.initial_state()
    plies = 0
    while not GAME.is_terminal(s):
        a = agent.select_action(GAME, s)
        assert a in list(GAME.legal_moves(s))
        s = GAME.apply(s, a)
        plies += 1
    assert 5 <= plies <= 9


def test_random_agent_is_deterministic_per_seed():
    s0 = GAME.initial_state()
    picks_a = [RandomAgent(seed=42).select_action(GAME, s0) for _ in range(3)]
    picks_b = [RandomAgent(seed=42).select_action(GAME, s0) for _ in range(3)]
    assert picks_a[0] == picks_b[0]
    # A fresh agent restarts its stream; the same agent continues its stream.
    agent = RandomAgent(seed=42)
    stream = [agent.select_action(GAME, s0) for _ in range(20)]
    assert len(set(stream)) > 1  # actually random over the 9 openings


def test_random_agent_is_uniform_over_legal_moves():
    agent = RandomAgent(seed=7)
    s0 = GAME.initial_state()
    counts = {a: 0 for a in GAME.legal_moves(s0)}
    n = 3000
    for _ in range(n):
        counts[agent.select_action(GAME, s0)] += 1
    for a, c in counts.items():
        assert abs(c / n - 1 / 9) < 0.03, (a, c)


def test_agent_has_a_name():
    assert RandomAgent(seed=0).name == "random"
