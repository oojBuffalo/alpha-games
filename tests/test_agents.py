"""Network-free baseline agents: rung 1 (uniform random) and the agent seam.

M1.6 (design doc §12, §9): the frozen ladder's absolute-strength anchor. Agents
act through the public ``Game`` interface only — ``select_action`` must return
a legal action at any nonterminal state, and identical seeds must reproduce
identical choices (the pair runner drives determinism from per-pair seeds).
"""

from __future__ import annotations

from core import MobilityAgent, RandomAgent
from games.connect4 import Connect4
from games.tictactoe import TicTacToe
from tests.fixtures.pass_game import (
    PassGame,
    Scenario,
    consecutive_trap_game,
    consecutive_win_game,
)

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
    assert MobilityAgent(seed=0).name == "mobility"


# --- rung 3: 1-ply mobility greedy (§12 M1.6 pin) ------------------------------


def test_mobility_agent_takes_an_immediate_win():
    # A terminal win has absolute priority over any mobility score.
    s = GAME.from_grid(["XX.", "OO.", "..."], to_play=0)
    for seed in range(5):
        assert MobilityAgent(seed=seed).select_action(GAME, s) == 2


def test_mobility_agent_prefers_win_over_loss_at_terminals():
    g = consecutive_trap_game()
    m = g.initial_state()
    m = g.apply(m, 0)  # forced into the consecutive node: 0 wins, 1 loses
    for seed in range(5):
        assert MobilityAgent(seed=seed).select_action(g, m) == 0


def test_mobility_agent_prefers_blocking_the_opponent():
    # consecutive_win_game root: action 0 leads to a node where the *mover*
    # moves again (opponent skipped, mobility +2); action 1 hands the opponent
    # two replies (mobility -2). The pinned rule must pick action 0.
    g = consecutive_win_game()
    for seed in range(5):
        assert MobilityAgent(seed=seed).select_action(g, g.initial_state()) == 0


def test_mobility_agent_minimizes_opponent_replies_on_connect4():
    # Column 0 holds five discs: completing it leaves the opponent 6 replies,
    # anything else leaves 7 — the pinned rule must fill the column.
    c4 = Connect4()
    s = c4.from_moves([0, 0, 0, 0, 0])
    reply_counts = {a: len(c4.legal_moves(c4.apply(s, a))) for a in c4.legal_moves(s)}
    assert sorted(set(reply_counts.values())) == [6, 7]
    for seed in range(5):
        assert MobilityAgent(seed=seed).select_action(c4, s) == 0


def test_mobility_agent_plays_on_rather_than_taking_a_draw():
    # A terminal draw ranks below every nonterminal successor: from state 0, P0
    # can lock an immediate tie (action 0 -> terminal (0, 0)) or play on into a
    # live position where the opponent replies next (action 1). The pinned order
    # WIN > nonterminal > DRAW > LOSS must keep the game going.
    g = PassGame(
        Scenario(
            start=0,
            to_play={0: 0, 2: 1},
            edges={0: [(0, 1), (1, 2)], 2: [(0, 3)]},
            terminal={1: (0.0, 0.0), 3: (1.0, -1.0)},
        )
    )
    for seed in range(5):
        assert MobilityAgent(seed=seed).select_action(g, g.initial_state()) == 1


def test_mobility_agent_prefers_a_draw_over_a_loss():
    # ...but a terminal draw still outranks a terminal loss.
    g = PassGame(
        Scenario(
            start=0,
            to_play={0: 0},
            edges={0: [(0, 1), (1, 2)]},
            terminal={1: (0.0, 0.0), 2: (-1.0, 1.0)},
        )
    )
    for seed in range(5):
        assert MobilityAgent(seed=seed).select_action(g, g.initial_state()) == 0


def test_mobility_agent_breaks_ties_by_seed_deterministically():
    # Every TTT opening leaves the opponent exactly 8 replies: a full tie.
    s0 = GAME.initial_state()
    picks = {MobilityAgent(seed=s).select_action(GAME, s0) for s in range(30)}
    assert len(picks) > 1  # ties are randomized...
    a = MobilityAgent(seed=4).select_action(GAME, s0)
    b = MobilityAgent(seed=4).select_action(GAME, s0)
    assert a == b  # ...but reproducible per seed
