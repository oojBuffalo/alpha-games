"""Player-aware (consecutive-mover) backup test (design doc §6.2, §12 M0).

TTT/C4 strictly alternate, so the player-aware backup — the sign rule that must survive
one player moving twice in a row — needs the synthetic pass fixtures. A backup that
wrongly assumes alternation would negate the value across the consecutive move and
misplay; these tests pin the sign end to end against the max-n solver.
"""

from __future__ import annotations

from core import MCTS
from tests.fixtures.pass_game import consecutive_trap_game, consecutive_win_game
from tests.reference.minimax import optimal_actions


def test_consecutive_win_backup_signs():
    g = consecutive_win_game()
    m = MCTS(g)
    m.run(3000, root_state=g.initial_state())

    # Root: player 0 should steer into the consecutive-move branch (action 0).
    assert m.best_action() == 0
    assert m.best_action() in optimal_actions(g, g.initial_state())

    q = m.action_values()
    # Sign is measured in the mover's (player 0's) perspective: the winning branch is
    # positive, the losing branch negative. A sign bug would invert this.
    assert q[0] > 0.0 > q[1]

    # The consecutive (player-0-again) node must itself play the winning move.
    m.advance(0)
    m.run(3000)
    assert m.best_action() == 0
    q2 = m.action_values()
    assert q2[0] > 0.0 > q2[1]


def test_consecutive_trap_backup_avoids_self_loss():
    g = consecutive_trap_game()
    m = MCTS(g)
    m.run(500, root_state=g.initial_state())
    m.advance(0)  # forced single move into the consecutive node
    m.run(3000)

    # At the consecutive node, player 0 must avoid the trap (action 1 self-destructs).
    assert m.best_action() == 0
    q = m.action_values()
    assert q[0] > 0.0 > q[1]
