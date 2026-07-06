"""Subtree-reuse + virtual-loss invariant tests (design doc §6.2, §12 M0).

Guards against stale or duplicated statistics carried across moves, and proves the
virtual-loss mechanism (which lets batched selection at M5 avoid re-picking an in-flight
leaf) both diversifies selection and leaves no residue after a completed search.
"""

from __future__ import annotations

from core import MCTS
from games.connect4 import Connect4
from games.tictactoe import TicTacToe


def _all_nodes(root):
    nodes, stack = [], [root]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        nodes.append(node)
        stack.extend(node.children)
    return nodes


def test_root_visit_accounting_and_no_virtual_loss_residue():
    game = TicTacToe()
    m = MCTS(game)
    n_sims = 500
    root = m.run(n_sims, root_state=game.initial_state())

    # Simulation 1 expands the root (no edge visit); every later sim traverses exactly
    # one root edge -> the root's visit counts sum to n_sims - 1.
    assert sum(root.N) == n_sims - 1

    for node in _all_nodes(root):
        assert all(v == 0 for v in node.vloss)  # no virtual-loss residue
        for i in range(len(node.actions)):
            if node.N[i] == 0:
                assert node.W[i] == 0.0
                assert node.Q[i] == 0.0
            else:
                assert abs(node.Q[i] - node.W[i] / node.N[i]) < 1e-12


def test_subtree_reuse_preserves_child_statistics():
    game = TicTacToe()
    m = MCTS(game)
    root = m.run(800, root_state=game.initial_state())

    action = m.best_action()
    i = root.actions.index(action)
    child = root.children[i]
    assert child is not None

    # The chosen child was the freshly-expanded leaf exactly once; every other visit
    # into it descended further, so its edge visits sum to (visits into child) - 1.
    assert sum(child.N) == root.N[i] - 1

    snap_n, snap_w = list(child.N), list(child.W)
    m.advance(action)
    assert m.root is child  # the subtree object itself is reused
    assert m.root.N == snap_n  # ... with its statistics intact
    assert m.root.W == snap_w

    # Continuing from the reused (already-expanded) root: each sim is exactly one edge
    # visit, so the counts advance by exactly the number of new simulations.
    before = sum(m.root.N)
    m.run(200)
    assert sum(m.root.N) == before + 200


def test_virtual_loss_diversifies_in_flight_selection():
    game = Connect4(4, 4, 3)  # branching factor 4 gives clear diversification
    m = MCTS(game)
    m.run(1, root_state=game.initial_state())  # expand the root only; edge stats are zero
    root = m.root
    assert sum(root.N) == 0
    assert all(v == 0 for v in root.vloss)

    # Two in-flight descents without backup: the virtual loss from the first must steer
    # the second onto a different root edge.
    path1, _, _ = m._descend(apply_vloss=True)
    i1 = path1[0][1]
    assert root.vloss[i1] == 1
    path2, _, _ = m._descend(apply_vloss=True)
    i2 = path2[0][1]
    assert i2 != i1

    # Manually unwind the two in-flight descents; no residue must remain.
    for path in (path1, path2):
        for node, i in path:
            node.vloss[i] -= m.virtual_loss
    for node in _all_nodes(root):
        assert all(v == 0 for v in node.vloss)
