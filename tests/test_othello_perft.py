"""Othello perft goldens against the published reference table.

Reference: Aart Bik, "Perft for Reversi" (aartbik.blogspot.com, 2009) —
depths 1..12 from the standard start: 4, 12, 56, 244, 1396, 8200, 55092,
390216, 3005288, ... Passing moves first occur at depth 9, so depths <= 8 are
independent of the pass-counting convention; at deeper depths that table counts
a pass as a node, which matches this adapter's explicit-pass action exactly.

A wrong flip direction, a missed flanking line, or a bad legality rule shifts
these counts immediately — the cheapest whole-move-generator differential we
have for a game with no second (oracle) engine.
"""

from __future__ import annotations

import pytest

from games.othello import Othello

GAME = Othello()

# (depth, leaf count) from the published table; depths here are pass-free.
PERFT_TABLE = {1: 4, 2: 12, 3: 56, 4: 244, 5: 1_396, 6: 8_200, 7: 55_092}


def perft(state, depth: int) -> int:
    """Count length-``depth`` action sequences from ``state`` (terminals prune)."""
    if depth == 0:
        return 1
    if GAME.is_terminal(state):
        return 0
    total = 0
    for a in GAME.legal_moves(state):
        total += perft(GAME.apply(state, a), depth - 1)
    return total


@pytest.mark.parametrize("depth", sorted(PERFT_TABLE))
def test_perft_matches_published_table(depth):
    # Depth 7 runs in ~1 s pure Python — cheap enough for the non-slow battery.
    assert perft(GAME.initial_state(), depth) == PERFT_TABLE[depth]
