"""MCTS-vs-minimax on Othello endgames: backup through forced passes (M1.5).

The §12 M1.5 "consecutive-mover backup on a real game" check, layered on M0's
synthetic pass fixtures: around a forced pass the *placement* sequence goes
consecutive (place → opponent passes → place again), and search must carry
values through the pass ply with the right sign. Endgame fixtures are frozen
from seeded random playouts off the standard start; the max-n reference solver
provides ground truth, so nothing here is hand-scored.
"""

from __future__ import annotations

import pytest

from core import MCTS
from games.othello import PASS, Othello
from tests.reference.minimax import optimal_values, reachable_states, subtree_size

GAME = Othello()

# Nonterminal 7-empty endgames reached by seeded random play from the standard
# start; each subtree is solver-sized and contains forced-pass states.
ENDGAMES = [
    (
        [
            "BBBBBB..",
            ".BBBBBBB",
            "BBBBBWBW",
            "BBBBWBW.",
            "BBWWWWWW",
            "BWBWWWBW",
            "BBBWWBB.",
            "B.BW.BBB",
        ],
        1,
    ),
    (
        [
            "WB.W.B.B",
            "WWWWW..B",
            "WBWWWWBB",
            "BBBWWBWB",
            ".BWWWBBB",
            "WWBWWWBB",
            ".WBBBBBB",
            "WWWWWWWW",
        ],
        1,
    ),
    (
        [
            "WWWWWWWW",
            "BWWWWBWW",
            "BWWWBBWW",
            "BWWWWBBW",
            "..WWBBBW",
            "..WWWWBW",
            "..BWWWWW",
            ".BBBBBBW",
        ],
        0,
    ),
]


def _mcts_move(state, sims):
    m = MCTS(GAME)
    m.run(sims, root_state=state)
    return m.best_action()


@pytest.mark.parametrize("rows,to_play", ENDGAMES, ids=["end-w1", "end-w2", "end-b"])
def test_mcts_plays_optimally_on_pass_endgames(rows, to_play):
    root = GAME.from_grid(rows, to_play)
    states = reachable_states(GAME, root)
    # The property is only meaningful if the swept region really contains
    # forced passes (backup must cross pass plies, not just alternation).
    assert any(
        not GAME.is_terminal(s) and list(GAME.legal_moves(s)) == [PASS] for s in states
    )
    value_cache: dict = {}
    size_cache: dict = {}
    candidates = [
        (s, subtree_size(GAME, s, size_cache))
        for s in states
        if not GAME.is_terminal(s) and subtree_size(GAME, s, size_cache) <= 200
    ]
    step = max(1, len(candidates) // 15)
    tested = 0
    for s, sz in candidates[::step]:
        mover = GAME.current_player(s)
        target = optimal_values(GAME, s, value_cache)[mover]
        action = _mcts_move(s, min(1500, max(240, 30 * sz)))
        achieved = optimal_values(GAME, GAME.apply(s, action), value_cache)[mover]
        assert achieved >= target - 1e-9, (
            f"MCTS blundered: chose {action} (value {achieved}) < optimal {target}"
        )
        tested += 1
    assert tested >= 10


# Decisive consecutive-placement states (found by solver search inside the
# endgame subtrees): the unique winning move forces the opponent to pass, so
# the mover places twice in a row; the only alternative loses outright.
BACKUP_CASES = [
    (
        [
            "BBBBBBBW",
            "WWWWWWWW",
            "BWBBBWWW",
            "BBWBBBWW",
            "BBWWBWWW",
            "BWBWBWBW",
            "BBBBBBB.",
            "B.BBBBBB",
        ],
        1,  # White to move
        57,  # winning: opponent must pass, White completes at 55
        55,  # losing alternative
    ),
    (
        [
            "WWWW.BBB",
            "WWWWW.BB",
            "WBWWWWBB",
            "WWBWWBBB",
            "WWWWWBBB",
            "WWBWWWBB",
            "BBBBBBBB",
            "WWWWWWWW",
        ],
        0,  # Black to move
        13,  # winning through the opponent's forced pass
        4,  # losing alternative
    ),
]


@pytest.mark.parametrize("rows,to_play,win,lose", BACKUP_CASES, ids=["white-mover", "black-mover"])
def test_backup_through_forced_pass_prefers_winning_line(rows, to_play, win, lose):
    s = GAME.from_grid(rows, to_play)
    assert sorted(GAME.legal_moves(s)) == sorted([win, lose])

    # Structure: the winning move forces a pass, then the same player places
    # again — the consecutive-placement shape the backup must survive.
    after_win = GAME.apply(s, win)
    assert list(GAME.legal_moves(after_win)) == [PASS]
    assert GAME.current_player(GAME.apply(after_win, PASS)) == to_play

    # Solver ground truth: win is +1 for the mover, lose is at most a draw.
    cache: dict = {}
    assert optimal_values(GAME, after_win, cache)[to_play] == 1.0
    assert optimal_values(GAME, GAME.apply(s, lose), cache)[to_play] == -1.0

    m = MCTS(GAME)
    m.run(800, root_state=s)
    assert m.best_action() == win
    q = m.action_values()
    # Q in the mover's perspective: winning branch positive, losing negative.
    # A backup that mishandles the pass ply inverts or dilutes these signs.
    assert q[win] > 0.0 > q[lose]


def test_mcts_smoke_with_subtree_advance_from_the_start():
    m = MCTS(GAME)
    s0 = GAME.initial_state()
    m.run(300, root_state=s0)
    a = m.best_action()
    assert a in list(GAME.legal_moves(s0))
    m.advance(a)
    m.run(300)
    assert m.best_action() in list(GAME.legal_moves(GAME.apply(s0, a)))
