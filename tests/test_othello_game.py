"""Othello rules: openings, flips, explicit pass, termination, disc-diff utility.

M1.5 abstraction test (design doc §12 M1.5): Othello realizes the §6.1 pass
invariant the *other* way — an explicit pass action (id 64) in a flat 64+1 head,
with strict alternation at the action level while the placement sequence goes
non-monotone (a passed player can regain a move). Conventions per the §12 M1.5
pin block: 0-indexed row-major, ``a = r*8 + c``, Black = player 0 moves first.
"""

from __future__ import annotations

from games.othello import PASS, Othello

GAME = Othello()


def _discs(board):
    black = {divmod(i, 8) for i, v in enumerate(board) if v == 0}
    white = {divmod(i, 8) for i, v in enumerate(board) if v == 1}
    return black, white


def test_initial_position():
    s = GAME.initial_state()
    board, to_play = s
    black, white = _discs(board)
    assert black == {(3, 4), (4, 3)}
    assert white == {(3, 3), (4, 4)}
    assert to_play == 0  # Black = player 0 moves first (§12 M1.5 pin)
    assert GAME.current_player(s) == 0
    assert not GAME.is_terminal(s)


def test_opening_moves():
    # Black's four standard openings: (2,3), (3,2), (4,5), (5,4) as r*8+c.
    assert sorted(GAME.legal_moves(GAME.initial_state())) == [19, 26, 37, 44]


def test_apply_opening_flips_and_alternates():
    s = GAME.apply(GAME.initial_state(), 19)  # Black d3: places (2,3), flips (3,3)
    board, to_play = s
    black, white = _discs(board)
    assert black == {(2, 3), (3, 3), (3, 4), (4, 3)}
    assert white == {(4, 4)}
    assert to_play == 1  # apply always hands the move to the opponent
    # White's standard replies: (2,2), (2,4), (4,2).
    assert sorted(GAME.legal_moves(s)) == [18, 20, 34]


def test_apply_flips_multiple_directions():
    s = GAME.from_grid(
        [
            "B.B.....",
            ".WW.....",
            "BW......",
            "........",
            "........",
            "........",
            "........",
            "........",
        ],
        to_play=0,
    )
    nxt = GAME.apply(s, 18)  # Black (2,2): flips (2,1) left, (1,2) up, (1,1) diagonal
    board, to_play = nxt
    black, white = _discs(board)
    assert black == {(0, 0), (0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)}
    assert white == set()
    assert to_play == 1


def test_apply_does_not_mutate_input():
    s0 = GAME.initial_state()
    before = s0[0]
    GAME.apply(s0, 19)
    assert s0[0] == before
