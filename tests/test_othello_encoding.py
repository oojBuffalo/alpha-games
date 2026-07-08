"""Othello encoding surface (M1.5-carried, §12 M1.5 scope note).

M1.5 stands up Othello's own encoding — the flat 64+1 action codec,
``policy_shape``, and the 2 mover-relative input planes — M2-class machinery
pulled forward for the second game. Conventions per the §12 M1.5 pin block:
``a = r*8 + c``, pass = 64, planes (own, opponent) as nested tuples, ``T=1``.
"""

from __future__ import annotations

from games.othello import PASS, Othello

GAME = Othello()


def test_policy_shape_is_flat_65():
    assert GAME.policy_shape == (65,)


def test_action_codec_bijection_over_the_full_head():
    for a in range(64):
        move = GAME.decode_action(a)
        assert move == divmod(a, 8)  # (r, c), row-major
        assert GAME.encode_action(move) == a
    assert GAME.decode_action(PASS) == "pass"
    assert GAME.encode_action("pass") == PASS


def test_action_codec_literal_goldens():
    # Hand-derived pins so a consistent-but-wrong convention can't survive.
    assert GAME.encode_action((0, 0)) == 0
    assert GAME.encode_action((0, 7)) == 7
    assert GAME.encode_action((7, 0)) == 56
    assert GAME.encode_action((2, 3)) == 19  # the d3 opening
    assert GAME.decode_action(63) == (7, 7)


def test_input_planes_declared():
    assert GAME.input_planes == 2


def test_encode_state_is_mover_relative():
    s0 = GAME.initial_state()
    own, opp = GAME.encode_state(s0)
    assert len(own) == 8 and all(len(row) == 8 for row in own)
    # Black to move: own = Black discs, opponent = White discs.
    assert {(r, c) for r in range(8) for c in range(8) if own[r][c]} == {(3, 4), (4, 3)}
    assert {(r, c) for r in range(8) for c in range(8) if opp[r][c]} == {(3, 3), (4, 4)}

    # After Black plays d3, White is the mover: the planes swap perspective.
    s1 = GAME.apply(s0, 19)
    own1, opp1 = GAME.encode_state(s1)
    assert {(r, c) for r in range(8) for c in range(8) if own1[r][c]} == {(4, 4)}
    assert {(r, c) for r in range(8) for c in range(8) if opp1[r][c]} == {
        (2, 3),
        (3, 3),
        (3, 4),
        (4, 3),
    }
