"""Othello D4 symmetry: 8 elements, pass fixed-point, real plane transforms.

§12 M1.5: the full D4 group (8 elements) exercised through ``symmetry_group`` —
including the pass-id fixed-point check (perm[64] == 64 for every element) and
closure/bijectivity on the 65-action head. Unlike Blokus (plane slot is an M2
sentinel), M1.5 carries its own encoding, so the plane transforms are real and
tested for equivariance against ``encode_state``.

Independence (the M1 [F4] pattern): equivariance properties transform *states*
with cell maps hardcoded here, and *actions* only through the adapter's
declared permutations.
"""

from __future__ import annotations

import random

from games.othello import PASS, Othello
from games.othello.symmetry import (
    GROUP_NAMES,
    action_permutation,
    plane_transform,
    state_transform,
)

GAME = Othello()

# D4 cell maps hardcoded independently of symmetry.py, per the §12 M1.5 pin
# block (rot90 = clockwise).
_TEST_MAPS = {
    "identity": lambda r, c: (r, c),
    "rot90": lambda r, c: (c, 7 - r),
    "rot180": lambda r, c: (7 - r, 7 - c),
    "rot270": lambda r, c: (7 - c, r),
    "flip_h": lambda r, c: (r, 7 - c),
    "flip_v": lambda r, c: (7 - r, c),
    "diag": lambda r, c: (c, r),
    "antidiag": lambda r, c: (7 - c, 7 - r),
}


def test_group_order_and_names_pinned():
    assert GROUP_NAMES == (
        "identity",
        "rot90",
        "rot180",
        "rot270",
        "flip_h",
        "flip_v",
        "diag",
        "antidiag",
    )


def test_cell_map_group_laws():
    m90, m180, m270 = _TEST_MAPS["rot90"], _TEST_MAPS["rot180"], _TEST_MAPS["rot270"]
    mdiag, manti = _TEST_MAPS["diag"], _TEST_MAPS["antidiag"]
    # rot90 is clockwise: top-left corner -> top-right corner.
    assert m90(0, 0) == (0, 7)
    assert m90(0, 7) == (7, 7)
    for r in range(8):
        for c in range(8):
            assert m90(*m90(r, c)) == m180(r, c)
            assert m90(*m180(r, c)) == m270(r, c)
            assert m90(*m270(r, c)) == (r, c)  # rot90 has order 4
            for g in ("rot180", "flip_h", "flip_v", "diag", "antidiag"):
                assert _TEST_MAPS[g](*_TEST_MAPS[g](r, c)) == (r, c)  # involutions
            assert manti(r, c) == m180(*mdiag(r, c))


def test_action_perms_pass_fixed_point_closed_bijective():
    for g in GROUP_NAMES:
        perm = action_permutation(g)
        assert len(perm) == 65
        assert perm[PASS] == PASS  # pass is a fixed point of every element
        assert sorted(perm) == list(range(65))  # closed + bijective on the head
        m = _TEST_MAPS[g]
        for a in range(64):
            r, c = divmod(a, 8)
            tr, tc = m(r, c)
            assert perm[a] == tr * 8 + tc, (g, a)


def test_adapter_declares_full_d4_with_real_plane_transforms():
    group = GAME.symmetry_group
    assert len(group) == 8
    s0 = GAME.initial_state()
    for name, (plane_t, perm) in zip(GROUP_NAMES, group, strict=True):
        assert tuple(perm) == action_permutation(name)
        # Real plane transform (not a sentinel): identity on the symmetric start.
        transformed = plane_t(GAME.encode_state(s0))
        assert len(transformed) == 2
        if name in ("identity", "rot180"):
            assert transformed == GAME.encode_state(s0)


def _random_states(n_games, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n_games):
        s = GAME.initial_state()
        while not GAME.is_terminal(s):
            out.append(s)
            s = GAME.apply(s, rng.choice(list(GAME.legal_moves(s))))
        out.append(s)
    return out


def test_move_gen_equivariance_on_random_states():
    # legal(g(s)) == g(legal(s)), states transformed with the test's own maps,
    # actions through the adapter's declared permutations only.
    states = _random_states(3, seed=5)[::7]
    perms = {name: perm for name, (_, perm) in zip(GROUP_NAMES, GAME.symmetry_group, strict=True)}
    for s in states:
        legal = list(GAME.legal_moves(s))
        for g in GROUP_NAMES:
            m = _TEST_MAPS[g]
            board = s[0]
            tboard = [None] * 64
            for i in range(64):
                r, c = divmod(i, 8)
                tr, tc = m(r, c)
                tboard[tr * 8 + tc] = board[i]
            ts = (tuple(tboard), s[1])
            assert sorted(GAME.legal_moves(ts)) == sorted(perms[g][a] for a in legal), g


def test_plane_transform_equivariance_on_random_states():
    # encode_state(g·s) == plane_transform_g(encode_state(s)).
    states = _random_states(2, seed=11)[::5]
    for s in states:
        planes = GAME.encode_state(s)
        for g in GROUP_NAMES:
            assert GAME.encode_state(state_transform(g)(s)) == plane_transform(g)(planes), g


def test_apply_equivariance_on_random_states():
    # g·apply(s, a) == apply(g·s, g·a) for every legal a, including passes.
    states = [st for st in _random_states(2, seed=23)[::9] if not GAME.is_terminal(st)]
    assert states
    for s in states:
        for g in GROUP_NAMES:
            t = state_transform(g)
            perm = action_permutation(g)
            for a in GAME.legal_moves(s):
                assert t(GAME.apply(s, a)) == GAME.apply(t(s), perm[a]), (g, a)
