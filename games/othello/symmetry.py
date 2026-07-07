"""Full D4 symmetry of Othello: cell maps, 65-head action permutations, plane transforms.

Element order and orientation per the §12 M1.5 pin block: ``(identity, rot90,
rot180, rot270, flip_h, flip_v, diag, antidiag)`` with rot90 = clockwise. The
pass action (id 64) is a fixed point of every permutation, and each permutation
is a bijection of the flat 65-action head.

Unlike Blokus (whose plane slot is a raising sentinel until M2), M1.5 carries
Othello's own encoding, so :func:`plane_transform` is the *real* first slot of
the adapter's ``symmetry_group``. :func:`state_transform` is a module utility
over ``(board, to_play)`` state tuples for the equivariance tests.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

BOARD_SIZE = 8
PASS = 64
_LAST = BOARD_SIZE - 1

GROUP_NAMES: tuple[str, ...] = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "diag",
    "antidiag",
)

_CELL_MAPS: dict[str, Callable[[int, int], tuple[int, int]]] = {
    "identity": lambda r, c: (r, c),
    "rot90": lambda r, c: (c, _LAST - r),  # clockwise (§12 M1.5 pin)
    "rot180": lambda r, c: (_LAST - r, _LAST - c),
    "rot270": lambda r, c: (_LAST - c, r),
    "flip_h": lambda r, c: (r, _LAST - c),
    "flip_v": lambda r, c: (_LAST - r, c),
    "diag": lambda r, c: (c, r),
    "antidiag": lambda r, c: (_LAST - c, _LAST - r),
}


def cell_map(name: str) -> Callable[[int, int], tuple[int, int]]:
    """Return the cell map ``(r, c) -> (r', c')`` for group element ``name``."""
    return _CELL_MAPS[name]


@cache
def action_permutation(name: str) -> tuple[int, ...]:
    """Return the 65-length policy-head permutation for group element ``name``.

    Placement ids map through the cell map; pass (id 64) maps to itself — the
    §12 M1.5 pass-id fixed-point requirement.

    Args:
        name: Group element name from :data:`GROUP_NAMES`.

    Returns:
        Tuple ``perm`` with ``perm[a]`` the image of action ``a``.
    """
    m = _CELL_MAPS[name]
    perm = []
    for a in range(64):
        r, c = divmod(a, BOARD_SIZE)
        tr, tc = m(r, c)
        perm.append(tr * BOARD_SIZE + tc)
    perm.append(PASS)
    return tuple(perm)


def plane_transform(name: str) -> Callable:
    """Return the plane-tensor transform for group element ``name``.

    Operates on ``encode_state`` output — a tuple of 8×8 nested-tuple planes —
    moving the value at ``(r, c)`` to the image cell in every plane. Mover
    perspective is untouched (board symmetry, no player relabeling).

    Args:
        name: Group element name from :data:`GROUP_NAMES`.

    Returns:
        Callable mapping a plane tuple to its transformed plane tuple.
    """
    m = _CELL_MAPS[name]

    def transform(planes):
        out = []
        for plane in planes:
            grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    tr, tc = m(r, c)
                    grid[tr][tc] = plane[r][c]
            out.append(tuple(tuple(row) for row in grid))
        return tuple(out)

    return transform


def state_transform(name: str) -> Callable:
    """Return a state-level transform for group element ``name``.

    Module utility for the equivariance tests (not part of ``symmetry_group``):
    maps ``(board, to_play)`` to the transformed board with ``to_play`` fixed.

    Args:
        name: Group element name from :data:`GROUP_NAMES`.

    Returns:
        Callable mapping a state tuple to its transformed state tuple.
    """
    m = _CELL_MAPS[name]

    def transform(state):
        board, to_play = state
        tboard = [0] * 64
        for i in range(64):
            r, c = divmod(i, BOARD_SIZE)
            tr, tc = m(r, c)
            tboard[tr * BOARD_SIZE + tc] = board[i]
        return (tuple(tboard), to_play)

    return transform
