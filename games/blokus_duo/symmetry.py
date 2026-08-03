"""Klein-4 symmetry of Blokus Duo: cell maps, (g,a)→a′ table, head perms, plane transforms.

The group is {identity, 180°, main diagonal, anti-diagonal} — the set-stabilizer
of the start squares {(4,4),(9,9)} with no own/opponent relabeling (design doc
§8; full D4 is deferred because 90°-class images are off-support). Action maps
are built decode → transform cells → re-encode, which is ``anchor(g(cells))``
by construction and so immune to the doc's named failure mode
``g(anchor) != anchor(g(cells))`` (naive anchor transport is wrong for most
ids under 180°).

[F6, revised per PR #2 review]: the adapter's ``symmetry_group`` elements pair
:func:`plane_transform` — over the 46-plane ``encode_state`` output; the slot
was a raising sentinel until M2 landed the plane encoding — with the full
17,836-length permutation, off-support ids mapped to themselves (documented
never-legal filler). :func:`state_transform` stays a module-level utility over
engine-state tuples for the M1/M2 equivariance tests.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

from games.blokus_duo.actions import (
    BOARD_SIZE,
    IN_BOUNDS_ACTIONS,
    NUM_ACTIONS,
    action_cells,
    encode_cells,
)

_LAST = BOARD_SIZE - 1

# Element order is the adapter's group order; identity is element 0 (D9 counts
# it among the 4 augmentation symmetries).
GROUP_NAMES: tuple[str, ...] = ("identity", "rot180", "diag", "antidiag")

_CELL_MAPS: dict[str, Callable[[int, int], tuple[int, int]]] = {
    "identity": lambda r, c: (r, c),
    "rot180": lambda r, c: (_LAST - r, _LAST - c),
    "diag": lambda r, c: (c, r),
    "antidiag": lambda r, c: (_LAST - c, _LAST - r),
}


def cell_map(name: str) -> Callable[[int, int], tuple[int, int]]:
    """Return the cell map ``(r, c) -> (r', c')`` for group element ``name``."""
    return _CELL_MAPS[name]


def transform_action(name: str, action: int) -> int:
    """Map an in-bounds action id through group element ``name``.

    Decode → transform cells → re-encode: the resulting anchor is the bbox
    top-left of the transformed cells, never a transported anchor.

    Args:
        name: Group element name from :data:`GROUP_NAMES`.
        action: In-bounds flat action id.

    Returns:
        The image action id (always in-bounds: the group stabilizes the board).
    """
    m = _CELL_MAPS[name]
    return encode_cells([m(r, c) for r, c in action_cells(action)])


@cache
def build_action_maps() -> dict[str, dict[int, int]]:
    """Build the (g,a)→a′ maps over all in-bounds ids for every group element.

    Returns:
        Per group element, a dict from in-bounds action id to its image.
    """
    return {g: {a: transform_action(g, a) for a in IN_BOUNDS_ACTIONS} for g in GROUP_NAMES}


@cache
def full_permutation(name: str) -> tuple[int, ...]:
    """Return the full 17,836-length policy-head permutation for ``name``.

    Off-support (out-of-bounds) ids map to themselves — identity filler for
    slots that are never legal, so the permutation is total over the head.

    Args:
        name: Group element name from :data:`GROUP_NAMES`.

    Returns:
        Tuple ``perm`` with ``perm[a]`` the image of action ``a``.
    """
    perm = list(range(NUM_ACTIONS))
    for a, image in build_action_maps()[name].items():
        perm[a] = image
    return tuple(perm)


def plane_transform(name: str) -> Callable:
    """Return the plane-tensor transform for group element ``name``.

    Operates on ``encode_state`` output — 46 nested-tuple 14×14 planes (D3) —
    moving the value at ``(r, c)`` to the image cell in every plane. Constant
    inventory/flag broadcast planes are invariant under the map but go through
    the same code path (no special-casing); mover perspective is untouched
    (board symmetry, no player relabeling).

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
    """Return an engine-state-level transform for group element ``name`` [F6].

    Module utility (used by the M1/M2 equivariance tests), deliberately *not*
    exposed through the adapter's ``symmetry_group``. Works on the shared
    engine state tuple with occupancies as either frozensets of cells
    (oracle) or 196-bit ints (bitboard); inventories, flags, and ``to_play``
    are invariant under board symmetry (no player relabeling).

    Args:
        name: Group element name from :data:`GROUP_NAMES`.

    Returns:
        Callable mapping a state tuple to its transformed state tuple.
    """
    m = _CELL_MAPS[name]

    def transform_occ(occ):
        if isinstance(occ, int):
            bb = 0
            for i in range(BOARD_SIZE * BOARD_SIZE):
                if occ >> i & 1:
                    r, c = divmod(i, BOARD_SIZE)
                    tr, tc = m(r, c)
                    bb |= 1 << (tr * BOARD_SIZE + tc)
            return bb
        return frozenset(m(r, c) for r, c in occ)

    def transform(state):
        return (transform_occ(state[0]), transform_occ(state[1]), *state[2:])

    return transform
