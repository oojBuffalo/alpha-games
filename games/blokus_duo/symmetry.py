"""Klein-4 symmetry of Blokus Duo: cell maps, the (g,a)→a′ table, head permutations.

The group is {identity, 180°, main diagonal, anti-diagonal} — the set-stabilizer
of the start squares {(4,4),(9,9)} with no own/opponent relabeling (design doc
§8; full D4 is deferred because 90°-class images are off-support). Action maps
are built decode → transform cells → re-encode, which is ``anchor(g(cells))``
by construction and so immune to the doc's named failure mode
``g(anchor) != anchor(g(cells))`` (naive anchor transport is wrong for most
ids under 180°).

[F6, revised per PR #2 review]: the adapter's ``symmetry_group`` elements carry
a *raising sentinel* in the first slot — core documents that slot as a plane
transform, which cannot exist before M2's ``encode_state``, so advertising an
engine-state transform there would hand generic augmentation the wrong
representation. M2 replaces the sentinel with the real plane transform. The
second slot is the full 17,836-length permutation, with off-support ids mapped
to themselves (documented never-legal filler). :func:`state_transform` stays a
module-level utility over engine-state tuples for the M1 equivariance tests.
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


def plane_transform_placeholder(state):
    """Raising sentinel for the plane-transform slot of ``symmetry_group``.

    Core documents the first ``SymmetryElement`` slot as a state-*plane*
    transform; the plane encoding does not exist until M2's ``encode_state``,
    so until then the slot fails loudly instead of advertising a transform
    over the wrong representation (PR #2 review).

    Args:
        state: Ignored.

    Raises:
        NotImplementedError: Always — the plane transform lands with M2.
    """
    raise NotImplementedError(
        "plane-side symmetry transform lands with M2 encode_state; for "
        "engine-state tuples use games.blokus_duo.symmetry.state_transform"
    )


def state_transform(name: str) -> Callable:
    """Return an engine-state-level transform for group element ``name`` [F6].

    Module utility (used by the M1 equivariance tests), deliberately *not*
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
