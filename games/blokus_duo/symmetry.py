"""Per-instance board symmetry: cell maps, (g,a)→a′ table, head perms, plane transforms.

The group is **computed**, never hardcoded (§8): it is the D4 set-stabilizer of
*this instance's* start squares, with no own/opponent relabeling. The full game's
{(4,4),(9,9)} and the §5.3 micro instance's {(1,1),(3,3)} both sit on the main
diagonal and symmetric about the centre, so both yield the Klein four-group
{identity, rot180, diag, antidiag}; the 90°-class elements are dropped because
they move a start square off the pair (their images are rule-consistent but
off-support). Element order is the :data:`_D4_ELEMENTS` order filtered, which
reproduces the M1 full-game order and names exactly.

Action maps are built decode → transform cells → re-encode, which is
``anchor(g(cells))`` by construction and so immune to the doc's named failure
mode ``g(anchor) != anchor(g(cells))`` (naive anchor transport is wrong for most
ids under 180°).

[F6, revised per PR #2 review]: the adapter's ``symmetry_group`` elements pair
:meth:`SymmetryGroup.plane_transform` — over the ``encode_state`` plane tensor;
the slot was a raising sentinel until M2 landed the plane encoding — with the
full ``H*W*K``-length permutation, off-support ids mapped to themselves
(documented never-legal filler). :meth:`SymmetryGroup.state_transform` stays a
utility over engine-state tuples for the M1/M2 equivariance tests.

A :class:`SymmetryGroup` carries one instance's group and tables; the
module-level names at the bottom are the full game's, so every M1/M2 caller
keeps importing exactly what it did before (the same pattern as
``actions.ActionCodec``).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

from games.blokus_duo.actions import ActionCodec, action_codec
from games.blokus_duo.config import FULL_CONFIG, BlokusConfig

CellMap = Callable[[int, int], tuple[int, int]]

# The eight D4 elements as builders over ``last = board_size - 1``, in the order
# that fixes the group's element order once filtered. Rotations are clockwise.
# The full game's stabilizer is elements 0, 2, 4, 5 → ("identity", "rot180",
# "diag", "antidiag"), the M1 order and names, unchanged.
_D4_ELEMENTS: tuple[tuple[str, Callable[[int], CellMap]], ...] = (
    ("identity", lambda n: lambda r, c: (r, c)),
    ("rot90", lambda n: lambda r, c: (c, n - r)),
    ("rot180", lambda n: lambda r, c: (n - r, n - c)),
    ("rot270", lambda n: lambda r, c: (n - c, r)),
    ("diag", lambda n: lambda r, c: (c, r)),
    ("antidiag", lambda n: lambda r, c: (n - c, n - r)),
    ("flipv", lambda n: lambda r, c: (n - r, c)),
    ("fliph", lambda n: lambda r, c: (r, n - c)),
)


class SymmetryGroup:
    """One instance's declared symmetry group and its derived permutations.

    Args:
        config: The instance config. Its board size fixes the D4 cell maps and
            its start squares select the stabilizing subgroup.

    Attributes:
        config: The config this group was built from.
        names: Group element names in element order; ``names[0]`` is always
            ``"identity"`` (D9 counts it among the 4 augmentation symmetries).
        codec: The instance's :class:`~games.blokus_duo.actions.ActionCodec` —
            the action space the ``(g,a)→a′`` table permutes.
    """

    def __init__(self, config: BlokusConfig):
        self.config = config
        self.codec: ActionCodec = action_codec(config)
        last = config.board_size - 1
        starts = set(config.start_squares)
        # The set-stabilizer: keep g iff it permutes the start squares among
        # themselves (a subgroup of D4 by construction — every element also
        # stabilizes the square board).
        maps = {name: build(last) for name, build in _D4_ELEMENTS}
        self.names: tuple[str, ...] = tuple(
            name for name, _ in _D4_ELEMENTS if {maps[name](*sq) for sq in starts} == starts
        )
        self._cell_maps: dict[str, CellMap] = {name: maps[name] for name in self.names}
        self._action_maps: dict[str, dict[int, int]] | None = None
        self._permutations: dict[str, tuple[int, ...]] = {}

    def cell_map(self, name: str) -> CellMap:
        """Return the cell map ``(r, c) -> (r', c')`` for group element ``name``.

        Args:
            name: Group element name from :attr:`names`.

        Returns:
            The element's cell map.

        Raises:
            KeyError: If ``name`` is not an element of this instance's group.
        """
        return self._cell_maps[name]

    def transform_action(self, name: str, action: int) -> int:
        """Map an in-bounds action id through group element ``name``.

        Decode → transform cells → re-encode: the resulting anchor is the bbox
        top-left of the transformed cells, never a transported anchor.

        Args:
            name: Group element name from :attr:`names`.
            action: In-bounds flat action id.

        Returns:
            The image action id (always in-bounds: the group stabilizes the board).
        """
        m = self._cell_maps[name]
        return self.codec.encode_cells([m(r, c) for r, c in self.codec.action_cells(action)])

    def action_maps(self) -> dict[str, dict[int, int]]:
        """Build (once) the (g,a)→a′ maps over all in-bounds ids, per element.

        Returns:
            Per group element, a dict from in-bounds action id to its image.
        """
        if self._action_maps is None:
            self._action_maps = {
                g: {a: self.transform_action(g, a) for a in self.codec.in_bounds_actions}
                for g in self.names
            }
        return self._action_maps

    def full_permutation(self, name: str) -> tuple[int, ...]:
        """Return the full ``H*W*K``-length policy-head permutation for ``name``.

        Off-support (out-of-bounds) ids map to themselves — identity filler for
        slots that are never legal, so the permutation is total over the head
        (17,836 entries for the full game, 225 for the micro instance).

        Args:
            name: Group element name from :attr:`names`.

        Returns:
            Tuple ``perm`` with ``perm[a]`` the image of action ``a``.
        """
        perm = self._permutations.get(name)
        if perm is None:
            values = list(range(self.codec.num_actions))
            for a, image in self.action_maps()[name].items():
                values[a] = image
            perm = tuple(values)
            self._permutations[name] = perm
        return perm

    def plane_transform(self, name: str) -> Callable:
        """Return the plane-tensor transform for group element ``name``.

        Operates on ``encode_state`` output — the instance's nested-tuple ``H×W``
        planes (§5.2: 46 for the full game, 12 for micro) — moving the value at
        ``(r, c)`` to the image cell in every plane. Constant inventory/flag
        broadcast planes are invariant under the map but go through the same code
        path (no special-casing); mover perspective is untouched (board symmetry,
        no player relabeling).

        Args:
            name: Group element name from :attr:`names`.

        Returns:
            Callable mapping a plane tuple to its transformed plane tuple.
        """
        m = self._cell_maps[name]
        size = self.config.board_size

        def transform(planes):
            out = []
            for plane in planes:
                grid = [[0] * size for _ in range(size)]
                for r in range(size):
                    for c in range(size):
                        tr, tc = m(r, c)
                        grid[tr][tc] = plane[r][c]
                out.append(tuple(tuple(row) for row in grid))
            return tuple(out)

        return transform

    def state_transform(self, name: str) -> Callable:
        """Return an engine-state-level transform for group element ``name`` [F6].

        Module utility (used by the M1/M2 equivariance tests), deliberately *not*
        exposed through the adapter's ``symmetry_group``. Works on the shared
        engine state tuple with occupancies as either frozensets of cells
        (oracle) or ``H*W``-bit ints (bitboard); inventories, flags, and
        ``to_play`` are invariant under board symmetry (no player relabeling).

        Args:
            name: Group element name from :attr:`names`.

        Returns:
            Callable mapping a state tuple to its transformed state tuple.
        """
        m = self._cell_maps[name]
        size = self.config.board_size

        def transform_occ(occ):
            if isinstance(occ, int):
                bb = 0
                for i in range(size * size):
                    if occ >> i & 1:
                        r, c = divmod(i, size)
                        tr, tc = m(r, c)
                        bb |= 1 << (tr * size + tc)
                return bb
            return frozenset(m(r, c) for r, c in occ)

        def transform(state):
            return (transform_occ(state[0]), transform_occ(state[1]), *state[2:])

        return transform


@cache
def symmetry_group(config: BlokusConfig = FULL_CONFIG) -> SymmetryGroup:
    """Return ``config``'s symmetry group, built once per config.

    Args:
        config: The instance config; defaults to the full 14×14 game.

    Returns:
        The shared :class:`SymmetryGroup` for that config.
    """
    return SymmetryGroup(config)


# --- full-game bindings (§8: Klein-4 over 14×14×91) -------------------------------
# The M1/M2 module surface, unchanged: names below are the FULL_CONFIG group's.

FULL_GROUP = symmetry_group(FULL_CONFIG)

# Element order is the adapter's group order; identity is element 0 (D9 counts
# it among the 4 augmentation symmetries).
GROUP_NAMES: tuple[str, ...] = FULL_GROUP.names

cell_map = FULL_GROUP.cell_map
transform_action = FULL_GROUP.transform_action
build_action_maps = FULL_GROUP.action_maps
full_permutation = FULL_GROUP.full_permutation
plane_transform = FULL_GROUP.plane_transform
state_transform = FULL_GROUP.state_transform
