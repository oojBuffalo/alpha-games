"""Action encoding for the ``H×W×K`` policy head (design doc §5.1, D2).

Convention pins (§5.1): ``action_id = (r*W + c)*K + o`` — cell-major flatten,
0-indexed ``(r, c)``, anchor = board cell where the origin-normalized
orientation's bounding-box top-left lands. An anchor is in-bounds iff
``r + h <= H and c + w <= W`` for the orientation's ``h×w`` bbox. The full game
instantiates this at ``(r*14 + c)*91 + o`` over 17,836 raw actions; the §5.3
micro instance at ``(r*5 + c)*9 + o`` over 225.

An :class:`ActionCodec` carries one instance's dimensions and tables; the
module-level names below are the full game's codec, so every M1 caller keeps
importing exactly what it did before. Both engines (oracle, bitboard) share only
this codec plus the piece data; the encoding is pinned by literal hand-derived
goldens in ``tests/test_blokus_actions.py`` [F3].
"""

from __future__ import annotations

from functools import cache

from games.blokus_duo.config import FULL_CONFIG, BlokusConfig
from games.blokus_duo.pieces import Cells, build_orientation_table


def _bbox(cells: Cells) -> tuple[int, int]:
    h = 1 + max(r for r, _ in cells)
    w = 1 + max(c for _, c in cells)
    return h, w


class ActionCodec:
    """One instance's action space: flatten, anchors, in-bounds set, openings.

    Args:
        config: The instance config. Its orientation table (piece ids and
            orientation ids re-derived within the piece subset, invariant 4)
            fixes every id this codec produces.

    Attributes:
        config: The config this codec was built from.
        board_size: Board edge length ``H = W``.
        start_squares: The instance's 0-indexed start squares.
        num_orientations: ``K``, the number of global orientation ids.
        num_actions: Raw head size ``H*W*K`` (in-bounds and out-of-bounds).
        orientation_cells: Per orientation id, its origin-normalized cells.
        orientation_piece: Per orientation id, its piece index.
        orientation_bbox: Per orientation id, its ``(h, w)`` bounding box.
        in_bounds_actions: Sorted tuple of every in-bounds action id.
        opening_actions: Per start square, the in-bounds ids covering it.
        fixture_conventions: The encoding conventions embedded in generated
            fixtures alongside the orientation hash (the hash covers the
            orientation table, not the flatten/anchor conventions).
    """

    def __init__(self, config: BlokusConfig):
        table = build_orientation_table(config)
        self.config = config
        self.board_size = config.board_size
        self.start_squares = config.start_squares
        self.orientation_cells: tuple[Cells, ...] = tuple(o for orients in table for o in orients)
        self.orientation_piece: tuple[int, ...] = tuple(
            i for i, orients in enumerate(table) for _ in orients
        )
        self.num_orientations = len(self.orientation_cells)
        self.num_actions = self.board_size * self.board_size * self.num_orientations
        self.orientation_bbox: tuple[tuple[int, int], ...] = tuple(
            _bbox(o) for o in self.orientation_cells
        )
        # Orientation lookup for encode_cells: origin-normalized cells -> id.
        self._orientation_id: dict[Cells, int] = {
            cells: o for o, cells in enumerate(self.orientation_cells)
        }
        self.in_bounds_actions: tuple[int, ...] = self._enumerate_in_bounds()
        # Opening actions per start square (414 each at 14×14, 21 each at 5×5;
        # §4): the in-bounds placements covering that square. The two sets are
        # disjoint in both pinned instances (no piece bbox spans both squares).
        self.opening_actions: dict[tuple[int, int], tuple[int, ...]] = {
            sq: tuple(a for a in self.in_bounds_actions if sq in self.action_cells(a))
            for sq in self.start_squares
        }
        self.fixture_conventions = {
            "axis_order": "(r,c) 0-indexed row-major",
            "flatten": f"(r*{self.board_size}+c)*{self.num_orientations}+o",
            "anchor": "bbox-top-left",
            "board_size": self.board_size,
            "start_squares": [list(sq) for sq in self.start_squares],
        }

    def encode(self, r: int, c: int, o: int) -> int:
        """Encode an anchor cell and orientation id as a flat action id.

        Args:
            r: Anchor row (0-indexed).
            c: Anchor column (0-indexed).
            o: Global orientation id in ``[0, num_orientations)``.

        Returns:
            The flat action id ``(r*W + c)*K + o``.
        """
        return (r * self.board_size + c) * self.num_orientations + o

    def decode(self, action: int) -> tuple[int, int, int]:
        """Decode a flat action id into ``(r, c, o)``.

        Args:
            action: Flat action id in ``[0, num_actions)``.

        Returns:
            Tuple ``(anchor_row, anchor_col, orientation_id)``.
        """
        cell, o = divmod(action, self.num_orientations)
        r, c = divmod(cell, self.board_size)
        return r, c, o

    def action_cells(self, action: int) -> Cells:
        """Return the absolute board cells covered by ``action``, sorted.

        Args:
            action: Flat action id.

        Returns:
            The orientation's cells translated by the anchor, as a sorted tuple.
        """
        r, c, o = self.decode(action)
        return tuple(sorted((r + dr, c + dc) for dr, dc in self.orientation_cells[o]))

    def action_piece(self, action: int) -> int:
        """Return the piece index ``action`` places.

        Codec surface (not a table both engines must share for shapes): it maps
        an already-encoded id back to the piece it spends, which both the oracle
        and the bitboard engine need to update inventories.

        Args:
            action: Flat action id.

        Returns:
            The piece index within this config's piece set.
        """
        return self.orientation_piece[self.decode(action)[2]]

    def encode_cells(self, cells) -> int:
        """Encode absolute placement cells as a flat action id (``encode_action``).

        The anchor is the bounding-box top-left of the absolute cells (D2), which
        coincides with translating the origin-normalized orientation.

        Args:
            cells: Iterable of absolute ``(row, col)`` tuples of one placement.

        Returns:
            The flat action id.

        Raises:
            KeyError: If the cells are not a translate of any fixed orientation.
        """
        pts = sorted(cells)
        mr = min(r for r, _ in pts)
        mc = min(c for _, c in pts)
        norm = tuple((r - mr, c - mc) for r, c in pts)
        return self.encode(mr, mc, self._orientation_id[norm])

    def _enumerate_in_bounds(self) -> tuple[int, ...]:
        out = []
        for o, (h, w) in enumerate(self.orientation_bbox):
            for r in range(self.board_size - h + 1):
                for c in range(self.board_size - w + 1):
                    out.append(self.encode(r, c, o))
        return tuple(sorted(out))


@cache
def action_codec(config: BlokusConfig = FULL_CONFIG) -> ActionCodec:
    """Return ``config``'s action codec, built once per config.

    Args:
        config: The instance config; defaults to the full 14×14 game.

    Returns:
        The shared :class:`ActionCodec` for that config.
    """
    return ActionCodec(config)


# --- full-game bindings (§5.1: 14×14×91) -----------------------------------------
# The M1 module surface, unchanged: names below are the FULL_CONFIG codec's.

FULL_CODEC = action_codec(FULL_CONFIG)

BOARD_SIZE = FULL_CODEC.board_size  # 14
NUM_ORIENTATIONS = FULL_CODEC.num_orientations  # 91
NUM_ACTIONS = FULL_CODEC.num_actions  # 17,836

# Start squares, 0-indexed (§5.1; the doc's (5,5)/(10,10) is 1-indexed display).
START_SQUARES: tuple[tuple[int, int], ...] = FULL_CODEC.start_squares

# Embedded in every generated fixture alongside the orientation hash [F3].
FIXTURE_CONVENTIONS = FULL_CODEC.fixture_conventions

# Per-orientation bbox, indexed by orientation id.
ORIENTATION_BBOX: tuple[tuple[int, int], ...] = FULL_CODEC.orientation_bbox

# All in-bounds placements (13,729), sorted by action id.
IN_BOUNDS_ACTIONS: tuple[int, ...] = FULL_CODEC.in_bounds_actions

# Opening actions per start square (414 each; design doc §4).
OPENING_ACTIONS: dict[tuple[int, int], tuple[int, ...]] = FULL_CODEC.opening_actions

encode = FULL_CODEC.encode
decode = FULL_CODEC.decode
action_cells = FULL_CODEC.action_cells
encode_cells = FULL_CODEC.encode_cells
