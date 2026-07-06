"""Action encoding for the 14×14×91 policy head (design doc §5.1, D2).

Convention pins (§5.1): ``action_id = (r*14 + c)*91 + o`` — cell-major flatten,
0-indexed ``(r, c)``, anchor = board cell where the origin-normalized
orientation's bounding-box top-left lands. An anchor is in-bounds iff
``r + h <= 14 and c + w <= 14`` for the orientation's ``h×w`` bbox.

Both engines (oracle, bitboard) share only this encoding plus the piece data;
the encoding is pinned by literal hand-derived goldens in
``tests/test_blokus_actions.py`` [F3].
"""

from __future__ import annotations

from games.blokus_duo.pieces import ORIENTATION_CELLS, Cells

BOARD_SIZE = 14
NUM_ORIENTATIONS = len(ORIENTATION_CELLS)  # 91
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE * NUM_ORIENTATIONS  # 17,836

# Start squares, 0-indexed (§5.1; the doc's (5,5)/(10,10) is 1-indexed display).
START_SQUARES: tuple[tuple[int, int], ...] = ((4, 4), (9, 9))


def encode(r: int, c: int, o: int) -> int:
    """Encode an anchor cell and orientation id as a flat action id.

    Args:
        r: Anchor row (0-indexed).
        c: Anchor column (0-indexed).
        o: Global orientation id (0–90).

    Returns:
        The flat action id ``(r*14 + c)*91 + o``.
    """
    return (r * BOARD_SIZE + c) * NUM_ORIENTATIONS + o


def decode(action: int) -> tuple[int, int, int]:
    """Decode a flat action id into ``(r, c, o)``.

    Args:
        action: Flat action id in ``[0, NUM_ACTIONS)``.

    Returns:
        Tuple ``(anchor_row, anchor_col, orientation_id)``.
    """
    cell, o = divmod(action, NUM_ORIENTATIONS)
    r, c = divmod(cell, BOARD_SIZE)
    return r, c, o


def _bbox(cells: Cells) -> tuple[int, int]:
    h = 1 + max(r for r, _ in cells)
    w = 1 + max(c for _, c in cells)
    return h, w


# Per-orientation bbox, indexed by orientation id.
ORIENTATION_BBOX: tuple[tuple[int, int], ...] = tuple(_bbox(o) for o in ORIENTATION_CELLS)


def action_cells(action: int) -> Cells:
    """Return the absolute board cells covered by ``action``, sorted.

    Args:
        action: Flat action id.

    Returns:
        The orientation's cells translated by the anchor, as a sorted tuple.
    """
    r, c, o = decode(action)
    return tuple(sorted((r + dr, c + dc) for dr, dc in ORIENTATION_CELLS[o]))


# Orientation lookup for encode_cells: origin-normalized sorted cells -> global id.
_ORIENTATION_ID: dict[Cells, int] = {cells: o for o, cells in enumerate(ORIENTATION_CELLS)}


def encode_cells(cells) -> int:
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
    return encode(mr, mc, _ORIENTATION_ID[norm])


def _enumerate_in_bounds() -> tuple[int, ...]:
    out = []
    for o, (h, w) in enumerate(ORIENTATION_BBOX):
        for r in range(BOARD_SIZE - h + 1):
            for c in range(BOARD_SIZE - w + 1):
                out.append(encode(r, c, o))
    return tuple(sorted(out))


# All in-bounds placements (13,729), sorted by action id.
IN_BOUNDS_ACTIONS: tuple[int, ...] = _enumerate_in_bounds()

# Opening actions per start square (414 each; design doc §4): the in-bounds
# placements covering that square. The two sets are disjoint (no bbox spans both).
OPENING_ACTIONS: dict[tuple[int, int], tuple[int, ...]] = {
    sq: tuple(a for a in IN_BOUNDS_ACTIONS if sq in action_cells(a)) for sq in START_SQUARES
}
