#!/usr/bin/env python3
"""Independently enumerate every derived constant of the micro-Blokus config (§5.5).

Deliberately standalone: this script shares **no code** with ``games/blokus_duo/``
— it grows the polyomino set from scratch, runs its own D4 transforms, its own
placement enumeration and its own scoring bounds. It exists to justify the
numbers written into the design doc's M2.5 amendment *before* any of them is
hardcoded, per the working principle "verify load-bearing claims independently
before hardcoding numbers".

Run it twice: the output is deterministic and must be byte-identical.

Usage::

    python3 scripts/enumerate_micro_config.py            # human-readable report
    python3 scripts/enumerate_micro_config.py --json     # machine-readable

The constants it prints are the ones ``tests/test_micro_config.py`` asserts
against the parameterized engine, closing the loop between doc, script and code.
"""

from __future__ import annotations

import argparse
import hashlib
import json

# --- the pinned micro instance (design doc §5.5) --------------------------------

BOARD_SIZE = 5
MAX_PIECE_ORDER = 3
START_SQUARES = ((1, 1), (3, 3))  # 0-indexed (§5.1); (2,2)/(4,4) in 1-indexed display

# Scoring bonuses carried over unchanged from the full game (§4).
ALL_PLACED_BONUS = 15
MONOMINO_LAST_BONUS = 5

Cells = tuple[tuple[int, int], ...]


# --- independent polyomino enumeration ------------------------------------------


def _normalize(cells) -> Cells:
    """Translate ``cells`` so the minimum row and column are 0, then sort.

    Args:
        cells: Iterable of ``(row, col)`` pairs.

    Returns:
        The origin-normalized, lexicographically sorted cell tuple.
    """
    pts = list(cells)
    mr = min(r for r, _ in pts)
    mc = min(c for _, c in pts)
    return tuple(sorted((r - mr, c - mc) for r, c in pts))


def _d4_images(cells) -> list[Cells]:
    """Return the 8 D4 images of ``cells``, each origin-normalized.

    Uses a reflect-then-rotate formulation (independent of ``pieces.py``'s
    rotate-then-reflect loop) so agreement between the two is evidence, not a
    shared bug.

    Args:
        cells: Iterable of ``(row, col)`` pairs.

    Returns:
        Eight normalized cell tuples (with duplicates for symmetric shapes).
    """
    out = []
    for mirror in (False, True):
        pts = [(r, -c) for r, c in cells] if mirror else list(cells)
        for _ in range(4):
            pts = [(-c, r) for r, c in pts]  # rotate 90°
            out.append(_normalize(pts))
    return out


def _canonical(cells) -> Cells:
    """Return the lexicographically least D4 image — the free piece's identity."""
    return min(_d4_images(cells))


def _fixed_orientations(cells) -> list[Cells]:
    """Return the piece's distinct fixed orientations, lexicographically sorted."""
    return sorted(set(_d4_images(cells)))


def grow_free_polyominoes(max_order: int) -> list[Cells]:
    """Enumerate free polyominoes of orders 1..``max_order`` by cell growth.

    Growth enumeration rather than hand-drawn art: start from the monomino and
    repeatedly add an orthogonally adjacent cell, deduping by canonical form.

    Args:
        max_order: Largest polyomino order to enumerate (inclusive).

    Returns:
        Canonical cell tuples ordered by ``(size, canonical form)`` — the §5.1
        piece order, re-derived within the subset.
    """
    frontier: set[Cells] = {((0, 0),)}
    found: list[Cells] = sorted(frontier)
    for _ in range(max_order - 1):
        grown: set[Cells] = set()
        for shape in frontier:
            occupied = set(shape)
            for r, c in shape:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cell = (r + dr, c + dc)
                    if cell not in occupied:
                        grown.add(_canonical([*shape, cell]))
        frontier = grown
        found.extend(sorted(grown))
    return sorted(found, key=lambda p: (len(p), p))


# --- derived constants -----------------------------------------------------------


def orientation_table(pieces: list[Cells]) -> tuple[tuple[Cells, ...], ...]:
    """Build the per-piece fixed-orientation table for ``pieces`` (invariant 4)."""
    return tuple(tuple(_fixed_orientations(p)) for p in pieces)


def orientation_table_hash(table: tuple[tuple[Cells, ...], ...]) -> str:
    """Hash the orientation table with per-piece boundaries preserved (§5.1).

    Args:
        table: Per-piece tuples of fixed orientations.

    Returns:
        Hex sha256 of the canonical JSON serialization.
    """
    payload = [[[list(cell) for cell in orient] for orient in orients] for orients in table]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("ascii")).hexdigest()


def _bbox(cells: Cells) -> tuple[int, int]:
    return 1 + max(r for r, _ in cells), 1 + max(c for _, c in cells)


def in_bounds_placements(orientations: list[Cells], board: int) -> list[tuple[int, int, int]]:
    """Enumerate every in-bounds ``(anchor_row, anchor_col, orientation_id)``.

    Args:
        orientations: Flat orientation list in global-id order.
        board: Board edge length.

    Returns:
        Sorted list of in-bounds placements as ``(r, c, o)`` triples.
    """
    out = []
    for o, cells in enumerate(orientations):
        h, w = _bbox(cells)
        for r in range(board - h + 1):
            for c in range(board - w + 1):
                out.append((r, c, o))
    return sorted(out)


def placement_cells(placement: tuple[int, int, int], orientations: list[Cells]) -> frozenset:
    """Return the absolute board cells covered by an ``(r, c, o)`` placement."""
    r, c, o = placement
    return frozenset((r + dr, c + dc) for dr, dc in orientations[o])


def d4_stabilizer(squares, board: int) -> list[str]:
    """Compute the D4 set-stabilizer of ``squares`` — the micro symmetry group (§8).

    Args:
        squares: The start squares as ``(row, col)`` pairs.
        board: Board edge length.

    Returns:
        Names of the D4 elements setwise-fixing ``squares``, in the canonical
        element order.
    """
    last = board - 1
    maps = {
        "identity": lambda r, c: (r, c),
        "rot90": lambda r, c: (c, last - r),
        "rot180": lambda r, c: (last - r, last - c),
        "rot270": lambda r, c: (last - c, r),
        "flip_h": lambda r, c: (r, last - c),
        "flip_v": lambda r, c: (last - r, c),
        "diag": lambda r, c: (c, r),
        "antidiag": lambda r, c: (last - c, last - r),
    }
    target = set(squares)
    return [name for name, m in maps.items() if {m(r, c) for r, c in target} == target]


def enumerate_config() -> dict:
    """Enumerate every derived micro constant.

    Returns:
        A JSON-serializable dict of the pinned inputs and all derived constants.
    """
    pieces = grow_free_polyominoes(MAX_PIECE_ORDER)
    table = orientation_table(pieces)
    flat = [o for orients in table for o in orients]
    placements = in_bounds_placements(flat, BOARD_SIZE)
    opening = {
        f"{sq[0]},{sq[1]}": sum(1 for p in placements if sq in placement_cells(p, flat))
        for sq in START_SQUARES
    }
    squares_per_set = sum(len(p) for p in pieces)
    max_score = ALL_PLACED_BONUS + MONOMINO_LAST_BONUS
    min_score = -squares_per_set
    has_monomino = any(len(p) == 1 for p in pieces)
    return {
        "board_size": BOARD_SIZE,
        "max_piece_order": MAX_PIECE_ORDER,
        "start_squares_0indexed": [list(sq) for sq in START_SQUARES],
        "start_squares_1indexed_display": [[r + 1, c + 1] for r, c in START_SQUARES],
        "num_pieces": len(pieces),
        "pieces_canonical": [[list(cell) for cell in p] for p in pieces],
        "orientations_per_piece": [len(o) for o in table],
        "num_orientations": len(flat),
        "policy_shape": [BOARD_SIZE, BOARD_SIZE, len(flat)],
        "num_raw_actions": BOARD_SIZE * BOARD_SIZE * len(flat),
        "num_in_bounds_placements": len(placements),
        "opening_actions_per_start_square": opening,
        "num_legal_openings": sum(opening.values()),
        "has_monomino": has_monomino,
        "input_planes": 2 + 2 * len(pieces) + (2 if has_monomino else 0),
        "squares_per_set": squares_per_set,
        "score_min": min_score,
        "score_max": max_score,
        "max_abs_score_diff": max_score - min_score,
        "symmetry_group": d4_stabilizer(START_SQUARES, BOARD_SIZE),
        "orientation_table_hash": orientation_table_hash(table),
    }


def _report(data: dict) -> str:
    """Render the enumerated constants as the human-readable report."""
    lines = [
        "micro-Blokus derived constants (independently enumerated)",
        "=" * 58,
        f"board                      {data['board_size']}x{data['board_size']}",
        f"piece subset               all free polyominoes of order <= {data['max_piece_order']}"
        f" ({data['num_pieces']} pieces)",
        f"start squares (0-indexed)  {data['start_squares_0indexed']}",
        f"start squares (1-indexed)  {data['start_squares_1indexed_display']}  [display only]",
        "-" * 58,
        f"orientations per piece     {data['orientations_per_piece']}",
        f"num_orientations           {data['num_orientations']}",
        f"policy_shape               {tuple(data['policy_shape'])}",
        f"raw actions                {data['num_raw_actions']}",
        f"in-bounds placements       {data['num_in_bounds_placements']}",
        f"openings per start square  {data['opening_actions_per_start_square']}",
        f"legal openings (total)     {data['num_legal_openings']}",
        f"input planes               {data['input_planes']}"
        f"  (2 occ + 2x{data['num_pieces']} inv + 2 mono flags)",
        f"squares per set            {data['squares_per_set']}",
        f"score range                [{data['score_min']}, {data['score_max']}]",
        f"max |score_diff| (aux div) {data['max_abs_score_diff']}",
        f"symmetry group             {data['symmetry_group']}",
        f"orientation-table hash     {data['orientation_table_hash']}",
    ]
    return "\n".join(lines)


def main() -> None:
    """Print the enumerated constants (human-readable by default, JSON on request)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    data = enumerate_config()
    print(json.dumps(data, indent=2, sort_keys=True) if args.json else _report(data))


if __name__ == "__main__":
    main()
