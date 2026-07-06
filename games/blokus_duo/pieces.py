"""The 21 Blokus polyominoes (orders 1–5) as canonical cell data.

This module is the single piece-data surface shared by the oracle and the
bitboard engines (design doc §12 M1) — everything downstream of it is
independently implemented and differential-tested, so the data itself is
guarded by the F1 growth-enumeration cross-check in ``tests/test_blokus_pieces.py``.

Conventions (design doc §5.1 "Convention pins (M1)"): cells are 0-indexed
``(row, col)`` tuples; a piece's canonical form is the lexicographically least
of its 8 origin-normalized D4 images, stored as ``tuple(sorted(cells))``;
``BASE_PIECES`` is ordered by ``(size, canonical form)``, which also fixes the
M2 inventory-plane order (D3).
"""

from __future__ import annotations

import hashlib
import json

Cells = tuple[tuple[int, int], ...]

# Hand drawings ('X' = cell), named per the common Blokus convention. The
# drawings are inputs only: each is canonicalized before being stored.
_PIECE_ART: dict[str, str] = {
    "I1": "X",
    "I2": "XX",
    "I3": "XXX",
    "V3": "XX\nX.",
    "I4": "XXXX",
    "L4": "XXX\nX..",
    "O4": "XX\nXX",
    "S4": "XX.\n.XX",
    "T4": "XXX\n.X.",
    "F5": ".XX\nXX.\n.X.",
    "I5": "XXXXX",
    "L5": "X.\nX.\nX.\nXX",
    "N5": ".X\n.X\nXX\nX.",
    "P5": "XX\nXX\nX.",
    "T5": "XXX\n.X.\n.X.",
    "U5": "X.X\nXXX",
    "V5": "X..\nX..\nXXX",
    "W5": "X..\nXX.\n.XX",
    "X5": ".X.\nXXX\n.X.",
    "Y5": ".X\nXX\n.X\n.X",
    "Z5": "XX.\n.X.\n.XX",
}


def _cells_from_art(art: str) -> frozenset[tuple[int, int]]:
    """Parse an 'X'/'.' drawing into a set of ``(row, col)`` cells.

    Args:
        art: Newline-separated rows of ``X`` (cell) and ``.`` (empty).

    Returns:
        The occupied cells as a frozenset of ``(row, col)`` tuples.
    """
    return frozenset(
        (r, c)
        for r, row in enumerate(art.split("\n"))
        for c, ch in enumerate(row)
        if ch == "X"
    )


def normalize(cells) -> Cells:
    """Translate ``cells`` so min row and min col are 0; return ``tuple(sorted(...))``.

    Args:
        cells: An iterable of ``(row, col)`` tuples.

    Returns:
        The origin-normalized, lexicographically sorted cell tuple — the §5.1
        canonical representation of one fixed orientation.
    """
    pts = list(cells)
    mr = min(r for r, _ in pts)
    mc = min(c for _, c in pts)
    return tuple(sorted((r - mr, c - mc) for r, c in pts))


def d4_orientations(cells) -> list[Cells]:
    """Return the piece's distinct fixed orientations, lexicographically sorted.

    Generates all 8 D4 images (4 rotations × optional reflection), normalizes
    each per :func:`normalize`, dedupes, and sorts — the §5.1 orientation-ID
    assignment order.

    Args:
        cells: An iterable of ``(row, col)`` tuples.

    Returns:
        Sorted list of distinct origin-normalized orientation cell tuples.
    """
    pts = list(cells)
    images = set()
    for _ in range(4):
        pts = [(c, -r) for r, c in pts]  # rotate 90°
        images.add(normalize(pts))
        images.add(normalize((r, -c) for r, c in pts))  # + reflection
    return sorted(images)


def canonical_form(cells) -> Cells:
    """Return the lexicographically least D4 orientation — the piece's identity.

    Args:
        cells: An iterable of ``(row, col)`` tuples.

    Returns:
        The canonical (lex-least origin-normalized) cell tuple.
    """
    return d4_orientations(cells)[0]


def _build_pieces() -> tuple[tuple[Cells, ...], tuple[str, ...]]:
    entries = sorted(
        ((canonical_form(_cells_from_art(art)), name) for name, art in _PIECE_ART.items()),
        key=lambda e: (len(e[0]), e[0]),
    )
    return tuple(p for p, _ in entries), tuple(n for _, n in entries)


# Ordered by (size, canonical form) — §5.1; PIECE_NAMES is parallel.
BASE_PIECES, PIECE_NAMES = _build_pieces()


def build_orientation_table() -> tuple[tuple[Cells, ...], ...]:
    """Build the per-piece orientation table (§5.1 convention pins).

    Deterministic by construction: orientations come from :func:`d4_orientations`
    (deduped, lexicographically sorted) over :data:`BASE_PIECES` — never
    set-iteration order.

    Returns:
        Per piece (in ``BASE_PIECES`` order), the tuple of its distinct fixed
        orientations, each an origin-normalized sorted cell tuple.
    """
    return tuple(tuple(d4_orientations(p)) for p in BASE_PIECES)


ORIENTATIONS = build_orientation_table()
# Global orientation ids 0-90: piece-major traversal order (§5.1).
ORIENTATION_CELLS: tuple[Cells, ...] = tuple(o for orients in ORIENTATIONS for o in orients)
ORIENTATION_PIECE: tuple[int, ...] = tuple(
    i for i, orients in enumerate(ORIENTATIONS) for _ in orients
)


def orientation_table_hash() -> str:
    """Return the sha256 digest of the orientation table (§5.1).

    The serialization is canonical JSON nested per piece — the per-piece
    boundaries are part of the hashed structure, so a piece↔orientation
    regrouping cannot collide with the same flat orientation list. This digest
    is serialized into every fixture, checkpoint, and replay dataset
    (write-side M1, validate-on-load M3).

    Returns:
        Hex sha256 digest of the canonical serialization.
    """
    payload = [
        [[list(cell) for cell in orient] for orient in orients] for orients in ORIENTATIONS
    ]
    blob = json.dumps(payload, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(blob).hexdigest()
