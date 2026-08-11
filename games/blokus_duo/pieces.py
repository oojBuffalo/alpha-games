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

Config-parameterized since M2.5 (§5.3): :func:`build_pieces`,
:func:`build_orientation_table` and :func:`orientation_table_hash` take a
:class:`~games.blokus_duo.config.BlokusConfig` and derive the subset's tables
from scratch — pieces re-sorted by ``(size, canonical form)`` and renumbered
from 0, orientation ids re-assigned by the same sorted-canonical-cell rule
(invariant 4), *never* the full table restricted. The module-level constants
below stay bound to the full 14×14 game and are what every M1 caller imports.
This module imports nothing from the package (the config type only under
``TYPE_CHECKING``), so ``config`` can import it without a cycle.
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - annotation-only import (no runtime cycle)
    from games.blokus_duo.config import BlokusConfig

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
        (r, c) for r, row in enumerate(art.split("\n")) for c, ch in enumerate(row) if ch == "X"
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


def _build_pieces(
    names: tuple[str, ...] | None = None,
) -> tuple[tuple[Cells, ...], tuple[str, ...]]:
    art = _PIECE_ART if names is None else {name: _PIECE_ART[name] for name in names}
    entries = sorted(
        ((canonical_form(_cells_from_art(a)), name) for name, a in art.items()),
        key=lambda e: (len(e[0]), e[0]),
    )
    return tuple(p for p, _ in entries), tuple(n for _, n in entries)


# Ordered by (size, canonical form) — §5.1; PIECE_NAMES is parallel.
BASE_PIECES, PIECE_NAMES = _build_pieces()


@cache
def build_pieces(config: BlokusConfig) -> tuple[tuple[Cells, ...], tuple[str, ...]]:
    """Build ``config``'s piece set: the canonical pieces it names, renumbered.

    The subset is re-sorted by ``(size, canonical form)`` and indexed from 0, so
    piece ids (and therefore the D3 inventory-plane order) are the subset's own —
    not positions inherited from the full 21. ``config.piece_names = None`` is
    the full game and reproduces :data:`BASE_PIECES` / :data:`PIECE_NAMES`.

    Args:
        config: The instance config (hashable; results are cached per config).

    Returns:
        ``(pieces, names)``: parallel tuples of canonical cell tuples and their
        piece names, in the subset's ``(size, canonical form)`` order.
    """
    return _build_pieces(config.piece_names)


@cache
def build_orientation_table(config: BlokusConfig | None = None) -> tuple[tuple[Cells, ...], ...]:
    """Build the per-piece orientation table (§5.1 convention pins).

    Deterministic by construction: orientations come from :func:`d4_orientations`
    (deduped, lexicographically sorted) over the config's pieces — never
    set-iteration order. Because the pieces are the subset's own, so are the
    resulting global orientation ids (invariant 4: re-derived, not restricted).

    Args:
        config: The instance config, or ``None`` for the full 14×14 game
            (equivalently ``FULL_CONFIG``, whose table is :data:`ORIENTATIONS`).

    Returns:
        Per piece (in piece order), the tuple of its distinct fixed
        orientations, each an origin-normalized sorted cell tuple.
    """
    pieces = BASE_PIECES if config is None else build_pieces(config)[0]
    return tuple(tuple(d4_orientations(p)) for p in pieces)


ORIENTATIONS = build_orientation_table()
# Global orientation ids 0-90: piece-major traversal order (§5.1).
ORIENTATION_CELLS: tuple[Cells, ...] = tuple(o for orients in ORIENTATIONS for o in orients)
ORIENTATION_PIECE: tuple[int, ...] = tuple(
    i for i, orients in enumerate(ORIENTATIONS) for _ in orients
)


def orientation_table_hash(config: BlokusConfig | None = None) -> str:
    """Return the sha256 digest of ``config``'s orientation table (§5.1).

    The serialization is canonical JSON nested per piece — the per-piece
    boundaries are part of the hashed structure, so a piece↔orientation
    regrouping cannot collide with the same flat orientation list. This digest
    is serialized into every fixture, checkpoint, and replay dataset
    (write-side M1, validate-on-load M3), which is what version-binds an
    instance's data to its own table: the micro instance's re-derived ids give
    it a different digest from the full game's, by construction.

    Args:
        config: The instance config, or ``None`` for the full 14×14 game.

    Returns:
        Hex sha256 digest of the canonical serialization.
    """
    table = build_orientation_table(config)
    payload = [[[list(cell) for cell in orient] for orient in orients] for orients in table]
    blob = json.dumps(payload, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(blob).hexdigest()
