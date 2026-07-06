"""Blokus piece-set goldens + the F1 growth-enumeration cross-check.

The base-piece data is the single surface shared by the oracle and bitboard
engines, so differential testing cannot protect it. The growth enumeration here
is an independent reconstruction of the free polyominoes of orders 1–5,
including its own D4 canonicalization — it must match the hand-defined set
exactly (a duplicated same-bbox pentomino would preserve every aggregate golden
and self-consistently corrupt the perft fixtures).
"""

from __future__ import annotations

from collections import Counter

from games.blokus_duo.pieces import BASE_PIECES, PIECE_NAMES

# --- independent D4 canonicalization (deliberately not imported from pieces) ---


def _d4_images(cells):
    """Return the set of origin-normalized sorted cell tuples over all 8 D4 images."""
    pts = list(cells)
    images = []
    for _ in range(4):
        pts = [(c, -r) for r, c in pts]  # rotate 90°
        images.append(pts)
        images.append([(r, -c) for r, c in pts])  # + reflection
    out = set()
    for img in images:
        mr = min(r for r, _ in img)
        mc = min(c for _, c in img)
        out.add(tuple(sorted((r - mr, c - mc) for r, c in img)))
    return out


def _canon(cells):
    return min(_d4_images(cells))


def _free_polyominoes(max_order):
    """Growth-enumerate all free polyominoes up to ``max_order``, canonicalized."""
    by_order = {1: {((0, 0),)}}
    for n in range(2, max_order + 1):
        grown = set()
        for poly in by_order[n - 1]:
            cells = set(poly)
            for r, c in poly:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (r + dr, c + dc)
                    if nb not in cells:
                        grown.add(_canon(cells | {nb}))
        by_order[n] = grown
    return by_order


# --- goldens ------------------------------------------------------------------


def test_twenty_one_pieces():
    assert len(BASE_PIECES) == 21
    assert len(PIECE_NAMES) == 21


def test_order_distribution():
    assert Counter(len(p) for p in BASE_PIECES) == {1: 1, 2: 1, 3: 2, 4: 5, 5: 12}


def test_eighty_nine_squares():
    assert sum(len(p) for p in BASE_PIECES) == 89


def test_pieces_are_canonical_and_ordered():
    # Each stored piece is its own lex-least D4 image (checked against the
    # independent canonicalizer), and the tuple is sorted by (size, canonical rep).
    for piece in BASE_PIECES:
        assert piece == _canon(piece)
    assert list(BASE_PIECES) == sorted(BASE_PIECES, key=lambda p: (len(p), p))


def test_growth_enumeration_matches_hand_defined_set():
    # [F1] Independent reconstruction of the full free-polyomino sets.
    by_order = _free_polyominoes(5)
    assert {n: len(s) for n, s in by_order.items()} == {1: 1, 2: 1, 3: 2, 4: 5, 5: 12}
    enumerated = {p for polys in by_order.values() for p in polys}
    hand_defined = {_canon(p) for p in BASE_PIECES}
    assert len(hand_defined) == 21  # no two drawings are D4-images of each other
    assert hand_defined == enumerated
