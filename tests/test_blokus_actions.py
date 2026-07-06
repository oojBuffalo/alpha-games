"""Action encode/decode goldens: bijection, in-bounds counts, literal encodings.

The [F3] literal goldens are hand-derived integers from the §5.1 convention pins
alone (never read back from the code): a consistent-but-wrong anchor or axis
convention is still bijective and passes every aggregate check, so only literal
values pin the convention itself.
"""

from __future__ import annotations

from games.blokus_duo.actions import (
    IN_BOUNDS_ACTIONS,
    NUM_ACTIONS,
    OPENING_ACTIONS,
    START_SQUARES,
    action_cells,
    decode,
    encode,
    encode_cells,
)
from games.blokus_duo.pieces import ORIENTATION_CELLS, ORIENTATION_PIECE

# --- golden counts --------------------------------------------------------------


def test_action_space_size():
    assert NUM_ACTIONS == 17_836


def test_in_bounds_count():
    assert len(IN_BOUNDS_ACTIONS) == 13_729


def test_openings_414_per_start_square():
    assert START_SQUARES == ((4, 4), (9, 9))
    for sq in START_SQUARES:
        covering = [a for a in IN_BOUNDS_ACTIONS if sq in action_cells(a)]
        assert len(covering) == 414
        assert sorted(covering) == sorted(OPENING_ACTIONS[sq])
    both = set(OPENING_ACTIONS[(4, 4)]) & set(OPENING_ACTIONS[(9, 9)])
    assert not both  # no piece bbox spans both start squares
    assert len(OPENING_ACTIONS[(4, 4)]) + len(OPENING_ACTIONS[(9, 9)]) == 828


# --- bijection -------------------------------------------------------------------


def test_encode_decode_bijection_over_in_bounds():
    seen = set()
    for a in IN_BOUNDS_ACTIONS:
        r, c, o = decode(a)
        assert encode(r, c, o) == a
        assert (r, c, o) not in seen
        seen.add((r, c, o))
    assert len(seen) == 13_729


def test_action_cells_match_orientation_and_stay_on_board():
    for a in IN_BOUNDS_ACTIONS:
        r, c, o = decode(a)
        cells = action_cells(a)
        assert len(cells) == len(ORIENTATION_CELLS[o])
        assert all(0 <= rr < 14 and 0 <= cc < 14 for rr, cc in cells)
        # cells are the orientation translated by the anchor
        assert tuple(sorted((rr - r, cc - c) for rr, cc in cells)) == ORIENTATION_CELLS[o]


def test_encode_cells_roundtrip():
    # encode_cells is the adapter-facing encode_action surface shared by both
    # engines; it must invert action_cells on every in-bounds id.
    for a in IN_BOUNDS_ACTIONS:
        assert encode_cells(action_cells(a)) == a


# --- [F3] literal hand-derived encodings -----------------------------------------


def test_literal_golden_monomino_origin():
    # Monomino: piece 0, orientation id 0. Anchor (0,0): (0*14+0)*91 + 0 = 0.
    assert encode(0, 0, 0) == 0
    assert action_cells(0) == ((0, 0),)


def test_literal_golden_monomino_start_square():
    # Monomino on start square (4,4): (4*14+4)*91 = 60*91 = 5460.
    assert encode(4, 4, 0) == 5460
    assert action_cells(5460) == ((4, 4),)


def test_literal_golden_domino_orientations():
    # Domino is piece 1; its horizontal form (0,0),(0,1) sorts before the
    # vertical (0,0),(1,0), so ids are 1 (horizontal) and 2 (vertical).
    assert encode(0, 0, 1) == 1
    assert action_cells(1) == ((0, 0), (0, 1))
    # Vertical domino at (12,13) — the extreme in-bounds anchor for a 2x1 bbox:
    # (12*14+13)*91 + 2 = 181*91 + 2 = 16473.
    a = encode(12, 13, 2)
    assert a == 16_473
    assert a in set(IN_BOUNDS_ACTIONS)
    assert action_cells(a) == ((12, 13), (13, 13))
    # One row lower is out of bounds for the vertical domino.
    assert encode(13, 13, 2) not in set(IN_BOUNDS_ACTIONS)


def test_literal_golden_L4_asymmetric():
    # Size-1..3 pieces contribute ids 0 | 1-2 | 3-4 (I3) | 5-8 (V3); the first
    # size-4 piece is I4 (ids 9-10), the second is L4, whose canonical
    # (lex-least) orientation is (0,0),(0,1),(0,2),(1,0) with id 11.
    # Anchor (2,3): (2*14+3)*91 + 11 = 31*91 + 11 = 2832.
    a = encode(2, 3, 11)
    assert a == 2832
    assert action_cells(a) == ((2, 3), (2, 4), (2, 5), (3, 3))
    assert len(ORIENTATION_CELLS[11]) == 4
    assert ORIENTATION_PIECE[11] == 5  # I1,I2,I3,V3,I4 precede it
