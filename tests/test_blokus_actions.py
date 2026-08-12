"""Action encode/decode goldens: bijection, in-bounds counts, literal encodings.

The [F3] literal goldens are hand-derived integers from the §5.1 convention pins
alone (never read back from the code): a consistent-but-wrong anchor or axis
convention is still bijective and passes every aggregate check, so only literal
values pin the convention itself.

M2.5 task 3 adds the §5.3 micro instance on the same axis: the same bijection
over all 159 micro in-bounds placements, the same literal-encoding pins at the
micro stride, and — the part the full game has no analogue for — a **cross-check
against ``scripts/enumerate_micro_config.py``**, the standalone enumerator that
shares no code with ``games/blokus_duo/``. Asserting the package against the
script (rather than re-hardcoding the doc's table a second time) is what closes
the doc → script → code loop: the numbers in CLAUDE.md and design doc §5.3 are
the script's output, so if the parameterized package ever disagrees with it, one
of the three is wrong and this fails.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import cache
from pathlib import Path

from games.blokus_duo.actions import (
    IN_BOUNDS_ACTIONS,
    NUM_ACTIONS,
    OPENING_ACTIONS,
    START_SQUARES,
    action_cells,
    action_codec,
    decode,
    encode,
    encode_cells,
)
from games.blokus_duo.config import MICRO_CONFIG
from games.blokus_duo.game import BlokusDuo
from games.blokus_duo.pieces import (
    ORIENTATION_CELLS,
    ORIENTATION_PIECE,
    build_orientation_table,
    build_pieces,
    orientation_table_hash,
)
from games.blokus_duo.symmetry import symmetry_group
from games.blokus_duo.targets import max_score_diff

ENUMERATOR = Path(__file__).resolve().parents[1] / "scripts" / "enumerate_micro_config.py"
MICRO_CODEC = action_codec(MICRO_CONFIG)

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


# --- §5.3 micro instance: goldens vs. the independent enumerator ------------------


@cache
def enumerated() -> dict:
    """Run ``scripts/enumerate_micro_config.py`` and return its constants.

    The script is standalone (scripts/ is not a package) and deterministic; it
    grows the polyomino set, runs its own D4 transforms and enumerates its own
    placements, sharing no code with the package under test.

    Returns:
        The enumerator's constants dict.
    """
    spec = importlib.util.spec_from_file_location("enumerate_micro_config", ENUMERATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.enumerate_config()


def test_micro_config_matches_the_independent_enumeration():
    data = enumerated()
    pieces, _ = build_pieces(MICRO_CONFIG)
    game = BlokusDuo(config=MICRO_CONFIG)
    assert data["board_size"] == MICRO_CONFIG.board_size == 5
    assert data["start_squares_0indexed"] == [list(sq) for sq in MICRO_CONFIG.start_squares]
    # Shapes, not just counts: the piece set is re-derived within the subset
    # and must be the very same polyominoes in the very same order (§5.1).
    assert data["num_pieces"] == len(pieces) == 4
    assert data["pieces_canonical"] == [[list(cell) for cell in p] for p in pieces]
    assert data["squares_per_set"] == sum(len(p) for p in pieces) == 9
    assert data["input_planes"] == game.input_planes == 12
    assert data["has_monomino"] is True


def test_micro_orientation_table_matches_the_independent_enumeration():
    data = enumerated()
    table = build_orientation_table(MICRO_CONFIG)
    assert data["orientations_per_piece"] == [len(o) for o in table] == [1, 2, 2, 4]
    assert data["num_orientations"] == MICRO_CODEC.num_orientations == 9
    # Orientation ids are re-derived within the subset (invariant 4), so the
    # micro instance carries its own hash — the version key of every micro
    # fixture, checkpoint and replay dataset.
    assert data["orientation_table_hash"] == orientation_table_hash(MICRO_CONFIG)
    assert data["orientation_table_hash"] == (
        "78ea621ae2d1e27e239ecffa5ff44c793ef15f2884198a0394d394083d3e37e4"
    )


def test_micro_action_space_matches_the_independent_enumeration():
    data = enumerated()
    game = BlokusDuo(config=MICRO_CONFIG)
    assert data["policy_shape"] == list(game.policy_shape) == [5, 5, 9]
    assert data["num_raw_actions"] == MICRO_CODEC.num_actions == 225
    assert data["num_in_bounds_placements"] == len(MICRO_CODEC.in_bounds_actions) == 159
    assert data["opening_actions_per_start_square"] == {"1,1": 21, "3,3": 21}
    assert data["num_legal_openings"] == 42
    for sq in MICRO_CONFIG.start_squares:
        covering = [a for a in MICRO_CODEC.in_bounds_actions if sq in MICRO_CODEC.action_cells(a)]
        assert len(covering) == data["opening_actions_per_start_square"][f"{sq[0]},{sq[1]}"]
        assert tuple(covering) == MICRO_CODEC.opening_actions[sq]
    both = set(MICRO_CODEC.opening_actions[(1, 1)]) & set(MICRO_CODEC.opening_actions[(3, 3)])
    assert not both  # no piece bbox spans both start squares


def test_micro_score_bounds_and_group_match_the_independent_enumeration():
    data = enumerated()
    assert (data["score_min"], data["score_max"]) == (-9, 20)
    assert data["max_abs_score_diff"] == max_score_diff(MICRO_CONFIG) == 29
    # The group is computed as the D4 set-stabilizer on both sides, never
    # hardcoded — the script's own D4 code path against symmetry.py's.
    names = symmetry_group(MICRO_CONFIG).names
    assert sorted(data["symmetry_group"]) == sorted(names)
    assert set(names) == {"identity", "rot180", "diag", "antidiag"}  # Klein-4
    assert names[0] == "identity"


def test_micro_encode_decode_bijection_over_in_bounds():
    seen = set()
    for a in MICRO_CODEC.in_bounds_actions:
        r, c, o = MICRO_CODEC.decode(a)
        assert MICRO_CODEC.encode(r, c, o) == a
        assert (r, c, o) not in seen
        seen.add((r, c, o))
    assert len(seen) == enumerated()["num_in_bounds_placements"] == 159


def test_micro_action_cells_match_orientation_and_stay_on_board():
    table = build_orientation_table(MICRO_CONFIG)
    flat = [o for orients in table for o in orients]
    for a in MICRO_CODEC.in_bounds_actions:
        r, c, o = MICRO_CODEC.decode(a)
        cells = MICRO_CODEC.action_cells(a)
        assert len(cells) == len(flat[o])
        assert all(0 <= rr < 5 and 0 <= cc < 5 for rr, cc in cells)
        assert tuple(sorted((rr - r, cc - c) for rr, cc in cells)) == flat[o]


def test_micro_encode_cells_roundtrip():
    for a in MICRO_CODEC.in_bounds_actions:
        assert MICRO_CODEC.encode_cells(MICRO_CODEC.action_cells(a)) == a


def test_micro_literal_golden_encodings():
    # [F3] at the micro stride, hand-derived from the §5.1 pins: the flatten is
    # (r*5+c)*9+o, so nothing here can be right by accident from the full game's
    # (r*14+c)*91+o — a codec that leaked the full stride fails on every line.
    assert MICRO_CODEC.encode(0, 0, 0) == 0
    assert MICRO_CODEC.action_cells(0) == ((0, 0),)
    # Monomino on start square (1,1): (1*5+1)*9 + 0 = 54.
    assert MICRO_CODEC.encode(1, 1, 0) == 54
    assert MICRO_CODEC.action_cells(54) == ((1, 1),)
    # Domino is piece 1: horizontal (0,0),(0,1) is id 1, vertical is id 2.
    assert MICRO_CODEC.action_cells(MICRO_CODEC.encode(0, 0, 1)) == ((0, 0), (0, 1))
    # Vertical domino at (3,4) — the extreme in-bounds anchor for a 2x1 bbox:
    # (3*5+4)*9 + 2 = 19*9 + 2 = 173.
    a = MICRO_CODEC.encode(3, 4, 2)
    assert a == 173
    assert a in set(MICRO_CODEC.in_bounds_actions)
    assert MICRO_CODEC.action_cells(a) == ((3, 4), (4, 4))
    # One row lower is out of bounds for the vertical domino.
    assert MICRO_CODEC.encode(4, 4, 2) not in set(MICRO_CODEC.in_bounds_actions)
    # The V-tromino is the last micro piece (ids 5-8), all 2x2-bbox; id 5 is its
    # lex-least orientation. Anchor (2,3): (2*5+3)*9 + 5 = 13*9 + 5 = 122.
    a = MICRO_CODEC.encode(2, 3, 5)
    assert a == 122
    assert MICRO_CODEC.action_cells(a) == ((2, 3), (2, 4), (3, 3))
    assert MICRO_CODEC.orientation_piece[5] == 3  # I1, I2, I3 precede V3
