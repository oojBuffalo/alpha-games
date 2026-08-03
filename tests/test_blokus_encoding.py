"""Blokus 46-plane state encoding (M2, D3/§5.2).

Plane order pinned by D3: own occupancy, opponent occupancy, 21 own-inventory
planes, 21 opponent-inventory planes (piece order = ``pieces.BASE_PIECES``,
§5.1), own/opponent monomino-last flags. Mover-relative ("own" = side to move,
no side-to-move plane), constant broadcast planes for inventory/flags, nested
14×14 tuples over {0, 1}, ``T=1``. Plus the §6.1 contract addition owned here:
``input_shape``, incl. the Othello ``(8, 8)`` backfill. Plus the M2 plane-side
symmetry: the Klein-4 ``plane_transform`` now filling the adapter's
``symmetry_group`` first slot, tested equivariant against
``encode_state ∘ state_transform`` (M1 [F4] pattern: states transform through
the module utility, planes through the transform under test).
"""

from __future__ import annotations

import random

from games.blokus_duo import BlokusDuo
from games.blokus_duo.actions import BOARD_SIZE, encode
from games.blokus_duo.bitboard import BitboardEngine
from games.blokus_duo.oracle import MONOMINO, OracleEngine
from games.blokus_duo.pieces import BASE_PIECES
from games.blokus_duo.symmetry import GROUP_NAMES, plane_transform, state_transform
from games.othello import Othello
from tests.test_blokus_oracle import make_state

GAME = BlokusDuo()
ORACLE_GAME = BlokusDuo(OracleEngine())
NUM_PIECES = len(BASE_PIECES)  # 21

# Plane indices in the D3 order (own = side to move).
OWN_OCC, OPP_OCC = 0, 1
OWN_INV, OPP_INV = 2, 2 + NUM_PIECES  # 21-plane blocks starting at 2 and 23
OWN_MONO, OPP_MONO = 2 + 2 * NUM_PIECES, 3 + 2 * NUM_PIECES  # 44, 45

ZEROS = tuple((0,) * BOARD_SIZE for _ in range(BOARD_SIZE))
ONES = tuple((1,) * BOARD_SIZE for _ in range(BOARD_SIZE))


def plane_cells(plane):
    """Return the set of ``(r, c)`` cells set to 1 in a 14×14 plane."""
    return {(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if plane[r][c]}


def bitboard_cells(bb: int):
    """Return the set of ``(r, c)`` cells set in a 196-bit occupancy int."""
    return {divmod(i, BOARD_SIZE) for i in range(BOARD_SIZE * BOARD_SIZE) if bb >> i & 1}


# --- declared surface -------------------------------------------------------------


def test_input_planes_and_shape_declared():
    assert GAME.input_planes == 46
    assert GAME.input_shape == (14, 14)


def test_othello_input_shape_backfill():
    # §6.1 contract addition: Othello's flat (65,) head carries no geometry,
    # so (H, W) must come from the declared property (M1.5 predates it).
    assert Othello().input_shape == (8, 8)


# --- initial-state goldens --------------------------------------------------------


def test_initial_state_planes_on_both_engines():
    for game in (GAME, ORACLE_GAME):
        planes = game.encode_state(game.initial_state())
        assert len(planes) == game.input_planes == 46
        h, w = game.input_shape
        assert all(len(p) == h and all(len(row) == w for row in p) for p in planes)
        # Empty board, full inventories, no completion bonus yet.
        assert planes[OWN_OCC] == ZEROS and planes[OPP_OCC] == ZEROS
        assert all(planes[i] == ONES for i in range(OWN_INV, OWN_INV + 2 * NUM_PIECES))
        assert planes[OWN_MONO] == ZEROS and planes[OPP_MONO] == ZEROS


# --- mover-relativity -------------------------------------------------------------


def test_planes_swap_perspective_after_a_move():
    # P0 opens with the monomino on (4,4); P1 becomes the mover, so "own"
    # planes flip to P1's empty board / full inventory and P0's placement
    # shows up on the opponent side.
    s1 = GAME.apply(GAME.initial_state(), encode(4, 4, 0))
    assert GAME.current_player(s1) == 1
    planes = GAME.encode_state(s1)
    assert planes[OWN_OCC] == ZEROS
    assert plane_cells(planes[OPP_OCC]) == {(4, 4)}
    assert all(planes[OWN_INV + i] == ONES for i in range(NUM_PIECES))
    assert planes[OPP_INV + MONOMINO] == ZEROS


# --- inventory planes -------------------------------------------------------------


def test_placed_piece_zeroes_its_inventory_plane():
    # Crafted mid-game state: P0 (the mover) has placed exactly the monomino.
    inv0 = [i for i in range(NUM_PIECES) if i != MONOMINO]
    s = make_state(occ0=[(4, 4)], inv0=inv0, to_play=0)
    planes = ORACLE_GAME.encode_state(s)
    assert planes[OWN_INV + MONOMINO] == ZEROS
    assert all(planes[OWN_INV + i] == ONES for i in inv0)
    assert all(planes[OPP_INV + i] == ONES for i in range(NUM_PIECES))


# --- monomino-last flag planes ----------------------------------------------------


def test_monomino_last_flag_plane_on_crafted_end_state():
    # P0 empties their inventory by placing the monomino last: the engine sets
    # the flag, the encoding must broadcast it — on the own side, because
    # apply leaves to_play at the last mover once no player can move.
    s = make_state(occ0=[(0, 0)], inv0=[MONOMINO], inv1=[], to_play=0)
    s1 = ORACLE_GAME.apply(s, encode(1, 1, 0))
    assert ORACLE_GAME.is_terminal(s1) and s1[4] is True
    planes = ORACLE_GAME.encode_state(s1)
    assert planes[OWN_MONO] == ONES
    assert planes[OPP_MONO] == ZEROS
    # Same flags viewed from the other side land on the opponent plane.
    flipped = tuple(list(s1[:6]) + [1, s1[7]])
    planes = ORACLE_GAME.encode_state(flipped)
    assert planes[OWN_MONO] == ZEROS
    assert planes[OPP_MONO] == ONES


# --- occupancy vs. engine state on random playouts --------------------------------


def test_occupancy_planes_match_engine_cells_on_seeded_playouts():
    for engine, extract in ((BitboardEngine(), bitboard_cells), (OracleEngine(), set)):
        game = BlokusDuo(engine)
        rng = random.Random(11)
        s = game.initial_state()
        while not game.is_terminal(s):
            planes = game.encode_state(s)
            mover = game.current_player(s)
            assert plane_cells(planes[OWN_OCC]) == extract(s[mover])
            assert plane_cells(planes[OPP_OCC]) == extract(s[1 - mover])
            s = game.apply(s, rng.choice(list(game.legal_moves(s))))


def test_encodings_agree_across_engines():
    # Differential: one seeded random game on the bitboard adapter, replayed
    # move-for-move through the oracle — every encoded tensor must be
    # identical (the encoding hides the occupancy representation).
    rng = random.Random(23)
    s_bb, s_or = GAME.initial_state(), ORACLE_GAME.initial_state()
    while not GAME.is_terminal(s_bb):
        assert GAME.encode_state(s_bb) == ORACLE_GAME.encode_state(s_or)
        a = rng.choice(list(GAME.legal_moves(s_bb)))
        s_bb, s_or = GAME.apply(s_bb, a), ORACLE_GAME.apply(s_or, a)
    assert ORACLE_GAME.is_terminal(s_or)
    assert GAME.encode_state(s_bb) == ORACLE_GAME.encode_state(s_or)


# --- plane-side symmetry (M2): Klein-4 equivariance -------------------------------


def _seeded_states(game, seed, stride):
    """Sample every ``stride``-th state (terminal included) of one seeded random game."""
    rng = random.Random(seed)
    s = game.initial_state()
    states = [s]
    while not game.is_terminal(s):
        s = game.apply(s, rng.choice(list(game.legal_moves(s))))
        states.append(s)
    return states[::stride]


def test_plane_transform_equivariance_on_seeded_states():
    # plane_transform_g(encode_state(s)) == encode_state(g·s) for every group
    # element, on both engines (state_transform handles bitboard ints and
    # oracle frozensets alike).
    for game, seed in ((GAME, 31), (ORACLE_GAME, 37)):
        for s in _seeded_states(game, seed, stride=4):
            planes = game.encode_state(s)
            for g in GROUP_NAMES:
                assert plane_transform(g)(planes) == game.encode_state(state_transform(g)(s)), g


def test_plane_transforms_are_involutions():
    # Klein-4: every element is self-inverse on the plane tensor too.
    for s in _seeded_states(GAME, seed=41, stride=9):
        planes = GAME.encode_state(s)
        for g in GROUP_NAMES:
            assert plane_transform(g)(plane_transform(g)(planes)) == planes, g


def test_symmetry_group_identity_round_trips_exactly():
    # symmetry_group[0] is identity: its plane slot must return an asymmetric
    # mid-game encoding unchanged (exact tuple equality).
    identity_plane_t, _ = GAME.symmetry_group[0]
    s1 = GAME.apply(GAME.initial_state(), encode(4, 4, 0))
    planes = GAME.encode_state(s1)
    assert identity_plane_t(planes) == planes
