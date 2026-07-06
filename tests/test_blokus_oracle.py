"""Oracle (cell-grid reference engine): openings, legality, apply, scoring.

Hand-built positions pin each legality clause separately (corner-contact
required, own-edge-contact forbidden, overlap forbidden, availability,
opponent-contact free) and the [F2] monomino-last flag semantics: the flag is
set iff the placed piece is the monomino AND the inventory empties on that
placement.
"""

from __future__ import annotations

from games.blokus_duo.actions import OPENING_ACTIONS, action_cells, encode
from games.blokus_duo.oracle import OracleEngine

ENGINE = OracleEngine()
FULL_INV = frozenset(range(21))
MONO = 0  # piece index of the monomino (size-1 sorts first)
DOMINO = 1


def make_state(occ0=(), occ1=(), inv0=FULL_INV, inv1=FULL_INV, m0=False, m1=False, to_play=0):
    return (
        frozenset(occ0),
        frozenset(occ1),
        frozenset(inv0),
        frozenset(inv1),
        m0,
        m1,
        to_play,
        False,
    )


# --- openings --------------------------------------------------------------------


def test_initial_legal_actions_are_the_828_openings():
    legal = ENGINE.legal_actions(ENGINE.initial_state(), 0)
    assert len(legal) == 828
    assert set(legal) == set(OPENING_ACTIONS[(4, 4)]) | set(OPENING_ACTIONS[(9, 9)])


def test_p2_opening_covers_the_other_square():
    # P1 plays the monomino on (4,4); P2 must cover (9,9). A (9,9)-covering
    # placement can never reach (4,4), so all 414 remain legal.
    s = ENGINE.place(ENGINE.initial_state(), encode(4, 4, 0))
    legal = ENGINE.legal_actions(s, 1)
    assert len(legal) == 414
    assert set(legal) == set(OPENING_ACTIONS[(9, 9)])
    for a in legal:
        assert (9, 9) in action_cells(a)


# --- post-opening legality (hand-built) --------------------------------------------


def test_monomino_only_diagonal_contact_is_legal():
    # Own single cell at (0,0); only monomino in hand. The sole diagonal
    # neighbor on the board is (1,1): corner contact required, edge contact
    # ((0,1),(1,0)) forbidden, overlap ((0,0)) forbidden.
    s = make_state(occ0=[(0, 0)], inv0=[MONO])
    assert ENGINE.legal_actions(s, 0) == [encode(1, 1, 0)]


def test_availability_empty_inventory_has_no_moves():
    s = make_state(occ0=[(0, 0)], inv0=[])
    assert ENGINE.legal_actions(s, 0) == []


def test_opponent_edge_contact_free_but_overlap_forbidden():
    # Own at (0,0), opponent at (1,2), domino in hand. Horizontal at (1,1)
    # would overlap the opponent; vertical at (1,1) touches the opponent
    # edge-wise, which is free. Both touch own (0,0) only diagonally.
    s = make_state(occ0=[(0, 0)], occ1=[(1, 2)], inv0=[DOMINO])
    assert ENGINE.legal_actions(s, 0) == [encode(1, 1, 2)]


def test_own_edge_contact_forbidden_for_domino():
    # Without the opponent cell, both domino placements at (1,1) are legal;
    # placements covering (0,1) or (1,0) (own edge contact) never appear.
    s = make_state(occ0=[(0, 0)], inv0=[DOMINO])
    legal = ENGINE.legal_actions(s, 0)
    assert legal == sorted([encode(1, 1, 1), encode(1, 1, 2)])
    for a in legal:
        assert not {(0, 1), (1, 0)} & set(action_cells(a))


# --- place -------------------------------------------------------------------------


def test_place_updates_occupancy_and_inventory():
    s1 = ENGINE.place(ENGINE.initial_state(), encode(4, 4, 0))
    assert s1[0] == frozenset({(4, 4)})
    assert s1[1] == frozenset()
    assert s1[2] == FULL_INV - {MONO}
    assert s1[3] == FULL_INV
    assert s1[4] is False and s1[5] is False


# --- [F2] monomino-last flag + scoring ----------------------------------------------


def test_mono_last_flag_set_when_monomino_empties_inventory():
    s = make_state(occ0=[(0, 0)], inv0=[MONO])
    s1 = ENGINE.place(s, encode(1, 1, 0))
    assert s1[4] is True
    assert ENGINE.scores(s1)[0] == 20  # +15 all placed, +5 monomino last


def test_mono_early_does_not_set_flag():
    s = make_state(occ0=[(0, 0)], inv0=[MONO, DOMINO])
    s1 = ENGINE.place(s, encode(1, 1, 0))
    assert s1[4] is False  # monomino placed, but inventory did not empty


def test_completion_with_other_piece_scores_fifteen():
    s = make_state(occ0=[(0, 0)], inv0=[DOMINO])
    s1 = ENGINE.place(s, encode(1, 1, 2))
    assert s1[4] is False
    assert ENGINE.scores(s1)[0] == 15


def test_blocked_with_mono_in_hand_gets_no_bonus():
    # Monomino in hand but no legal placement: score is -1 (one unplaced square).
    s = make_state(occ0=[(0, 0)], occ1=[(1, 1)], inv0=[MONO])
    assert ENGINE.legal_actions(s, 0) == []
    assert ENGINE.scores(s)[0] == -1


def test_initial_scores_are_minus_89():
    assert ENGINE.scores(ENGINE.initial_state()) == (-89, -89)
