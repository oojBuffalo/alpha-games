"""Oracle (cell-grid reference engine): openings, legality, apply, scoring.

Hand-built positions pin each legality clause separately (corner-contact
required, own-edge-contact forbidden, overlap forbidden, availability,
opponent-contact free) and the [F2] monomino-last flag semantics: the flag is
set iff the placed piece is the monomino AND the inventory empties on that
placement.

M2.5 adds the config axis (task 3): the same clauses and the same scoring cases
run on the §5.3 micro instance, where both §4 bonuses are *live* — the +15
all-placed bonus is reachable but not automatic under the reduced piece set, and
the monomino is in the subset, so the +5 monomino-last case is structurally
present rather than dead. Both are asserted reachable in actual play, not only
in hand-built states.
"""

from __future__ import annotations

import random

from games.blokus_duo.actions import OPENING_ACTIONS, action_cells, action_codec, encode
from games.blokus_duo.config import FULL_CONFIG, MICRO_CONFIG
from games.blokus_duo.game import BlokusDuo
from games.blokus_duo.oracle import OracleEngine
from games.blokus_duo.pieces import build_pieces

ENGINE = OracleEngine()
MICRO_ENGINE = OracleEngine(MICRO_CONFIG)
MICRO_CODEC = action_codec(MICRO_CONFIG)
FULL_INV = frozenset(range(21))
MICRO_INV = frozenset(range(4))
MONO = 0  # piece index of the monomino (size-1 sorts first)
DOMINO = 1
MICRO_V3 = 3  # the V-tromino, last in the micro piece order (I1, I2, I3, V3)


def make_state(
    occ0=(),
    occ1=(),
    inv0=None,
    inv1=None,
    m0=False,
    m1=False,
    to_play=0,
    config=FULL_CONFIG,
):
    """Build an oracle-layout state tuple for ``config``.

    Args:
        occ0: P1's occupied cells.
        occ1: P2's occupied cells.
        inv0: P1's remaining piece indices, or ``None`` for a full set.
        inv1: P2's remaining piece indices, or ``None`` for a full set.
        m0: P1's monomino-last flag.
        m1: P2's monomino-last flag.
        to_play: The mover.
        config: The instance the state belongs to — it fixes the piece count a
            full inventory means (21 pieces full, 4 micro).

    Returns:
        The shared engine state tuple, always nonterminal (the adapter owns
        terminal normalization).
    """
    full = frozenset(range(len(build_pieces(config)[0])))
    return (
        frozenset(occ0),
        frozenset(occ1),
        full if inv0 is None else frozenset(inv0),
        full if inv1 is None else frozenset(inv1),
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


# --- the same clauses on the §5.3 micro instance ------------------------------------


def micro_state(**kwargs):
    """Build a micro-instance oracle state (``make_state`` with the micro config)."""
    return make_state(config=MICRO_CONFIG, **kwargs)


def test_micro_monomino_only_diagonal_contact_is_legal():
    # The corner-contact / edge-contact / overlap clauses are board-size
    # independent, so the 14×14 case above must reproduce verbatim at 5×5 —
    # with the *micro* codec's ids (monomino is orientation 0 there too, but
    # the flatten stride is 9, not 91).
    s = micro_state(occ0=[(0, 0)], inv0=[MONO])
    assert MICRO_ENGINE.legal_actions(s, 0) == [MICRO_CODEC.encode(1, 1, 0)]


def test_micro_availability_empty_inventory_has_no_moves():
    assert MICRO_ENGINE.legal_actions(micro_state(occ0=[(0, 0)], inv0=[]), 0) == []


def test_micro_own_edge_contact_forbidden_for_domino():
    s = micro_state(occ0=[(0, 0)], inv0=[DOMINO])
    legal = MICRO_ENGINE.legal_actions(s, 0)
    assert legal == sorted([MICRO_CODEC.encode(1, 1, 1), MICRO_CODEC.encode(1, 1, 2)])
    for a in legal:
        assert not {(0, 1), (1, 0)} & set(MICRO_CODEC.action_cells(a))


def test_micro_mono_last_flag_set_when_monomino_empties_inventory():
    s = micro_state(occ0=[(0, 0)], inv0=[MONO])
    s1 = MICRO_ENGINE.place(s, MICRO_CODEC.encode(1, 1, 0))
    assert s1[4] is True
    assert MICRO_ENGINE.scores(s1)[0] == 20  # +15 all placed, +5 monomino last


def test_micro_mono_early_does_not_set_flag():
    s = micro_state(occ0=[(0, 0)], inv0=[MONO, DOMINO])
    s1 = MICRO_ENGINE.place(s, MICRO_CODEC.encode(1, 1, 0))
    assert s1[4] is False  # monomino placed, but inventory did not empty


def test_micro_completion_with_other_piece_scores_fifteen():
    # V-tromino, orientation 5 = ((0,0),(0,1),(1,0)), anchored at (1,1): touches
    # own (0,0) only diagonally and empties the inventory.
    s = micro_state(occ0=[(0, 0)], inv0=[MICRO_V3])
    a = MICRO_CODEC.encode(1, 1, 5)
    assert MICRO_CODEC.action_cells(a) == ((1, 1), (1, 2), (2, 1))
    assert a in MICRO_ENGINE.legal_actions(s, 0)
    s1 = MICRO_ENGINE.place(s, a)
    assert s1[4] is False
    assert MICRO_ENGINE.scores(s1)[0] == 15


def test_micro_blocked_with_mono_in_hand_gets_no_bonus():
    s = micro_state(occ0=[(0, 0)], occ1=[(1, 1)], inv0=[MONO])
    assert MICRO_ENGINE.legal_actions(s, 0) == []
    assert MICRO_ENGINE.scores(s)[0] == -1


def test_micro_initial_scores_are_minus_9():
    # 9 squares per micro set (orders 1+2+3+3), the −9 end of the pinned
    # [−9, +20] range behind the 29 aux divisor.
    assert MICRO_ENGINE.initial_state()[2] == MICRO_INV
    assert MICRO_ENGINE.scores(MICRO_ENGINE.initial_state()) == (-9, -9)


def test_micro_both_completion_bonuses_are_reachable_in_play():
    # §5.3's load-bearing claim: under the reduced set the +15 all-placed bonus
    # is reachable but not automatic, and the monomino-last +5 is live. If
    # either were structurally dead the micro loop would silently stop
    # exercising the score → z → aux path the full game depends on.
    game = BlokusDuo(MICRO_ENGINE)
    rng = random.Random(0)
    scores_seen = set()
    mono_last_seen = False
    for _ in range(40):
        s = game.initial_state()
        while not game.is_terminal(s):
            s = game.apply(s, rng.choice(list(game.legal_moves(s))))
        final = MICRO_ENGINE.scores(s)
        for p in (0, 1):
            scores_seen.add(final[p])
            # The flag is the only witness of the +5: it is not recoverable
            # from occupancy + inventory (§4 scoring-state caveat).
            assert s[4 + p] is (final[p] == 20)
            mono_last_seen |= s[4 + p]
            assert -9 <= final[p] <= 20
    assert 20 in scores_seen and mono_last_seen  # monomino placed last
    assert 15 in scores_seen  # all placed, something else last
    assert any(s < 0 for s in scores_seen)  # not automatic
