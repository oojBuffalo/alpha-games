"""BlokusDuo adapter: pass-invariant normalization, terminal utility, MCTS tracer.

The adapter realizes forced passes by *skipping* the blocked player (no pass
action in the ``H×W×K`` head, §5.1/§6.1): ``apply`` hands the move to the first
of [opponent, mover] with a legal action, else marks the state terminal.

M2.5 task 3 adds the config axis: the pass normalization, the monotone-blocking
property and the tracer run on the §5.3 micro instance too — where forced passes
are not hypothetical (micro lines block a player as early as ply 3, frozen in
the perft fixture's pass-aware game tree).
"""

from __future__ import annotations

import random

import pytest

from core.mcts import MCTS
from games.blokus_duo import BlokusDuo
from games.blokus_duo.actions import OPENING_ACTIONS, action_codec, encode
from games.blokus_duo.bitboard import BitboardEngine
from games.blokus_duo.config import FULL_CONFIG, MICRO_CONFIG
from games.blokus_duo.oracle import OracleEngine
from tests.test_blokus_oracle import make_state

GAME = BlokusDuo()
# make_state builds oracle-style states (frozenset occupancies), so tests that
# start from crafted positions must inject the oracle engine explicitly.
ORACLE_GAME = BlokusDuo(OracleEngine())
MICRO_GAME = BlokusDuo(config=MICRO_CONFIG)
MICRO_ORACLE_GAME = BlokusDuo(OracleEngine(MICRO_CONFIG))
MICRO_CODEC = action_codec(MICRO_CONFIG)
MONO = 0


def test_default_engine_is_bitboard():
    # PR #2 review: the no-arg adapter is the production (fast) configuration,
    # so it must be bitboard-backed — occupancies are 196-bit ints, not
    # frozensets. Reference tests opt into the oracle explicitly.
    s0 = BlokusDuo().initial_state()
    assert isinstance(s0[0], int) and isinstance(s0[1], int)


def test_capabilities_and_targets():
    assert GAME.num_players == 2
    assert GAME.is_stochastic is False
    assert GAME.is_perfect_information is True
    assert GAME.policy_shape == (14, 14, 91)
    assert GAME.value_targets.aux_names == ("score_diff",)


def test_opening_alternates_players():
    s0 = GAME.initial_state()
    assert GAME.current_player(s0) == 0
    assert len(GAME.legal_moves(s0)) == 828
    s1 = GAME.apply(s0, encode(4, 4, 0))
    assert GAME.current_player(s1) == 1
    assert not GAME.is_terminal(s1)
    assert set(GAME.legal_moves(s1)) == set(OPENING_ACTIONS[(9, 9)])


def test_forced_pass_skips_blocked_opponent():
    # Opponent (P1) has an empty inventory — no legal actions ever. After P0
    # places, the move must come straight back to P0 (consecutive mover).
    s = make_state(occ0=[(0, 0)], inv0=[MONO, 1], inv1=[])
    s1 = ORACLE_GAME.apply(s, encode(1, 1, 0))
    assert not ORACLE_GAME.is_terminal(s1)
    assert ORACLE_GAME.current_player(s1) == 0


def test_termination_when_neither_player_can_move():
    # P0 places their last piece; P1 has nothing: no mover remains.
    s = make_state(occ0=[(0, 0)], inv0=[MONO], inv1=[])
    s1 = ORACLE_GAME.apply(s, encode(1, 1, 0))
    assert ORACLE_GAME.is_terminal(s1)


def test_terminal_utility_signs_and_draw():
    # P0 finishes with monomino last (+20); P1 is stuck with the monomino (-1):
    # P1's only diagonal off (13,13) is (12,12), pre-blocked by P0. (P1 must
    # already be on the board — an empty P1 board would reopen the opening rule.)
    s = make_state(occ0=[(0, 0), (12, 12)], occ1=[(13, 13)], inv0=[MONO], inv1=[MONO])
    s1 = ORACLE_GAME.apply(s, encode(1, 1, 0))
    assert ORACLE_GAME.is_terminal(s1)
    assert ORACLE_GAME.terminal_utility(s1, 0) == 1.0
    assert ORACLE_GAME.terminal_utility(s1, 1) == -1.0
    # Symmetric crafted terminal: equal scores are a draw (z = 0), not a loss.
    draw = make_state(occ0=[(0, 0)], occ1=[(13, 13)], inv0=[MONO], inv1=[MONO], to_play=0)
    draw = tuple(list(draw[:7]) + [True])
    assert ORACLE_GAME.terminal_utility(draw, 0) == 0.0
    assert ORACLE_GAME.terminal_utility(draw, 1) == 0.0


def test_encode_decode_action_surface():
    a = encode(4, 4, 0)
    assert GAME.decode_action(a) == ((4, 4),)
    assert GAME.encode_action(((4, 4),)) == a


def test_tracer_full_random_game_through_the_contract():
    # Tracer bullet: a complete random game via only the core contract, ending
    # in a zero-sum terminal with in-range utilities.
    rng = random.Random(7)
    s = GAME.initial_state()
    plies = 0
    while not GAME.is_terminal(s):
        moves = GAME.legal_moves(s)
        assert moves  # pass invariant
        s = GAME.apply(s, rng.choice(list(moves)))
        plies += 1
        assert plies <= 42  # design doc §3: games are <= 42 plies
    u0, u1 = GAME.terminal_utility(s, 0), GAME.terminal_utility(s, 1)
    assert u0 + u1 == 0.0
    assert u0 in (-1.0, 0.0, 1.0)


def test_tracer_tiny_mcts_search():
    # Tracer bullet: the M0 engine searches Blokus end-to-end (uniform priors).
    mcts = MCTS(GAME)
    root = mcts.run(8, GAME.initial_state())
    a = mcts.best_action(root)
    assert a in set(OPENING_ACTIONS[(4, 4)]) | set(OPENING_ACTIONS[(9, 9)])
    mcts.advance(a)
    assert mcts.root is not None


def test_mcts_smoke_with_subtree_advance():
    # Search / advance / search again on the (fast) bitboard-backed adapter:
    # subtree reuse must keep returning legal actions down the tree.
    game = BlokusDuo(BitboardEngine())
    mcts = MCTS(game)
    state = game.initial_state()
    mcts.run(16, state)
    for _ in range(3):
        a = mcts.best_action()
        assert a in set(game.legal_moves(state))
        state = game.apply(state, a)
        mcts.advance(a)
        mcts.run(16)


@pytest.mark.parametrize(
    "config,games",
    [pytest.param(FULL_CONFIG, 6, id="full"), pytest.param(MICRO_CONFIG, 200, id="micro")],
)
def test_blocked_stays_blocked_on_random_playouts(config, games):
    # Blokus blocking is monotone (§4) — an adapter-level property, never a
    # core assumption: once a player has no legal placement, they never
    # regain one for the rest of the game. Asserted per config (M2.5 task 3):
    # it is a property of *this instance's* rules, and core must keep making
    # no use of it (Othello's passing is non-monotone).
    engine = BitboardEngine(config)
    game = BlokusDuo(engine)
    rng = random.Random(17)
    for _ in range(games):
        s = game.initial_state()
        blocked = {0: False, 1: False}
        while not game.is_terminal(s):
            for p in (0, 1):
                has_moves = bool(engine.legal_actions(s, p))
                if blocked[p]:
                    assert not has_moves, f"player {p} regained a move"
                blocked[p] = not has_moves
            s = game.apply(s, rng.choice(list(game.legal_moves(s))))


# --- the §5.3 micro instance through the same adapter contract ---------------------


def test_micro_forced_pass_skips_blocked_opponent():
    # Same normalization at micro scale: P2's inventory is empty, so after P1
    # places the move must come straight back to P1 (consecutive mover).
    s = make_state(occ0=[(0, 0)], inv0=[MONO, 1], inv1=[], config=MICRO_CONFIG)
    s1 = MICRO_ORACLE_GAME.apply(s, MICRO_CODEC.encode(1, 1, 0))
    assert not MICRO_ORACLE_GAME.is_terminal(s1)
    assert MICRO_ORACLE_GAME.current_player(s1) == 0


def test_micro_termination_when_neither_player_can_move():
    s = make_state(occ0=[(0, 0)], inv0=[MONO], inv1=[], config=MICRO_CONFIG)
    s1 = MICRO_ORACLE_GAME.apply(s, MICRO_CODEC.encode(1, 1, 0))
    assert MICRO_ORACLE_GAME.is_terminal(s1)


def test_micro_forced_pass_occurs_in_real_play_and_normalizes_flags():
    # Not a hand-built curiosity: micro lines exist where the mover to come is
    # blocked at ply 3, so the adapter must hand the move back rather than
    # alternate blindly (the perft battery freezes the resulting node counts).
    # The scoring flags must survive the skip untouched — a normalization that
    # rewrote them would move the +5 bonus onto the wrong terminal.
    game = BlokusDuo(BitboardEngine(MICRO_CONFIG))
    seen = 0
    for a in game.legal_moves(game.initial_state()):
        s1 = game.apply(game.initial_state(), a)
        for b in game.legal_moves(s1):
            s2 = game.apply(s1, b)
            if game.current_player(s2) != 1 or game.is_terminal(s2):
                continue
            seen += 1
            assert s2[4] is False and s2[5] is False  # no set can be complete yet
            assert not game.is_terminal(s2)
    assert seen > 0


def test_micro_capabilities_and_targets():
    assert MICRO_GAME.num_players == 2
    assert MICRO_GAME.policy_shape == (5, 5, 9)
    assert MICRO_GAME.value_targets.aux_names == ("score_diff",)


def test_micro_tracer_full_random_game_through_the_contract():
    rng = random.Random(7)
    s = MICRO_GAME.initial_state()
    plies = 0
    while not MICRO_GAME.is_terminal(s):
        moves = MICRO_GAME.legal_moves(s)
        assert moves  # pass invariant
        s = MICRO_GAME.apply(s, rng.choice(list(moves)))
        plies += 1
        assert plies <= 8  # 4 pieces each
    u0, u1 = MICRO_GAME.terminal_utility(s, 0), MICRO_GAME.terminal_utility(s, 1)
    assert u0 + u1 == 0.0
    assert u0 in (-1.0, 0.0, 1.0)
