"""BlokusDuo adapter: pass-invariant normalization, terminal utility, MCTS tracer.

The adapter realizes forced passes by *skipping* the blocked player (no pass
action in the 14×14×91 head, §5.1/§6.1): ``apply`` hands the move to the first
of [opponent, mover] with a legal action, else marks the state terminal.
"""

from __future__ import annotations

import random

from core.mcts import MCTS
from games.blokus_duo import BlokusDuo
from games.blokus_duo.actions import OPENING_ACTIONS, encode
from tests.test_blokus_oracle import make_state

GAME = BlokusDuo()
MONO = 0


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
    s1 = GAME.apply(s, encode(1, 1, 0))
    assert not GAME.is_terminal(s1)
    assert GAME.current_player(s1) == 0


def test_termination_when_neither_player_can_move():
    # P0 places their last piece; P1 has nothing: no mover remains.
    s = make_state(occ0=[(0, 0)], inv0=[MONO], inv1=[])
    s1 = GAME.apply(s, encode(1, 1, 0))
    assert GAME.is_terminal(s1)


def test_terminal_utility_signs_and_draw():
    # P0 finishes with monomino last (+20); P1 is stuck with the monomino (-1):
    # P1's only diagonal off (13,13) is (12,12), pre-blocked by P0. (P1 must
    # already be on the board — an empty P1 board would reopen the opening rule.)
    s = make_state(occ0=[(0, 0), (12, 12)], occ1=[(13, 13)], inv0=[MONO], inv1=[MONO])
    s1 = GAME.apply(s, encode(1, 1, 0))
    assert GAME.is_terminal(s1)
    assert GAME.terminal_utility(s1, 0) == 1.0
    assert GAME.terminal_utility(s1, 1) == -1.0
    # Symmetric crafted terminal: equal scores are a draw (z = 0), not a loss.
    draw = make_state(occ0=[(0, 0)], occ1=[(13, 13)], inv0=[MONO], inv1=[MONO], to_play=0)
    draw = tuple(list(draw[:7]) + [True])
    assert GAME.terminal_utility(draw, 0) == 0.0
    assert GAME.terminal_utility(draw, 1) == 0.0


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
    from games.blokus_duo.bitboard import BitboardEngine

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


def test_blocked_stays_blocked_on_random_playouts():
    # Blokus blocking is monotone (§4) — an adapter-level property, never a
    # core assumption: once a player has no legal placement, they never
    # regain one for the rest of the game.
    from games.blokus_duo.bitboard import BitboardEngine

    engine = BitboardEngine()
    game = BlokusDuo(engine)
    rng = random.Random(17)
    for _ in range(6):
        s = game.initial_state()
        blocked = {0: False, 1: False}
        while not game.is_terminal(s):
            for p in (0, 1):
                has_moves = bool(engine.legal_actions(s, p))
                if blocked[p]:
                    assert not has_moves, f"player {p} regained a move"
                blocked[p] = not has_moves
            s = game.apply(s, rng.choice(list(game.legal_moves(s))))
