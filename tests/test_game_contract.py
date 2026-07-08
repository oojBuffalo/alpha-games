"""Adapter contract tests: the pass invariant, zero-sum terminals, deterministic apply.

Run against every M0 adapter, including the pass fixtures — these are the properties
``core/`` is allowed to assume, so each adapter must actually satisfy them.
"""

from __future__ import annotations

import random

import pytest

from games.blokus_duo import BlokusDuo
from games.blokus_duo.bitboard import BitboardEngine
from games.blokus_duo.oracle import OracleEngine
from games.connect4 import Connect4
from games.othello import Othello
from games.tictactoe import TicTacToe
from tests.fixtures.pass_game import consecutive_trap_game, consecutive_win_game

# (game, playouts-per-test): cheap adapters get 60; the deliberately-slow
# Blokus oracle engine gets fewer so the non-slow battery stays fast.
GAMES = [
    (TicTacToe(), 60),
    (Connect4(), 60),
    (Connect4(4, 4, 3), 60),
    (Connect4(3, 3, 3), 60),
    (consecutive_win_game(), 60),
    (consecutive_trap_game(), 60),
    (BlokusDuo(OracleEngine()), 5),
    (BlokusDuo(BitboardEngine()), 20),
    (Othello(), 30),
]
GAME_IDS = [
    "ttt",
    "c4-6x7",
    "c4-4x4x3",
    "c4-3x3x3",
    "pass-win",
    "pass-trap",
    "blokus-oracle",
    "blokus-bitboard",
    "othello",
]


def random_playout(game, rng):
    """Play uniformly-random moves to a terminal, asserting the pass invariant en route."""
    s = game.initial_state()
    while not game.is_terminal(s):
        moves = list(game.legal_moves(s))
        # Pass invariant: every nonterminal state has a mover with >= 1 legal action.
        assert len(moves) >= 1
        assert game.current_player(s) in range(game.num_players)
        s = game.apply(s, rng.choice(moves))
    return s


@pytest.mark.parametrize("game,n_playouts", GAMES, ids=GAME_IDS)
def test_pass_invariant_holds_on_random_playouts(game, n_playouts):
    rng = random.Random(0)
    for _ in range(n_playouts):
        assert game.is_terminal(random_playout(game, rng))


@pytest.mark.parametrize("game,n_playouts", GAMES, ids=GAME_IDS)
def test_terminals_are_zero_sum(game, n_playouts):
    rng = random.Random(1)
    for _ in range(n_playouts):
        term = random_playout(game, rng)
        utils = [game.terminal_utility(term, p) for p in range(game.num_players)]
        assert abs(sum(utils)) < 1e-9
        assert all(u in (-1.0, 0.0, 1.0) for u in utils)


@pytest.mark.parametrize("game,n_playouts", GAMES, ids=GAME_IDS)
def test_apply_is_deterministic_and_nonmutating(game, n_playouts):
    del n_playouts  # single traced playout regardless of budget
    rng = random.Random(2)
    s = game.initial_state()
    while not game.is_terminal(s):
        moves = list(game.legal_moves(s))
        a = rng.choice(moves)
        s1 = game.apply(s, a)
        s2 = game.apply(s, a)
        assert s1 == s2  # deterministic transition
        assert list(game.legal_moves(s)) == moves  # apply did not mutate the input state
        s = s1
