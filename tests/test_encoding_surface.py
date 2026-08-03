"""Abstract encoding surface (M2) + the TTT/Connect 4/fixture backfills it forced.

The M0 seam note in ``core/game.py`` promised "promoted to abstract methods at
M2, when the network lands" — this battery pins the promotion (a partial ``Game``
now fails at instantiation, not first use) and spot-checks the backfilled
surfaces: TTT and Connect 4's 2 mover-relative occupancy planes with identity
codecs, and the pass-game fixture's one-hot node encoding. Connect 4's standard
board is the tree's one non-square grid — the (6, 7) goldens keep any H == W
assumption from creeping in.
"""

from __future__ import annotations

import math

import pytest

from core.game import Action, Game, PlayerId, State, ValueTargetSpec
from games.blokus_duo import BlokusDuo
from games.connect4 import Connect4
from games.othello import Othello
from games.tictactoe import TicTacToe
from tests.fixtures.pass_game import consecutive_trap_game, consecutive_win_game

ENCODING_SURFACE = {
    "encode_state",
    "encode_action",
    "decode_action",
    "policy_shape",
    "input_planes",
    "input_shape",
}


def plane_cells(plane):
    """Return the set of ``(r, c)`` cells set to 1 in a nested-tuple plane."""
    return {(r, c) for r, row in enumerate(plane) for c, v in enumerate(row) if v}


# --- the promotion itself ----------------------------------------------------------


def test_encoding_surface_is_abstract():
    assert ENCODING_SURFACE <= Game.__abstractmethods__


class _NoEncodeState(Game):
    """Implements the full contract except ``encode_state`` — must not instantiate."""

    @property
    def num_players(self) -> int:
        return 2

    @property
    def is_stochastic(self) -> bool:
        return False

    @property
    def is_perfect_information(self) -> bool:
        return True

    @property
    def symmetry_group(self):
        return ()

    @property
    def value_targets(self) -> ValueTargetSpec:
        return ValueTargetSpec(primary_name="z")

    def initial_state(self) -> State:
        return 0

    def current_player(self, state: State) -> PlayerId:
        return 0

    def legal_moves(self, state: State):
        return [0]

    def apply(self, state: State, action: Action) -> State:
        return state

    def is_terminal(self, state: State) -> bool:
        return False

    def terminal_utility(self, state: State, player_id: PlayerId) -> float:
        return 0.0

    def encode_action(self, move: Action) -> Action:
        return move

    def decode_action(self, action: Action) -> Action:
        return action

    @property
    def policy_shape(self) -> tuple[int, ...]:
        return (1,)

    @property
    def input_planes(self) -> int:
        return 1

    @property
    def input_shape(self) -> tuple[int, int]:
        return (1, 1)


def test_missing_encode_state_raises_type_error_at_instantiation():
    with pytest.raises(TypeError, match="encode_state"):
        _NoEncodeState()


# --- cross-game surface coherence --------------------------------------------------

GAMES = [
    TicTacToe(),
    Connect4(),
    Connect4(4, 5, 3),
    consecutive_win_game(),
    consecutive_trap_game(),
    Othello(),
    BlokusDuo(),
]
GAME_IDS = ["ttt", "c4-6x7", "c4-4x5x3", "pass-win", "pass-trap", "othello", "blokus"]


@pytest.mark.parametrize("game", GAMES, ids=GAME_IDS)
def test_encoded_initial_state_matches_declared_shapes(game):
    planes = game.encode_state(game.initial_state())
    h, w = game.input_shape
    assert len(planes) == game.input_planes
    assert all(len(p) == h and all(len(row) == w for row in p) for p in planes)


@pytest.mark.parametrize("game", GAMES, ids=GAME_IDS)
def test_legal_action_ids_lie_within_the_policy_head(game):
    head_size = math.prod(game.policy_shape)
    for a in game.legal_moves(game.initial_state()):
        assert 0 <= a < head_size


# --- TTT backfill ------------------------------------------------------------------

TTT = TicTacToe()


def test_ttt_declared_surface():
    assert TTT.policy_shape == (9,)
    assert TTT.input_planes == 2
    assert TTT.input_shape == (3, 3)


def test_ttt_identity_codec_over_the_full_head():
    for a in range(9):
        assert TTT.decode_action(a) == a
        assert TTT.encode_action(a) == a


def test_ttt_placed_marks_land_in_the_right_plane_and_cell():
    s0 = TTT.initial_state()
    own, opp = TTT.encode_state(s0)
    assert plane_cells(own) == plane_cells(opp) == set()

    # X plays the center (cell 4): O becomes the mover, so the mark shows up
    # on the *opponent* plane at (1, 1).
    s1 = TTT.apply(s0, 4)
    own, opp = TTT.encode_state(s1)
    assert plane_cells(own) == set()
    assert plane_cells(opp) == {(1, 1)}

    # O answers in the corner (cell 0): X is the mover again — perspectives swap.
    s2 = TTT.apply(s1, 0)
    own, opp = TTT.encode_state(s2)
    assert plane_cells(own) == {(1, 1)}
    assert plane_cells(opp) == {(0, 0)}


# --- Connect 4 backfill ------------------------------------------------------------

C4 = Connect4()


def test_c4_declared_surface_is_non_square():
    assert C4.policy_shape == (7,)
    assert C4.input_planes == 2
    assert C4.input_shape == (6, 7)  # (H, W), H != W — the tree's one non-square grid
    # Parameterized variants derive their surface from the constructor.
    assert Connect4(4, 5, 3).input_shape == (4, 5)
    assert Connect4(4, 5, 3).policy_shape == (5,)


def test_c4_identity_codec_over_the_full_head():
    for a in range(7):
        assert C4.decode_action(a) == a
        assert C4.encode_action(a) == a


def test_c4_dropped_stones_land_in_the_right_plane_and_cell():
    # P0 drops in column 3: the stone falls to the bottom row (5, 3) and P1
    # becomes the mover, so it shows up on the *opponent* plane.
    s1 = C4.from_moves([3])
    own, opp = C4.encode_state(s1)
    assert plane_cells(own) == set()
    assert plane_cells(opp) == {(5, 3)}

    # P1 stacks on top: their stone sits at (4, 3) and P0 is the mover again.
    s2 = C4.from_moves([3, 3])
    own, opp = C4.encode_state(s2)
    assert plane_cells(own) == {(5, 3)}
    assert plane_cells(opp) == {(4, 3)}


# --- pass-fixture backfill ---------------------------------------------------------


def test_pass_game_one_hot_node_encoding():
    game = consecutive_win_game()  # nodes 0..6, actions {0, 1}
    assert game.input_planes == 1
    assert game.input_shape == (1, 7)
    assert game.policy_shape == (2,)
    assert game.encode_state(0) == (((1, 0, 0, 0, 0, 0, 0),),)
    assert game.encode_state(4) == (((0, 0, 0, 0, 1, 0, 0),),)
    assert game.encode_action(1) == game.decode_action(1) == 1
