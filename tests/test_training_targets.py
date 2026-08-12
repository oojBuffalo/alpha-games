"""The generic terminal-target surface ``training_targets`` (M2.5 task 4.1, §6.1).

The one ``core/game.py`` addition of M2.5: ``training_targets(state, player_id)
-> (z, aux)``, mover-relative, so a game-agnostic self-play loop materializes
*both* training targets without importing a game — core cannot derive Blokus's
``score_diff / max|diff|`` from ``ValueTargetSpec``'s names and weights alone.

Under test here: the primary target agrees with ``terminal_utility`` from both
perspectives on every adapter (including draws, which are ``z = 0`` and not
losses); the returned aux arity matches the declared spec; Blokus's aux is the
``targets.py`` value at its config's divisor; the concrete default covers the
no-aux adapters unchanged; an adapter that declares aux heads and forgets to
override fails loudly rather than training against missing targets; and
``core/runner.py``'s delegating wrapper delegates this method too (inheriting
the ABC default there would silently drop the inner game's aux).
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from core.game import Action, Game, PlayerId, State, ValueTargetSpec
from core.runner import _OpeningRestricted
from games.blokus_duo import BlokusDuo
from games.blokus_duo.config import FULL_CONFIG, MICRO_CONFIG
from games.blokus_duo.targets import max_score_diff, value_targets
from games.connect4 import Connect4
from games.othello import Othello
from games.tictactoe import TicTacToe

# Every adapter in the repo, with a seed that reaches a terminal state (Blokus
# appears at both pinned configs — the aux divisor is the config-derived half).
ADAPTERS = [
    pytest.param(TicTacToe(), id="tictactoe"),
    pytest.param(Connect4(), id="connect4"),
    pytest.param(Connect4(rows=4, cols=4, connect=3), id="connect4-small"),
    pytest.param(Othello(), id="othello"),
    pytest.param(BlokusDuo(), id="blokus-full"),
    pytest.param(BlokusDuo(config=MICRO_CONFIG), id="blokus-micro"),
]


_TERMINALS: dict[tuple[int, int], State] = {}


def play_random(game: Game, seed: int) -> State:
    """Play uniformly random legal moves from the initial state to a terminal one.

    Memoized per ``(adapter, seed)``: the same handful of terminal states is
    reused by every contract check below, so the full-game Blokus playouts run
    once each rather than once per assertion.

    Args:
        game: The adapter to play.
        seed: RNG seed for the move choices.

    Returns:
        The terminal state reached.
    """
    key = (id(game), seed)
    if key not in _TERMINALS:
        rng = random.Random(seed)
        state = game.initial_state()
        while not game.is_terminal(state):
            state = game.apply(state, rng.choice(list(game.legal_moves(state))))
        _TERMINALS[key] = state
    return _TERMINALS[key]


class _AuxDeclaringStub(Game):
    """Adapter declaring an aux head but *not* overriding ``training_targets``.

    The failure the default's arity assertion exists to catch: the declared spec
    promises a head the adapter never materializes.
    """

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
    def symmetry_group(self) -> Sequence:
        return ()

    @property
    def value_targets(self) -> ValueTargetSpec:
        return ValueTargetSpec(primary_name="z", aux_names=("made_up",), aux_loss_weights=(0.5,))

    def initial_state(self) -> State:
        return 0

    def current_player(self, state: State) -> PlayerId:
        return 0

    def legal_moves(self, state: State) -> Sequence[Action]:
        return [0]

    def apply(self, state: State, action: Action) -> State:
        return state

    def is_terminal(self, state: State) -> bool:
        return True

    def terminal_utility(self, state: State, player_id: PlayerId) -> float:
        return 0.0

    def encode_state(self, state: State):
        return ((0,),)

    def encode_action(self, move):
        return move

    def decode_action(self, action: Action):
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


# --- the contract, on every adapter ------------------------------------------------


@pytest.mark.parametrize("game", ADAPTERS)
def test_primary_target_is_the_terminal_utility_for_both_players(game):
    for seed in range(4):
        state = play_random(game, seed)
        for player in range(game.num_players):
            z, _ = game.training_targets(state, player)
            assert z == game.terminal_utility(state, player)


@pytest.mark.parametrize("game", ADAPTERS)
def test_primary_target_is_mover_relative_and_zero_sum(game):
    for seed in range(4):
        state = play_random(game, seed)
        z0, _ = game.training_targets(state, 0)
        z1, _ = game.training_targets(state, 1)
        assert z0 == -z1  # zero-sum in v1; draws give 0 == -0
        assert z0 in (-1.0, 0.0, 1.0)


@pytest.mark.parametrize("game", ADAPTERS)
def test_aux_arity_matches_the_declared_spec(game):
    state = play_random(game, 0)
    for player in range(game.num_players):
        _, aux = game.training_targets(state, player)
        assert isinstance(aux, tuple)
        assert len(aux) == len(game.value_targets.aux_names)
        assert len(aux) == len(game.value_targets.aux_loss_weights)


def test_no_aux_adapters_take_the_concrete_default():
    # TTT, Connect 4 and Othello declare no aux head, so the ABC default is
    # their implementation — no per-adapter override, no test edits.
    for game in (TicTacToe(), Connect4(), Othello()):
        assert type(game).training_targets is Game.training_targets
        assert game.value_targets.aux_names == ()
        assert game.training_targets(play_random(game, 1), 0)[1] == ()


def test_draws_give_z_zero_not_a_loss():
    # A drawn TTT line (perfect play draws) and a drawn micro-Blokus endgame:
    # z = 0 for both players, and the aux (where declared) is 0.0 too.
    ttt = TicTacToe()
    drawn = ttt.from_grid(["XOX", "XOO", "OXX"], 0)
    assert ttt.is_terminal(drawn)
    assert ttt.training_targets(drawn, 0) == (0.0, ())
    assert ttt.training_targets(drawn, 1) == (0.0, ())

    micro = BlokusDuo(config=MICRO_CONFIG)
    # Hand-built terminal state — layout (occ0, occ1, inv0, inv1, mono_last0,
    # mono_last1, to_play, terminal). Both players hold the same inventory (the
    # two trominoes, −6 each), so the score difference — and therefore both
    # targets — is exactly zero. The aux is a one-element *tuple* parallel to
    # the declared aux_names, not a bare float: 0/29 == 0.0 either way, so only
    # the arity distinguishes a correct head layout here.
    inv = frozenset({2, 3})
    tied = (frozenset(), frozenset(), inv, inv, False, False, 0, True)
    assert micro.training_targets(tied, 0) == (0.0, (0.0,))
    assert micro.training_targets(tied, 1) == (0.0, (0.0,))


# --- Blokus: the aux value, at each config's divisor --------------------------------


@pytest.mark.parametrize("config", [FULL_CONFIG, MICRO_CONFIG])
def test_blokus_aux_equals_the_targets_module_golden(config):
    game = BlokusDuo(config=config)
    engine = game._engine  # the adapter's own engine — scores are its ground truth
    for seed in range(3):
        state = play_random(game, 100 + seed)
        scores = engine.scores(state)
        for player in range(2):
            z, aux = game.training_targets(state, player)
            expected = value_targets(scores[player], scores[1 - player], config)
            assert (z, aux) == (float(expected[0]), (expected[1],))
            assert aux[0] == pytest.approx(
                (scores[player] - scores[1 - player]) / max_score_diff(config)
            )


def test_blokus_declares_one_aux_head_at_the_pinned_weight():
    for game in (BlokusDuo(), BlokusDuo(config=MICRO_CONFIG)):
        spec = game.value_targets
        assert spec.aux_names == ("score_diff",)
        assert spec.aux_loss_weights == (0.25,)  # λ_aux, §7 — not config-dependent


def test_micro_aux_uses_the_micro_divisor_not_the_full_games():
    # The same score pair normalizes differently per instance: /29 vs /109. A
    # micro adapter accidentally routed through MAX_SCORE_DIFF would pass every
    # sign check and silently mis-scale the aux head.
    full, micro = BlokusDuo(), BlokusDuo(config=MICRO_CONFIG)
    # Layout (occ0, occ1, inv0, inv1, mono_last0, mono_last1, to_play, terminal):
    # player 0 emptied its inventory *and* holds the monomino-last flag → 15 + 5
    # = 20 (§4); player 1 still holds piece 1 (the domino, 2 squares) → −2. The
    # occupancies are irrelevant to scoring, which reads inventories and flags
    # only, so they stay empty.
    scored = (frozenset(), frozenset(), frozenset(), frozenset({1}), True, False, 0, True)
    _, aux = micro.training_targets(scored, 0)
    assert aux[0] == pytest.approx((20 - -2) / 29)
    assert aux[0] != pytest.approx((20 - -2) / 109)
    assert full.value_targets == micro.value_targets  # declared shape is shared


# --- loud failure, and the delegating wrapper ---------------------------------------


def test_declaring_aux_without_overriding_fails_loudly():
    stub = _AuxDeclaringStub()
    with pytest.raises(NotImplementedError, match="training_targets"):
        stub.training_targets(stub.initial_state(), 0)


def test_opening_restricted_wrapper_delegates_training_targets():
    # _OpeningRestricted subclasses Game, so the concrete ABC default would be
    # inherited silently and drop Blokus's aux — it must delegate instead.
    inner = BlokusDuo(config=MICRO_CONFIG)
    wrapped = _OpeningRestricted(inner, lambda a: True)
    assert type(wrapped).training_targets is not Game.training_targets
    state = play_random(inner, 7)
    for player in range(2):
        assert wrapped.training_targets(state, player) == inner.training_targets(state, player)
        assert len(wrapped.training_targets(state, player)[1]) == 1
