"""Deliberately mis-declared adapters for the envelope-rejection negative test (§6.1).

Each stub declares a capability outside the v1 envelope; ``assert_v1_envelope`` (and
therefore constructing an ``MCTS`` over it) must fail loudly. The rules methods are
trivial stubs — the envelope check reads only the capability declarations, never the
rules — so the assertion must fire before any of them is touched.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.game import Action, PlayerId, State, ValueTargetSpec
from core.game import Game as _Game


class _StubGame(_Game):
    """A within-envelope stub; subclasses override one capability to breach it."""

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
        return ValueTargetSpec(primary_name="z")

    def initial_state(self) -> State:
        return 0

    def current_player(self, state: State) -> PlayerId:
        return 0

    def legal_moves(self, state: State) -> Sequence[Action]:
        return [0]

    def apply(self, state: State, action: Action) -> State:
        return state

    def is_terminal(self, state: State) -> bool:
        return False

    def terminal_utility(self, state: State, player_id: PlayerId) -> float:
        return 0.0


class ThreePlayerGame(_StubGame):
    """Breaches the envelope: declares three players (N-player is the M7 seam)."""

    @property
    def num_players(self) -> int:
        return 3


class StochasticGame(_StubGame):
    """Breaches the envelope: declares stochastic transitions."""

    @property
    def is_stochastic(self) -> bool:
        return True


class ImperfectInfoGame(_StubGame):
    """Breaches the envelope: declares imperfect information (permanently out of scope)."""

    @property
    def is_perfect_information(self) -> bool:
        return False
