"""BlokusDuo adapter: the ``Game`` contract over an interchangeable rules engine.

The adapter owns the §6.1 pass-invariant normalization — engines only place
pieces and report legality/scores. Forced passes are realized by *skipping* the
blocked player (no pass action in the 14×14×91 head): ``apply`` hands the move
to the first of [opponent, mover] with a legal action, else marks the state
terminal. Blocking is monotone in Blokus, but that is an adapter-level fact —
core never assumes it.

The engine (oracle or bitboard) is injected so the whole contract battery and
the differential fuzz can run against either implementation [F8].
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from core.game import Action, Game, PlayerId, State, SymmetryElement, ValueTargetSpec
from games.blokus_duo.actions import BOARD_SIZE, NUM_ORIENTATIONS, action_cells, encode_cells
from games.blokus_duo.oracle import OracleEngine
from games.blokus_duo.symmetry import GROUP_NAMES, full_permutation, state_transform
from games.blokus_duo.targets import value_target_spec, value_targets

_TO_PLAY, _TERMINAL = 6, 7


class BlokusDuo(Game):
    """Blokus Duo (14×14, 2-player) behind the generic ``Game`` interface.

    Args:
        engine: Rules engine providing ``initial_state`` / ``legal_actions`` /
            ``place`` / ``scores`` over the shared state tuple. Defaults to the
            cell-grid oracle; the bitboard engine drops in unchanged.
    """

    def __init__(self, engine=None):
        self._engine = engine if engine is not None else OracleEngine()

    # --- declared capabilities ---------------------------------------------------

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
    def symmetry_group(self) -> Sequence[SymmetryElement]:
        # Klein four-group (§8, D9): first slot is the state-level transform
        # (M2 rebinds the plane-tensor side), second the full 17,836-length
        # head permutation with identity filler on off-support ids [F6].
        return tuple(
            (state_transform(g), full_permutation(g)) for g in GROUP_NAMES
        )

    @property
    def value_targets(self) -> ValueTargetSpec:
        return value_target_spec()

    # --- core contract -------------------------------------------------------------

    def initial_state(self) -> State:
        return self._engine.initial_state()

    def current_player(self, state: State) -> PlayerId:
        return state[_TO_PLAY]

    def legal_moves(self, state: State) -> Sequence[Action]:
        return self._engine.legal_actions(state, state[_TO_PLAY])

    def apply(self, state: State, action: Action) -> State:
        """Place for the mover, then normalize ``to_play``/``terminal`` (§6.1).

        Args:
            state: A nonterminal state whose mover has ``action`` legal.
            action: Flat action id to play. Behavior is undefined for illegal
                actions (core only ever applies ids from ``legal_moves``).

        Returns:
            The successor state; its mover is guaranteed >= 1 legal action
            unless the state is terminal (pass invariant).
        """
        mover = state[_TO_PLAY]
        nxt = self._engine.place(state, action)
        parts = list(nxt)
        for player in (1 - mover, mover):
            if self._engine.legal_actions(nxt, player):
                parts[_TO_PLAY] = player
                parts[_TERMINAL] = False
                return tuple(parts)
        parts[_TERMINAL] = True
        return tuple(parts)

    def is_terminal(self, state: State) -> bool:
        return state[_TERMINAL]

    def terminal_utility(self, state: State, player_id: PlayerId) -> float:
        scores = self._engine.scores(state)
        z, _ = value_targets(scores[player_id], scores[1 - player_id])
        return float(z)

    # --- encoding surface (action side owned by M1; planes arrive at M2) ------------

    def encode_action(self, move: Any) -> Action:
        """Encode absolute placement cells as a flat action id."""
        return encode_cells(move)

    def decode_action(self, action: Action) -> Any:
        """Decode a flat action id into its absolute placement cells."""
        return action_cells(action)

    @property
    def policy_shape(self) -> tuple[int, ...]:
        return (BOARD_SIZE, BOARD_SIZE, NUM_ORIENTATIONS)
