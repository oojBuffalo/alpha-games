"""Synthetic "pass game" fixtures: tiny scripted games with consecutive same-player moves.

Design doc §12 (M0): TTT and Connect 4 strictly alternate and *cannot* exercise the
player-aware backup, so the consecutive-move path needs a dedicated test from day one.
These fixtures realize a forced pass by having ``current_player`` return the same player
on consecutive turns (the opponent is skipped) — a legal realization of the pass
invariant — with decisive payoffs so backup signs are checkable against the solver.

States are integers (node ids); the whole game is a small explicit tree. Because the
reference solver is max-n (no alternation assumption), the *same* solver values these
games, and MCTS must match — pinning the consecutive-mover backup sign end to end.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from core.game import Action, PlayerId, State, ValueTargetSpec
from core.game import Game as _Game


@dataclass(frozen=True)
class Scenario:
    """An explicit game tree.

    Attributes:
        start: The initial state id.
        to_play: Mover at each internal (nonterminal) state id.
        edges: For each internal state, the ``(action_id, next_state_id)`` transitions.
        terminal: Per-terminal-state payoff ``(u_player0, u_player1)`` (zero-sum).
    """

    start: int
    to_play: Mapping[int, PlayerId]
    edges: Mapping[int, Sequence[tuple[Action, int]]]
    terminal: Mapping[int, tuple[float, float]]


class PassGame(_Game):
    """A scripted 2-player game with (possibly) consecutive same-player moves."""

    def __init__(self, scenario: Scenario):
        self.s = scenario
        self._next = {(st, a): nxt for st, moves in scenario.edges.items() for (a, nxt) in moves}

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
        return self.s.start

    def current_player(self, state: State) -> PlayerId:
        return self.s.to_play[state]

    def legal_moves(self, state: State) -> Sequence[Action]:
        return [a for (a, _) in self.s.edges[state]]

    def apply(self, state: State, action: Action) -> State:
        return self._next[(state, action)]

    def is_terminal(self, state: State) -> bool:
        return state in self.s.terminal

    def terminal_utility(self, state: State, player_id: PlayerId) -> float:
        return self.s.terminal[state][player_id]


def consecutive_win_game() -> PassGame:
    """Player 0 moves twice in a row; the winning line runs through the consecutive move.

    Tree (mover in parens)::

        0 (P0) --0--> 1 (P0, consecutive) --0--> 3  (+1, -1)   P0 wins
                                          \\-1--> 4  (-1, +1)
              \\-1--> 2 (P1) --0--> 5  ( 0,  0)
                              \\-1--> 6  (-1, +1)

    Optimal: root action 0, then (consecutive) action 0 → +1 for P0. A backup that
    wrongly assumes alternation would negate the value at state 1 and misplay.
    """
    return PassGame(
        Scenario(
            start=0,
            to_play={0: 0, 1: 0, 2: 1},
            edges={0: [(0, 1), (1, 2)], 1: [(0, 3), (1, 4)], 2: [(0, 5), (1, 6)]},
            terminal={3: (1.0, -1.0), 4: (-1.0, 1.0), 5: (0.0, 0.0), 6: (-1.0, 1.0)},
        )
    )


def consecutive_trap_game() -> PassGame:
    """Player 0 is forced into a consecutive move where one choice self-destructs.

    Tree (mover in parens)::

        0 (P0) --0--> 1 (P0, consecutive) --0--> 2  (+1, -1)   good
                                          \\-1--> 3  (-1, +1)   trap: P0 loses

    Optimal: at state 1, action 0. Tests the *negative* sign of the consecutive-mover
    backup: a sign bug would rate the losing action 1 as good for P0.
    """
    return PassGame(
        Scenario(
            start=0,
            to_play={0: 0, 1: 0},
            edges={0: [(0, 1)], 1: [(0, 2), (1, 3)]},
            terminal={2: (1.0, -1.0), 3: (-1.0, 1.0)},
        )
    )
