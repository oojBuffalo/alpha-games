"""Paired/mirrored evaluation game runner (design doc §9, M1.6).

Drives seeded agents through the public ``Game`` interface. Single games report
exact terminal utilities; the §9 protocol — mirrored pairs with seats swapped,
per-pair seeds, draws scored 0.5, game-specific opening balancing — layers on
top and emits one record per pair, the resampling unit §1's bootstrap consumes
at M4.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from core.agents import Agent
from core.game import Game


@dataclass(frozen=True)
class GameRecord:
    """Outcome of a single evaluation game.

    Attributes:
        utilities: Terminal utility per player id (seat), zero-sum in v1.
        plies: Number of actions applied from the initial state to terminal.
    """

    utilities: tuple[float, ...]
    plies: int


def play_game(game: Game, agents: Sequence[Agent]) -> GameRecord:
    """Play one game to terminal, agent ``agents[p]`` moving as player ``p``.

    Args:
        game: The game to play.
        agents: One agent per player id (seat order = player-id order).

    Returns:
        The finished game's :class:`GameRecord`.
    """
    state = game.initial_state()
    plies = 0
    while not game.is_terminal(state):
        mover = game.current_player(state)
        state = game.apply(state, agents[mover].select_action(game, state))
        plies += 1
    utilities = tuple(game.terminal_utility(state, p) for p in range(game.num_players))
    return GameRecord(utilities=utilities, plies=plies)
