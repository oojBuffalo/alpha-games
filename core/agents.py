"""Baseline agents over the ``Game`` interface (design doc §9, M1.6).

The agent seam the evaluation runner drives: an agent picks one legal action at
a nonterminal state, through the public ``Game`` surface only. The network-free
ladder rungs live here when they are interface-generic (rung 1 now, rung 3 with
this milestone); game-specific rungs (Blokus's rung 2) live with their adapter.
Search- and network-backed rungs (4–8) arrive at M3/M4 behind the same seam.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from core.game import Action, Game, State


class Agent(ABC):
    """A move-selecting opponent driven by the evaluation runner.

    Agents are stateful only in their RNG stream: ``select_action`` must not
    carry game state between calls, so one agent instance can be reused across
    games (the runner reseeds per pair for reproducibility).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in match records and Elo tables."""

    @abstractmethod
    def select_action(self, game: Game, state: State) -> Action:
        """Return a legal action for the mover at nonterminal ``state``."""


class RandomAgent(Agent):
    """Ladder rung 1: uniform over legal moves — the Elo anchor, fixed at 0.

    Args:
        seed: Seed for the agent's private RNG stream.
    """

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "random"

    def select_action(self, game: Game, state: State) -> Action:
        return self._rng.choice(list(game.legal_moves(state)))
