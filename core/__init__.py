"""Game-agnostic AlphaZero engine core.

M0 surface: the ``Game`` interface + v1-envelope assertion, and the sparse,
player-aware PUCT search. Nothing here is Blokus- or network-specific; adapters
live under ``games/`` and the network plugs into search at M2.
"""

from core.agents import Agent, RandomAgent
from core.game import EnvelopeError, Game, ValueTargetSpec, assert_v1_envelope
from core.mcts import MCTS

__all__ = [
    "Game",
    "ValueTargetSpec",
    "EnvelopeError",
    "assert_v1_envelope",
    "MCTS",
    "Agent",
    "RandomAgent",
]
