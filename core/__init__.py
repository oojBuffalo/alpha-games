"""Game-agnostic AlphaZero engine core.

M0 surface: the ``Game`` interface + v1-envelope assertion, and the sparse,
player-aware PUCT search. Nothing here is Blokus- or network-specific; adapters
live under ``games/`` and the network plugs into search at M2. M3 adds ladder
rung 4 (``UCTAgent``, ``core/uct.py``): a standalone UCT/UCB1 searcher with
uniform-random rollouts, frozen as its own opponent rather than a PUCT config.
"""

from core.agents import Agent, MobilityAgent, RandomAgent
from core.game import EnvelopeError, Game, ValueTargetSpec, assert_v1_envelope
from core.mcts import MCTS
from core.uct import UCTAgent, UCTSearcher

__all__ = [
    "Game",
    "ValueTargetSpec",
    "EnvelopeError",
    "assert_v1_envelope",
    "MCTS",
    "Agent",
    "RandomAgent",
    "MobilityAgent",
    "UCTAgent",
    "UCTSearcher",
]
