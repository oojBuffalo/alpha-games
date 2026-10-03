"""Blokus Duo's M4 eval profile: rungs 1-4 + the start-square balancer.

The adapter's own declaration of its frozen network-free ladder
(``core.eval_profile.EvalProfile``), exposed at the package top level
(``games/blokus_duo/__init__.py``) and registered by name in
``games/registry.py`` -- the M4 analogue of how ``games/registry.py`` already
exposes a picklable ``Game`` factory per adapter. Every piece here already
exists (design doc §12 M1.6's convention pins, §9's rung numbering); this
module only assembles them into the profile the orchestrator resolves
through ``games/registry.build_eval_profile``.
"""

from __future__ import annotations

from core.agents import MobilityAgent, RandomAgent
from core.eval_profile import EvalProfile
from core.uct import UCTAgent
from games.blokus_duo.baselines import LargestPieceAgent, start_square_balancer

#: Rungs 1-4 (§12 M1.6's convention pins + M3's rung 4): uniform-random,
#: largest-piece, 1-ply mobility greedy, UCT + uniform rollouts. Every
#: constructor here already satisfies ``EvalProfile``'s seed-independent-name
#: contract (each reports a fixed ``.name`` regardless of the seed it was
#: built with).
BLOKUS_EVAL_PROFILE = EvalProfile(
    network_free_rungs={
        1: RandomAgent,
        2: LargestPieceAgent,
        3: MobilityAgent,
        4: UCTAgent,
    },
    opening_balancer=start_square_balancer,
)
