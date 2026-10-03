"""Blokus Duo adapter (14×14, 2-player) — design doc §4–§6, milestone M1.

``GAME_CONFIGS`` is the adapter's declaration of the instance configs a run
config may name (``core.runconfig`` reads it by convention from the package top
level); ``BlokusDuo`` is the ``core.Game`` implementation. ``EVAL_PROFILE`` is
this adapter's M4 eval-profile declaration (``games/registry.py`` reads it by
the same package-top-level convention) -- rungs 1-4 plus the start-square
opening balancer (design doc §12 M1.6/§9).
"""

from games.blokus_duo.config import GAME_CONFIGS
from games.blokus_duo.eval_profile import BLOKUS_EVAL_PROFILE as EVAL_PROFILE
from games.blokus_duo.game import BlokusDuo

__all__ = ["GAME_CONFIGS", "EVAL_PROFILE", "BlokusDuo"]
