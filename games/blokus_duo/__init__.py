"""Blokus Duo adapter (14×14, 2-player) — design doc §4–§6, milestone M1.

``GAME_CONFIGS`` is the adapter's declaration of the instance configs a run
config may name (``core.runconfig`` reads it by convention from the package top
level); ``BlokusDuo`` is the ``core.Game`` implementation.
"""

from games.blokus_duo.config import GAME_CONFIGS
from games.blokus_duo.game import BlokusDuo

__all__ = ["GAME_CONFIGS", "BlokusDuo"]
