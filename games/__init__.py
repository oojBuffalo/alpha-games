"""Per-game adapters behind the ``core.Game`` interface.

Rule (design doc §Repo layout): adding a game touches only ``games/`` and ``configs/``.
M0 ships the two strictly-alternating reference games; Blokus (M1) and Othello (M1.5)
follow.
"""
