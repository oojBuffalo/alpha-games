"""Blokus-specific ladder pieces (design doc §9, §12 M1.6 pins).

Rung 2 — the largest-piece/coverage heuristic — and the start-square opening
balancer for the mirrored-pair runner. Both go through the adapter surface only
(``decode_action`` cell sets); no engine internals.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from core.agents import Agent
from core.game import Action, Game, State

_START_SQUARES = ((4, 4), (9, 9))


class LargestPieceAgent(Agent):
    """Ladder rung 2: play a maximal-size piece, uniform-random among ties.

    Args:
        seed: Seed for the tie-breaking RNG stream.
    """

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "largest-piece"

    def select_action(self, game: Game, state: State) -> Action:
        moves = list(game.legal_moves(state))
        sizes = [len(game.decode_action(a)) for a in moves]
        best = max(sizes)
        return self._rng.choice([a for a, sz in zip(moves, sizes, strict=True) if sz == best])


def start_square_balancer(game: Game, opening: Action) -> Callable[[Action], bool]:
    """Balancer for the pair runner: same start square across a pair (§12 M1.6).

    Args:
        game: The Blokus adapter (used only for ``decode_action``).
        opening: Game 1's opening action.

    Returns:
        Predicate accepting exactly the openings covering the start square that
        ``opening`` covered.
    """
    cells = set(game.decode_action(opening))
    (square,) = [s for s in _START_SQUARES if s in cells]
    return lambda a: square in set(game.decode_action(a))
