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


def _start_squares(game: Game) -> tuple[tuple[int, int], ...]:
    """Return the start squares of the Blokus instance ``game`` plays.

    Read from the adapter's declared instance config rather than hardcoded, so
    the balancer works unchanged on the §5.3 micro instance ((1,1)/(3,3)) and on
    the full game ((4,4)/(9,9)) — §12 M2.5 evaluates on the micro board through
    this same hook.

    Args:
        game: The Blokus adapter.

    Returns:
        The instance's two 0-indexed start squares.

    Raises:
        TypeError: If ``game`` declares no Blokus instance config — the balancer
            is game-specific by construction (the runner stays game-agnostic),
            so a foreign game must fail loudly rather than be balanced against
            some default board.
    """
    config = getattr(game, "config", None)
    squares = getattr(config, "start_squares", None)
    if squares is None:
        raise TypeError(
            f"{type(game).__name__} declares no Blokus instance config; the start-square "
            "balancer is specific to games/blokus_duo/"
        )
    return tuple(squares)


def start_square_balancer(game: Game, opening: Action) -> Callable[[Action], bool]:
    """Balancer for the pair runner: same start square across a pair (§12 M1.6).

    Args:
        game: The Blokus adapter (used for ``decode_action`` and for the
            instance's start squares).
        opening: Game 1's opening action.

    Returns:
        Predicate accepting exactly the openings covering the start square that
        ``opening`` covered.

    Raises:
        TypeError: If ``game`` is not a Blokus adapter (see :func:`_start_squares`).
        ValueError: If ``opening`` covers no start square, or covers both — an
            opening that is not a legal §4 opener would silently unbalance the
            pair instead of raising.
    """
    squares = _start_squares(game)
    cells = set(game.decode_action(opening))
    covered = [s for s in squares if s in cells]
    if len(covered) != 1:
        raise ValueError(
            f"opening action {opening} covers {len(covered)} start square(s) "
            f"({covered}); a legal opening covers exactly one"
        )
    (square,) = covered
    return lambda a: square in set(game.decode_action(a))
