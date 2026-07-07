"""Paired/mirrored evaluation game runner (design doc §9, M1.6).

Drives seeded agents through the public ``Game`` interface. Single games report
exact terminal utilities; the §9 protocol — mirrored pairs with seats swapped,
per-pair seeds, draws scored 0.5, game-specific opening balancing — layers on
top and emits one record per pair, the resampling unit §1's bootstrap consumes
at M4.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from core.agents import Agent
from core.game import Game

AgentFactory = Callable[[int], Agent]


@dataclass(frozen=True)
class GameRecord:
    """Outcome of a single evaluation game.

    Attributes:
        utilities: Terminal utility per player id (seat), zero-sum in v1.
        plies: Number of actions applied from the initial state to terminal.
        opening: The first action of the game (what opening balancing keys on).
    """

    utilities: tuple[float, ...]
    plies: int
    opening: int


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
    opening: int | None = None
    while not game.is_terminal(state):
        mover = game.current_player(state)
        action = agents[mover].select_action(game, state)
        if opening is None:
            opening = action
        state = game.apply(state, action)
        plies += 1
    utilities = tuple(game.terminal_utility(state, p) for p in range(game.num_players))
    assert opening is not None  # initial states are nonterminal in v1 games
    return GameRecord(utilities=utilities, plies=plies, opening=opening)


@dataclass(frozen=True)
class PairResult:
    """Outcome of one mirrored pair — the §1/§9 bootstrap's resampling unit.

    Attributes:
        pair_index: Position of the pair within its match.
        score_a: Agent A's score over both games (win 1, draw 0.5, loss 0 each).
        score_b: Agent B's score, ``2 − score_a`` in v1.
        games: The two :class:`GameRecord`s — index 0 has A in seat 0, index 1
            has the seats swapped (B in seat 0).
    """

    pair_index: int
    score_a: float
    score_b: float
    games: tuple[GameRecord, GameRecord]


def _score(utility: float) -> float:
    """Map a terminal utility in {-1, 0, +1} to a game score (draws = 0.5)."""
    return (utility + 1.0) / 2.0


# A balancer maps (game, game-1 opening) -> predicate over game-2 openings.
OpeningBalancer = Callable[[Game, int], Callable[[int], bool]]


class _OpeningRestricted(Game):
    """Delegating wrapper that filters ``legal_moves`` at the initial state only.

    How the runner realizes opening balancing without touching the agent seam:
    game 2's agents see a game whose opening choices are restricted by the
    balancer's predicate; every other state and method delegates unchanged.
    """

    def __init__(self, inner: Game, predicate: Callable[[int], bool]):
        self._inner = inner
        self._pred = predicate
        self._initial = inner.initial_state()

    @property
    def num_players(self):
        return self._inner.num_players

    @property
    def is_stochastic(self):
        return self._inner.is_stochastic

    @property
    def is_perfect_information(self):
        return self._inner.is_perfect_information

    @property
    def symmetry_group(self):
        return self._inner.symmetry_group

    @property
    def value_targets(self):
        return self._inner.value_targets

    def initial_state(self):
        return self._initial

    def current_player(self, state):
        return self._inner.current_player(state)

    def legal_moves(self, state):
        moves = self._inner.legal_moves(state)
        if state != self._initial:
            return moves
        filtered = [a for a in moves if self._pred(a)]
        if not filtered:
            raise ValueError("opening balancer excluded every legal opening")
        return filtered

    def apply(self, state, action):
        return self._inner.apply(state, action)

    def is_terminal(self, state):
        return self._inner.is_terminal(state)

    def terminal_utility(self, state, player_id):
        return self._inner.terminal_utility(state, player_id)

    # Encoding surface: agents may key on it (e.g. Blokus's rung 2 sizes moves
    # via decode_action), so it must delegate rather than fall through to the
    # base-class not-implemented stubs.

    def encode_state(self, state):
        return self._inner.encode_state(state)

    def encode_action(self, move):
        return self._inner.encode_action(move)

    def decode_action(self, action):
        return self._inner.decode_action(action)

    @property
    def policy_shape(self):
        return self._inner.policy_shape

    @property
    def input_planes(self):
        return self._inner.input_planes


def play_pairs(
    game: Game,
    factory_a: AgentFactory,
    factory_b: AgentFactory,
    n_pairs: int,
    seed: int,
    opening_balancer: OpeningBalancer | None = None,
) -> list[PairResult]:
    """Play ``n_pairs`` mirrored pairs between two agents (§9 protocol).

    Each pair plays two games with **seats swapped** (A first, then B first).
    Agents are rebuilt through their factories for every game, with the same
    per-pair seed in both games of a pair — mirrored RNG streams — and
    independent seeds across pairs, all derived from ``seed``.

    Args:
        game: The game to evaluate on.
        factory_a: Builds agent A from a seed.
        factory_b: Builds agent B from a seed.
        n_pairs: Number of mirrored pairs to play.
        seed: Master seed; results are a pure function of it.
        opening_balancer: Optional game-specific hook (§12 M1.6 pin): given
            game 1's opening action, returns a predicate restricting game 2's
            opening (e.g. Blokus's same-start-square rule).

    Returns:
        One :class:`PairResult` per pair, in play order.
    """
    master = random.Random(seed)
    results = []
    for i in range(n_pairs):
        seed_a, seed_b = master.getrandbits(64), master.getrandbits(64)
        rec_fwd = play_game(game, (factory_a(seed_a), factory_b(seed_b)))
        game_rev: Game = game
        if opening_balancer is not None:
            game_rev = _OpeningRestricted(game, opening_balancer(game, rec_fwd.opening))
        rec_rev = play_game(game_rev, (factory_b(seed_b), factory_a(seed_a)))
        score_a = _score(rec_fwd.utilities[0]) + _score(rec_rev.utilities[1])
        results.append(
            PairResult(
                pair_index=i,
                score_a=score_a,
                score_b=2.0 - score_a,
                games=(rec_fwd, rec_rev),
            )
        )
    return results
