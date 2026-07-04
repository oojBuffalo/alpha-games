"""Exact game-theoretic solver for the MCTS correctness oracle (design doc §12, M0).

This is deliberately independent of ``core.mcts``: a slow, obviously-correct reference
against which search is differential-tested — the search analogue of "oracle first"
(§13). A search sign/perspective bug corrupts every downstream signal exactly like a
rules bug, so high-sim MCTS is required to recover these values.

The solver is **max-n** over the ``Game`` interface: at each node only the mover's
utility component is maximized and the whole per-player value vector propagates up. It
therefore makes **no strict-alternation assumption** — it is correct across consecutive
same-player moves (the pass fixture) — and reduces to minimax for 2-player zero-sum.
States must be hashable (the reference games use tuples) so results can be memoized.
"""

from __future__ import annotations

from core.game import Action, Game, State

_TOL = 1e-9


def optimal_values(game: Game, state: State, cache: dict | None = None) -> tuple[float, ...]:
    """Return the per-player value vector under optimal play by all players.

    Args:
        game: The game to solve.
        state: The position to evaluate.
        cache: Optional memoization dict, reused across calls for speed.

    Returns:
        A tuple ``(v_0, ..., v_{num_players-1})`` of utilities under optimal play.
    """
    if cache is None:
        cache = {}
    return _solve(game, state, cache)


def _solve(game: Game, state: State, cache: dict) -> tuple[float, ...]:
    cached = cache.get(state)
    if cached is not None:
        return cached
    if game.is_terminal(state):
        vals = tuple(game.terminal_utility(state, p) for p in range(game.num_players))
        cache[state] = vals
        return vals
    mover = game.current_player(state)
    best: tuple[float, ...] | None = None
    for a in game.legal_moves(state):
        child_vals = _solve(game, game.apply(state, a), cache)
        if best is None or child_vals[mover] > best[mover]:
            best = child_vals
    assert best is not None  # nonterminal states have >= 1 legal move (pass invariant)
    cache[state] = best
    return best


def optimal_actions(game: Game, state: State, cache: dict | None = None) -> list[Action]:
    """Return every action preserving the mover's optimal value at ``state``.

    An MCTS move is "optimal" iff it lies in this set — i.e. it does not throw away the
    game-theoretic value. This is the non-flaky correctness property the oracle checks.

    Args:
        game: The game to solve.
        state: A nonterminal position.
        cache: Optional memoization dict.

    Returns:
        The list of game-theoretically optimal action ids at ``state``.
    """
    if cache is None:
        cache = {}
    mover = game.current_player(state)
    target = _solve(game, state, cache)[mover]
    result = []
    for a in game.legal_moves(state):
        if _solve(game, game.apply(state, a), cache)[mover] >= target - _TOL:
            result.append(a)
    return result


def subtree_size(game: Game, state: State, cache: dict | None = None) -> int:
    """Return the number of distinct states in ``state``'s subtree (incl. ``state``).

    Used to scale MCTS simulation budgets to a position's difficulty so the oracle
    test stays fast yet gives search enough sims to solve exactly.

    Args:
        game: The game.
        state: The root of the subtree.
        cache: Optional memoization dict keyed by state.

    Returns:
        The count of distinct reachable states from ``state`` inclusive.
    """
    if cache is None:
        cache = {}
    cached = cache.get(state)
    if cached is not None:
        return cached
    if game.is_terminal(state):
        cache[state] = 1
        return 1
    total = 1
    for a in game.legal_moves(state):
        total += subtree_size(game, game.apply(state, a), cache)
    cache[state] = total
    return total


def reachable_states(game: Game, state: State | None = None) -> list[State]:
    """Enumerate all states reachable from ``state`` (default: the initial state).

    Args:
        game: The game.
        state: Starting position; defaults to ``game.initial_state()``.

    Returns:
        A list of distinct reachable states (order is deterministic: DFS discovery).
    """
    if state is None:
        state = game.initial_state()
    seen: set = set()
    order: list = []
    stack = [state]
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        order.append(s)
        if not game.is_terminal(s):
            for a in game.legal_moves(s):
                stack.append(game.apply(s, a))
    return order
