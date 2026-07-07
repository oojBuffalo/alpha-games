"""Anchored-Elo scaffolding (design doc §9, §12 M1.6 pins). Pure stdlib.

Bradley–Terry/logistic ratings on the standard 400-point scale, fit by
coordinate ascent over per-matchup aggregate scores (draws already counted
0.5), with the anchor agent pinned at exactly 0 (rung 1 in the frozen ladder)
and one virtual draw per unordered matchup so extreme small samples stay
finite. This is M1.6 scaffolding — M4's pre-registered protocol supersedes it
for the §1 verdict.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from core.runner import PairResult

# One matchup's aggregate: (agent_a, agent_b, score_a, n_games) with score_a
# the total points A took off B over n_games (win 1, draw 0.5, loss 0).
Match = tuple[str, str, float, int]


def matches_from_pairs(name_a: str, name_b: str, pairs: Sequence[PairResult]) -> list[Match]:
    """Aggregate a ``play_pairs`` result into a single :data:`Match` entry.

    Args:
        name_a: Ladder name of agent A (the runner's ``factory_a`` side).
        name_b: Ladder name of agent B.
        pairs: The mirrored-pair results of the A-vs-B match.

    Returns:
        A one-element list, ready to concatenate into a ``fit_elo`` input.
    """
    return [(name_a, name_b, sum(p.score_a for p in pairs), 2 * len(pairs))]


def _expected(delta: float) -> float:
    """Expected score of the higher-rated side at Elo difference ``delta``."""
    return 1.0 / (1.0 + 10.0 ** (-delta / 400.0))


def fit_elo(
    matches: Iterable[Match],
    anchor: str,
    tol: float = 1e-9,
    max_iter: int = 10_000,
) -> dict[str, float]:
    """Fit anchored Bradley–Terry Elo ratings from matchup aggregates.

    Coordinate ascent: each agent's rating is moved (by bisection) to the value
    equating its total expected score with its total actual score, holding the
    others fixed; the anchor never moves from 0. One virtual draw per unordered
    matchup is added before fitting (§12 M1.6 pin).

    Args:
        matches: Matchup aggregates; the matchup graph must connect every agent
            to the anchor.
        anchor: Agent name pinned at Elo 0 (the ladder's rung 1).
        tol: Convergence threshold on the largest single-rating move.
        max_iter: Hard cap on ascent sweeps.

    Returns:
        Dict of agent name to Elo rating, with ``result[anchor] == 0.0``.

    Raises:
        ValueError: If ``anchor`` appears in no matchup, or some agent is not
            connected to the anchor through the matchup graph.
    """
    # Fold in the virtual draw and collapse duplicate matchups.
    totals: dict[tuple[str, str], tuple[float, int]] = {}
    for a, b, score_a, n_games in matches:
        key, score, n = (
            ((a, b), score_a, n_games) if a <= b else ((b, a), n_games - score_a, n_games)
        )
        prev_score, prev_n = totals.get(key, (0.0, 0))
        totals[key] = (prev_score + score, prev_n + n)
    totals = {k: (s + 0.5, n + 1) for k, (s, n) in totals.items()}

    agents = sorted({name for pair in totals for name in pair})
    if anchor not in agents:
        raise ValueError(f"anchor {anchor!r} appears in no matchup")
    _assert_connected(totals, agents, anchor)

    # opponents[i] = list of (j, score_i_vs_j, n_games_ij)
    opponents: dict[str, list[tuple[str, float, int]]] = {name: [] for name in agents}
    for (a, b), (score_a, n) in totals.items():
        opponents[a].append((b, score_a, n))
        opponents[b].append((a, n - score_a, n))

    ratings = dict.fromkeys(agents, 0.0)
    for _ in range(max_iter):
        biggest_move = 0.0
        for name in agents:
            if name == anchor:
                continue
            actual = sum(score for _, score, _ in opponents[name])

            def expected_at(r: float, name: str = name) -> float:
                return sum(n * _expected(r - ratings[j]) for j, _, n in opponents[name])

            lo, hi = ratings[name] - 800.0, ratings[name] + 800.0
            while expected_at(lo) > actual:
                lo -= 800.0
            while expected_at(hi) < actual:
                hi += 800.0
            for _ in range(80):  # bisection: expected_at is monotone in r
                mid = (lo + hi) / 2.0
                if expected_at(mid) < actual:
                    lo = mid
                else:
                    hi = mid
            new_r = (lo + hi) / 2.0
            biggest_move = max(biggest_move, abs(new_r - ratings[name]))
            ratings[name] = new_r
        if biggest_move < tol:
            break
    return ratings


def _assert_connected(totals: dict, agents: list[str], anchor: str) -> None:
    """Raise ``ValueError`` unless every agent reaches the anchor via matchups."""
    reached = {anchor}
    frontier = [anchor]
    while frontier:
        cur = frontier.pop()
        for a, b in totals:
            for x, y in ((a, b), (b, a)):
                if x == cur and y not in reached:
                    reached.add(y)
                    frontier.append(y)
    missing = set(agents) - reached
    if missing:
        raise ValueError(f"agents not connected to the anchor: {sorted(missing)}")
