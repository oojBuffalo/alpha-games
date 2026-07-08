"""Anchored-Elo scaffolding (M1.6, §12 pin): Bradley-Terry fit, rung 1 at 0.

The fit runs on per-matchup aggregate scores (draws already worth 0.5), adds one
virtual draw per unordered matchup (finite ratings on extreme small samples),
and pins the anchor agent at exactly 0 on the standard 400-point scale.
"""

from __future__ import annotations

import math

import pytest

from core import RandomAgent
from core.elo import fit_elo, matches_from_pairs
from core.runner import play_pairs
from games.tictactoe import TicTacToe


def _closed_form(p: float) -> float:
    """Elo difference whose expected score is exactly ``p``."""
    return 400.0 * math.log10(p / (1.0 - p))


def test_two_agent_fit_matches_the_closed_form():
    # A scores 30/40 vs B; with the pinned virtual draw the MLE is the closed
    # form at p = 30.5/41.
    ratings = fit_elo([("a", "b", 30.0, 40)], anchor="b")
    assert ratings["b"] == 0.0
    assert abs(ratings["a"] - _closed_form(30.5 / 41)) < 0.5


def test_all_draws_fit_to_equal_ratings():
    ratings = fit_elo([("a", "b", 20.0, 40)], anchor="b")
    assert abs(ratings["a"]) < 1e-6


def test_perfect_score_stays_finite_via_the_virtual_draw():
    ratings = fit_elo([("a", "b", 10.0, 10)], anchor="b")
    assert abs(ratings["a"] - _closed_form(10.5 / 11)) < 0.5  # ~ +529, not inf


def test_anchor_is_pinned_and_differences_are_anchor_invariant():
    matches = [("a", "b", 30.0, 40), ("b", "c", 30.0, 40)]
    by_c = fit_elo(matches, anchor="c")
    by_a = fit_elo(matches, anchor="a")
    assert by_c["c"] == 0.0 and by_a["a"] == 0.0
    for x, y in (("a", "b"), ("b", "c"), ("a", "c")):
        assert abs((by_c[x] - by_c[y]) - (by_a[x] - by_a[y])) < 1e-6


def test_chain_fit_is_transitively_ordered():
    matches = [("a", "b", 30.0, 40), ("b", "c", 30.0, 40)]
    ratings = fit_elo(matches, anchor="c")
    assert ratings["a"] > ratings["b"] > ratings["c"] == 0.0
    # BT is additive on independent chains: a ~ twice b (loose bound).
    assert abs(ratings["a"] - 2 * ratings["b"]) < 1.0


def test_fit_elo_rejects_a_zero_game_match():
    # A zero-game matchup would otherwise become a lone virtual draw and fit to
    # equal ratings — a fake "tied" result masking a ladder that never ran.
    with pytest.raises(ValueError):
        fit_elo([("a", "b", 0.0, 0)], anchor="b")


def test_matches_from_pairs_aggregates_scores_and_games():
    pairs = play_pairs(
        TicTacToe(), lambda s: RandomAgent(s), lambda s: RandomAgent(s), n_pairs=6, seed=2
    )
    (match,) = matches_from_pairs("x", "y", pairs)
    a, b, score_a, n_games = match
    assert (a, b, n_games) == ("x", "y", 12)
    assert score_a == sum(p.score_a for p in pairs)
