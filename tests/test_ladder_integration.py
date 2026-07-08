"""Frozen-ladder integration (M1.6): rungs 1-3 through runner + anchored Elo.

The milestone's purpose: a cheap absolute-strength anchor — "does X beat
random yet?" — before the M4 harness exists. These tests drive real matches
end to end (agents → mirrored pairs → matchup aggregates → anchored fit) and
assert the heuristic rungs rate strictly above the rung-1 anchor. Master seeds
make every match deterministic, so the assertions are exact, not statistical.
"""

from __future__ import annotations

import pytest

from core import MobilityAgent, RandomAgent
from core.elo import fit_elo, matches_from_pairs
from core.runner import play_pairs
from games.blokus_duo import BlokusDuo
from games.blokus_duo.baselines import LargestPieceAgent, start_square_balancer
from games.connect4 import Connect4


def test_blokus_rung2_beats_random_with_positive_anchored_elo():
    pairs = play_pairs(
        BlokusDuo(),
        lambda s: LargestPieceAgent(s),
        lambda s: RandomAgent(s),
        n_pairs=6,
        seed=101,
        opening_balancer=start_square_balancer,
    )
    score = sum(p.score_a for p in pairs)
    assert score > 6.0  # strictly better than an even split of the 12 games
    ratings = fit_elo(matches_from_pairs("largest-piece", "random", pairs), anchor="random")
    assert ratings["random"] == 0.0
    assert ratings["largest-piece"] > 100.0


def test_connect4_mobility_beats_random_with_positive_anchored_elo():
    pairs = play_pairs(
        Connect4(),
        lambda s: MobilityAgent(s),
        lambda s: RandomAgent(s),
        n_pairs=30,
        seed=202,
    )
    score = sum(p.score_a for p in pairs)
    assert score > 30.0
    ratings = fit_elo(matches_from_pairs("mobility", "random", pairs), anchor="random")
    assert ratings["mobility"] > 0.0


@pytest.mark.slow
def test_blokus_three_rung_ladder_rates_above_the_anchor():
    # The full network-free ladder on the real game, small counts (the
    # mobility agent evaluates every successor: ~8 s per Blokus game).
    game = BlokusDuo()
    r1 = lambda s: RandomAgent(s)  # noqa: E731
    r2 = lambda s: LargestPieceAgent(s)  # noqa: E731
    r3 = lambda s: MobilityAgent(s)  # noqa: E731
    matches = (
        matches_from_pairs(
            "largest-piece",
            "random",
            play_pairs(game, r2, r1, n_pairs=3, seed=7, opening_balancer=start_square_balancer),
        )
        + matches_from_pairs(
            "mobility",
            "random",
            play_pairs(game, r3, r1, n_pairs=1, seed=8, opening_balancer=start_square_balancer),
        )
        + matches_from_pairs(
            "mobility",
            "largest-piece",
            play_pairs(game, r3, r2, n_pairs=1, seed=9, opening_balancer=start_square_balancer),
        )
    )
    ratings = fit_elo(matches, anchor="random")
    assert ratings["random"] == 0.0
    assert ratings["largest-piece"] > 0.0
    assert ratings["mobility"] > 0.0
