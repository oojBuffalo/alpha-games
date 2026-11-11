"""Playout-cap randomization at the D6 schedule (design doc §7, §10 D6, §12 M5).

Two layers of ``core.selfplay``:

1. :func:`~core.selfplay.draw_tier` -- the per-ply budget draw. Plies below the
   always-full boundary search at ``sims`` with no draw; every later ply draws
   the mid tier with the configured probability and the fast tier otherwise,
   from the game's own ``pcr-tier`` stream.
2. :func:`~core.selfplay.play_game` -- searches each ply at exactly the drawn
   budget, carries the tier to the stored sample, and accounts for the
   simulations run exactly. With playout-cap randomization off it is
   bit-identical to the fixed-budget loop it replaced (the frozen golden below).

Every position is still stored here whatever its tier: dropping the fast tier
(D12) is a separate change that filters at the same append site.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
from collections import Counter

import pytest

import core.selfplay as selfplay
from core.mcts import MCTS
from core.runconfig import SelfPlayConfig
from core.seeding import GameRNGs
from core.selfplay import Tier, draw_tier, play_game, select_move
from games.tictactoe import TicTacToe

TTT = TicTacToe()

FULL = Tier(sims=512, stored=True, name="full")
MID = Tier(sims=256, stored=True, name="mid")
FAST = Tier(sims=64, stored=False, name="fast")


def d6_cfg(**overrides) -> SelfPlayConfig:
    """Build a self-play config running the D6 schedule.

    Args:
        **overrides: Fields to replace.

    Returns:
        ``sims`` 512 with playout-cap randomization on and the schedule
        scalars at their D6 defaults (mid 256, fast 64, probability 0.25,
        always-full boundary 2), then ``overrides`` applied.
    """
    fields = {
        "sims": 512,
        "k_temp": 4,
        "dirichlet_eps": 0.25,
        "dirichlet_alpha_numerator": 10.8,
        "root_noise": True,
        "playout_cap_randomization": True,
    }
    fields.update(overrides)
    return SelfPlayConfig(**fields)


def fixed_cfg(**overrides) -> SelfPlayConfig:
    """Build a fixed-budget self-play config (playout-cap randomization off).

    Args:
        **overrides: Fields to replace.

    Returns:
        The micro-shaped config: 64 sims at every ply.
    """
    return d6_cfg(**{"sims": 64, "playout_cap_randomization": False, **overrides})


# ==============================================================================
# 1. draw_tier
# ==============================================================================


def test_pcr_off_is_the_fixed_budget_at_every_ply_and_draws_nothing():
    rng = random.Random(0)
    before = rng.getstate()
    for ply in range(40):
        assert draw_tier(fixed_cfg(), ply, rng) == Tier(sims=64, stored=True, name="fixed")
    assert rng.getstate() == before


def test_plies_below_the_boundary_are_always_full_and_draw_nothing():
    rng = random.Random(0)
    before = rng.getstate()
    for _ in range(200):  # never anything else, whatever the stream would have said
        assert draw_tier(d6_cfg(), 0, rng) == FULL
        assert draw_tier(d6_cfg(), 1, rng) == FULL
    assert rng.getstate() == before

    assert draw_tier(d6_cfg(), 2, rng) in (MID, FAST)  # the first drawn ply
    assert rng.getstate() != before


def test_the_always_full_boundary_is_the_config_field():
    assert draw_tier(d6_cfg(pcr_full_plies=4), 3, random.Random(0)) == FULL
    assert draw_tier(d6_cfg(pcr_full_plies=4), 4, random.Random(0)) in (MID, FAST)
    assert draw_tier(d6_cfg(pcr_full_plies=0), 0, random.Random(0)) in (MID, FAST)


def test_the_tier_budgets_are_the_config_fields():
    cfg = d6_cfg(sims=400, pcr_mid_sims=100, pcr_fast_sims=10)
    rng = random.Random(0)
    tiers = {draw_tier(cfg, 0, rng)} | {draw_tier(cfg, 2, rng) for _ in range(200)}
    assert tiers == {Tier(400, True, "full"), Tier(100, True, "mid"), Tier(10, False, "fast")}


# The D6 tier of plies 0-23 of actor 3's game 0 under run seed 12345. Derived
# once from the frozen ``pcr-tier`` seed in tests/test_seeding.py's GOLDEN --
# ``random.Random(17553845787436354003).random() < 0.25`` per drawn ply -- not
# read back from ``draw_tier``.
GOLDEN_TIER_SEQUENCE = (
    ["full", "full", "mid", "mid", "fast", "fast", "mid", "fast"]
    + ["fast", "fast", "fast", "mid", "mid", "fast", "fast", "fast"]
    + ["fast", "fast", "fast", "fast", "mid", "fast", "fast", "fast"]
)


def _tier_names(run_seed: int, actor_id: int, game_index: int, plies: int) -> list[str]:
    """Draw the D6 tier name of each of a game's first ``plies`` plies.

    Args:
        run_seed: The run's root seed.
        actor_id: The actor playing the game.
        game_index: The game's durable index within that actor.
        plies: How many plies to draw, from ply 0.

    Returns:
        The tier names, in ply order.
    """
    rng = GameRNGs.for_actor_game(run_seed, actor_id, game_index).pcr_tier
    return [draw_tier(d6_cfg(), ply, rng).name for ply in range(plies)]


def test_seeded_tier_sequence_golden():
    assert _tier_names(12345, 3, 0, len(GOLDEN_TIER_SEQUENCE)) == GOLDEN_TIER_SEQUENCE


def test_same_seed_same_tier_sequence_and_other_actors_and_games_differ():
    base = _tier_names(2026, 0, 5, 64)
    assert _tier_names(2026, 0, 5, 64) == base
    assert _tier_names(2026, 1, 5, 64) != base  # another actor, same game index
    assert _tier_names(2026, 0, 6, 64) != base  # another game, same actor
    assert _tier_names(2027, 0, 5, 64) != base  # another run


@pytest.mark.parametrize("probability", [0.25, 0.6])
def test_the_mid_tier_fraction_follows_the_configured_probability(probability):
    cfg = d6_cfg(pcr_mid_probability=probability)
    rng = GameRNGs.for_actor_game(12345, 3, 0).pcr_tier
    draws = Counter(draw_tier(cfg, 2 + i, rng).name for i in range(10_000))

    assert set(draws) == {"mid", "fast"}
    assert abs(draws["mid"] / 10_000 - probability) < 0.02


# ==============================================================================
# 2. play_game
# ==============================================================================


@pytest.fixture
def searched(monkeypatch) -> list[int]:
    """Record the budget of every search ``play_game`` runs.

    Args:
        monkeypatch: The pytest fixture.

    Returns:
        A list that receives each ``MCTS.run`` call's ``num_simulations``, in
        call order -- what the engine was asked to search, not what
        ``play_game`` says it searched.
    """
    budgets: list[int] = []

    class RecordingMCTS(MCTS):
        def run(self, num_simulations, root_state=None, *, batch_size=1):
            budgets.append(num_simulations)
            return super().run(num_simulations, root_state, batch_size=batch_size)

    monkeypatch.setattr(selfplay, "MCTS", RecordingMCTS)
    return budgets


# A game whose draws cover all three tiers (asserted below, so the checks
# that follow cannot pass on a degenerate all-fast game).
D6_GAME = (2026, 3, 1)


def test_each_ply_is_searched_at_its_drawn_tier_and_the_sims_are_counted_exactly(searched):
    cfg = d6_cfg()
    result = play_game(TTT, None, cfg, GameRNGs.for_actor_game(*D6_GAME))

    twin = GameRNGs.for_actor_game(*D6_GAME).pcr_tier
    drawn = [draw_tier(cfg, ply, twin) for ply in range(result.plies)]
    assert {tier.name for tier in drawn} == {"full", "mid", "fast"}

    assert searched == [tier.sims for tier in drawn]
    assert searched[:2] == [512, 512]
    assert result.sims_run == sum(searched)
    assert result.sims_run != result.plies * cfg.sims  # the old product is wrong under PCR
    assert dict(result.tier_counts) == Counter(tier.name for tier in drawn)
    assert [name for name, _ in result.tier_counts] == ["fast", "full", "mid"]
    assert sum(count for _, count in result.tier_counts) == result.plies
    hash(result)  # a frozen result stays hashable, as it was before the tier fields


def test_the_tier_reaches_the_sample_and_every_ply_is_still_stored(searched):
    cfg = d6_cfg()
    result = play_game(TTT, None, cfg, GameRNGs.for_actor_game(*D6_GAME))

    twin = GameRNGs.for_actor_game(*D6_GAME).pcr_tier
    names = [draw_tier(cfg, ply, twin).name for ply in range(result.plies)]
    assert [sample.tier for sample in result.samples] == names
    assert [sample.ply for sample in result.samples] == list(range(result.plies))
    assert "fast" in names  # ...and it is stored: no drop-policy here


class _CountingRandom(random.Random):
    """A stream that counts its ``random()`` draws."""

    draws = 0

    def random(self) -> float:
        """Draw one float, counting the call.

        Returns:
            The next float of the underlying stream.
        """
        self.draws += 1
        return super().random()


def test_the_tier_draw_comes_from_the_pcr_tier_stream_once_per_drawn_ply():
    """The only consumer of ``pcr_tier`` is the tier draw, and it uses nothing else."""
    tier_stream = _CountingRandom(0)
    rngs = dataclasses.replace(GameRNGs.for_actor_game(*D6_GAME), pcr_tier=tier_stream)
    result = play_game(TTT, None, d6_cfg(), rngs)

    assert result.plies > 2
    assert tier_stream.draws == result.plies - 2  # plies 0-1 are never drawn

    twin = random.Random(0)
    names = ["full", "full"] + [
        "mid" if twin.random() < 0.25 else "fast" for _ in range(result.plies - 2)
    ]
    assert [sample.tier for sample in result.samples] == names


@pytest.mark.parametrize("leaf_batch_size", [1, 4])
def test_a_d6_game_matches_a_reference_loop_on_the_public_search_api(leaf_batch_size):
    """Only the budget varies by tier.

    The reference plays the same game with the engine directly: root noise on
    at every ply, D10 selection at every ply, the root's raw counts as the
    target, the subtree reused after every move, and the configured leaf batch
    size at every tier.
    """
    cfg = d6_cfg(leaf_batch_size=leaf_batch_size)
    result = play_game(TTT, None, cfg, GameRNGs.for_actor_game(*D6_GAME))

    rngs = GameRNGs.for_actor_game(*D6_GAME)
    search = MCTS(TTT, evaluate=None, root_noise=(0.25, 10.8, rngs.dirichlet))
    state = TTT.initial_state()
    search.set_root(state)
    moves, targets = [], []
    while not TTT.is_terminal(state):
        ply = len(moves)
        budget = 512 if ply < 2 else (256 if rngs.pcr_tier.random() < 0.25 else 64)
        search.run(budget, batch_size=leaf_batch_size)
        visits = search.action_visit_counts()
        targets.append(tuple(visits.items()))
        action = select_move(visits, ply, 4, rngs.move_selection)
        search.advance(action)
        state = TTT.apply(state, action)
        moves.append(action)

    assert result.moves == tuple(moves)
    assert [sample.sparse_pi for sample in result.samples] == targets


def test_a_d6_game_is_a_pure_function_of_its_seed():
    first = play_game(TTT, None, d6_cfg(), GameRNGs.for_actor_game(*D6_GAME))
    again = play_game(TTT, None, d6_cfg(), GameRNGs.for_actor_game(*D6_GAME))
    assert again == first


# ``play_game`` on the tree this change was built on (commit c77fd68), for
# Tic-Tac-Toe with the uniform evaluator at a fixed 64 sims: actor 3's games
# 0-7 under run seed 2026. Per ``leaf_batch_size``: the moves of each game, and
# one digest over every game's moves, utilities and every field a sample then
# had. Captured from that tree before the move loop was restructured.
FIXED_GOLDEN_GAMES = 8
FIXED_GOLDEN = {
    1: (
        [
            (5, 3, 6, 7, 4, 2, 0, 8, 1),
            (8, 1, 7, 6, 2, 5, 0, 4, 3),
            (1, 4, 2, 0, 8, 5, 3, 6, 7),
            (7, 0, 8, 5, 6),
            (8, 6, 7, 3, 0, 4, 5, 2),
            (4, 3, 8, 0, 6, 2, 7),
            (4, 5, 8, 7, 0),
            (5, 1, 4, 7, 3),
        ],
        "6cd150212855350fdc6e8472d606437ab16c95914a75ee661334e7b996ef74ef",
    ),
    4: (
        [
            (5, 3, 6, 7, 4, 2, 0, 8, 1),
            (8, 1, 7, 6, 2, 5, 0, 4, 3),
            (1, 4, 2, 0, 8, 5, 3, 6, 7),
            (7, 0, 8, 4, 6),
            (8, 6, 7, 3, 0, 4, 5, 2),
            (4, 3, 8, 0, 6, 2, 7),
            (4, 5, 8, 7, 0),
            (5, 1, 3, 6, 4),
        ],
        "e351919d6f37397ac49bd923ceec4e4316dbe180773ed78ff76711fcb5278a39",
    ),
}


@pytest.mark.parametrize("leaf_batch_size", sorted(FIXED_GOLDEN))
def test_pcr_off_is_bit_identical_to_the_fixed_budget_loop(searched, leaf_batch_size):
    cfg = fixed_cfg(leaf_batch_size=leaf_batch_size)
    digest = hashlib.sha256()
    results = []
    for index in range(FIXED_GOLDEN_GAMES):
        rngs = GameRNGs.for_actor_game(2026, 3, index)
        result = play_game(TTT, None, cfg, rngs, model_version=5, game_id=("run", "3", index))
        rows = [
            (s.planes, s.sparse_pi, s.ply, s.mover, s.z, s.aux, s.model_version, s.game_id)
            for s in result.samples
        ]
        digest.update(repr((result.moves, result.utilities, rows)).encode())
        results.append(result)
        # No tier draw happened: the stream is exactly where it started.
        fresh = GameRNGs.for_actor_game(2026, 3, index).pcr_tier
        assert rngs.pcr_tier.getstate() == fresh.getstate()

    golden_moves, golden_digest = FIXED_GOLDEN[leaf_batch_size]
    assert [result.moves for result in results] == golden_moves
    assert digest.hexdigest() == golden_digest

    plies = sum(result.plies for result in results)
    assert searched == [64] * plies
    assert [result.sims_run for result in results] == [64 * r.plies for r in results]
    assert all(result.tier_counts == (("fixed", result.plies),) for result in results)
    assert {sample.tier for result in results for sample in result.samples} == {"fixed"}
