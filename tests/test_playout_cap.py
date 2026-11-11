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
3. The D12 drop (design doc §7, §10 D12) -- a fast-tier ply is searched and
   played but never stored. The stored samples are exactly the non-fast plies,
   each still carrying sparse π + z + aux; the plies below the always-full
   boundary can never be dropped; and the played and stored position counts
   reconcile with the per-tier counts game by game.
4. The consecutive-mover transition count -- taken from the complete played
   sequence, so it is the same whichever plies were dropped.
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
from games.blokus_duo import BlokusDuo
from games.blokus_duo.config import MICRO_CONFIG
from games.tictactoe import TicTacToe
from tests.fixtures.pass_game import PassGame, Scenario, consecutive_trap_game

TTT = TicTacToe()
MICRO = BlokusDuo(config=MICRO_CONFIG)

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


def test_the_tier_reaches_the_sample_and_fast_plies_are_searched_but_not_stored(searched):
    cfg = d6_cfg()
    result = play_game(TTT, None, cfg, GameRNGs.for_actor_game(*D6_GAME))

    twin = GameRNGs.for_actor_game(*D6_GAME).pcr_tier
    names = [draw_tier(cfg, ply, twin).name for ply in range(result.plies)]
    kept = [(ply, name) for ply, name in enumerate(names) if name != "fast"]
    assert [(sample.ply, sample.tier) for sample in result.samples] == kept
    assert 0 < len(kept) < result.plies  # something was dropped, something was not
    assert len(searched) == len(result.moves) == len(names)  # ...and it was still played


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
    assert dict(result.tier_counts) == Counter(names)
    assert [sample.tier for sample in result.samples] == [n for n in names if n != "fast"]


def _reference_game(game, rngs, leaf_batch_size=1, k_temp=4):
    """Play one D6 game with the engine directly, keeping every ply.

    Root noise on at every ply, D10 selection at every ply, the root's raw
    counts as the target, the subtree reused after every move, and one leaf
    batch size at every tier. Nothing here knows about storage.

    Args:
        game: The adapter to play.
        rngs: The game's stream bundle; ``pcr_tier`` is drawn once per ply
            from ply 2.
        leaf_batch_size: Maximum leaves per evaluation.
        k_temp: The D10 temperature cutoff ply.

    Returns:
        ``(moves, budgets, targets, states)``: the actions played and, per
        ply, the budget searched, the root's raw ``(action, N)`` pairs after
        that search, and the state searched.
    """
    search = MCTS(game, evaluate=None, root_noise=(0.25, 10.8, rngs.dirichlet))
    state = game.initial_state()
    search.set_root(state)
    moves, budgets, targets, states = [], [], [], []
    while not game.is_terminal(state):
        ply = len(moves)
        budget = 512 if ply < 2 else (256 if rngs.pcr_tier.random() < 0.25 else 64)
        search.run(budget, batch_size=leaf_batch_size)
        visits = search.action_visit_counts()
        budgets.append(budget)
        targets.append(tuple(visits.items()))
        states.append(state)
        action = select_move(visits, ply, k_temp, rngs.move_selection)
        search.advance(action)
        state = game.apply(state, action)
        moves.append(action)
    return moves, budgets, targets, states


@pytest.mark.parametrize("k_temp", [4, 9])
@pytest.mark.parametrize("leaf_batch_size", [1, 4])
def test_a_d6_game_matches_a_reference_loop_on_the_public_search_api(leaf_batch_size, k_temp):
    """Only the budget varies by tier, and only the fast tier goes unstored.

    A dropped ply is still searched, still played and still advances the
    reused subtree, so the reference keeps every ply and the stored samples
    are its non-fast ones. This game's two mid-tier plies are 4 and 7, so at
    ``k_temp`` 4 they take the argmax branch and at 9 every ply -- full, mid
    and fast -- is sampled.
    """
    cfg = d6_cfg(leaf_batch_size=leaf_batch_size, k_temp=k_temp)
    result = play_game(TTT, None, cfg, GameRNGs.for_actor_game(*D6_GAME))

    moves, budgets, targets, _ = _reference_game(
        TTT, GameRNGs.for_actor_game(*D6_GAME), leaf_batch_size, k_temp
    )
    kept = [ply for ply, budget in enumerate(budgets) if budget != 64]

    assert {sample.ply: sample.tier for sample in result.samples}[4] == "mid"
    assert result.moves == tuple(moves)
    assert [sample.ply for sample in result.samples] == kept
    assert [sample.sparse_pi for sample in result.samples] == [targets[ply] for ply in kept]


def test_a_d6_game_is_a_pure_function_of_its_seed():
    first = play_game(TTT, None, d6_cfg(), GameRNGs.for_actor_game(*D6_GAME))
    again = play_game(TTT, None, d6_cfg(), GameRNGs.for_actor_game(*D6_GAME))
    assert again == first


# ``play_game`` at commit c77fd68, an ancestor of the branch this change is
# built on and the last tree before the move loop was restructured, for
# Tic-Tac-Toe with the uniform evaluator at a fixed 64 sims: actor 3's games
# 0-7 under run seed 2026. Per ``leaf_batch_size``: the moves of each game, and
# one digest over every game's moves, utilities and every field a sample then
# had.
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
    # Nothing is dropped: there is no fast tier with playout-cap randomization off.
    for result in results:
        assert [sample.ply for sample in result.samples] == list(range(result.plies))
        assert result.positions_stored == result.positions_played == result.plies
        assert result.consecutive_mover_transitions == 0  # Tic-Tac-Toe strictly alternates


# ==============================================================================
# 3. The D12 drop
# ==============================================================================


class _ScriptedTiers:
    """A stand-in ``pcr_tier`` stream that draws a scripted tier sequence.

    Args:
        tiers: The tier of each drawn ply, ``"mid"`` or ``"fast"``, in order
            from the first ply at or past the always-full boundary.
    """

    def __init__(self, tiers):
        self._draws = iter([0.0 if tier == "mid" else 0.99 for tier in tiers])

    def random(self) -> float:
        """Return the next scripted draw.

        Returns:
            ``0.0`` for a mid-tier ply, ``0.99`` for a fast-tier one.

        Raises:
            StopIteration: If the game outlasts the script.
        """
        return next(self._draws)


def _scripted(tiers, game=D6_GAME) -> GameRNGs:
    """Build a game's stream bundle with its tier stream scripted.

    Args:
        tiers: The tiers :class:`_ScriptedTiers` draws, in order.
        game: The ``(run_seed, actor_id, game_index)`` every other stream is
            derived from.

    Returns:
        The bundle, differing from the seeded one in ``pcr_tier`` only.
    """
    return dataclasses.replace(GameRNGs.for_actor_game(*game), pcr_tier=_ScriptedTiers(tiers))


# Plies 2-8: ply 2 is dropped, so stored ply 3 is searched on the subtree the
# dropped search grew.
SCRIPT = ("fast", "mid", "fast", "mid", "fast", "fast", "mid")
BUDGET = {"full": 512, "mid": 256, "fast": 64}


@pytest.mark.parametrize("game", [TTT, MICRO], ids=["tictactoe-no-aux", "blokus-micro-with-aux"])
def test_a_scripted_tier_sequence_stores_exactly_the_stored_tier_plies(game, searched):
    result = play_game(game, None, d6_cfg(), _scripted(SCRIPT))

    tiers = (("full", "full") + SCRIPT)[: result.plies]
    assert result.plies >= 5  # plies 2 (dropped), 3 (stored) and 4 (dropped) were all played
    assert searched == [BUDGET[tier] for tier in tiers]  # a dropped ply is still searched
    kept = [ply for ply, tier in enumerate(tiers) if tier != "fast"]
    assert [(sample.ply, sample.tier) for sample in result.samples] == [
        (ply, tiers[ply]) for ply in kept
    ]

    # Every stored sample is a whole one: the position, the post-search raw N
    # over its full legal set, and the terminal targets from its own mover.
    moves, budgets, targets, states = _reference_game(game, _scripted(SCRIPT))
    assert result.moves == tuple(moves)
    assert budgets == searched
    for sample in result.samples:
        state = states[sample.ply]
        assert sample.planes == game.encode_state(state)
        assert sample.mover == game.current_player(state)
        assert sample.sparse_pi == targets[sample.ply]
        assert [action for action, _ in sample.sparse_pi] == list(game.legal_moves(state))
        z, aux = game.training_targets(result.terminal_state, sample.mover)
        assert (sample.z, sample.aux) == (z, tuple(aux))
        assert len(sample.aux) == len(game.value_targets.aux_names)

    # Stored π is not isolated from its dropped neighbour: ply 3 inherits the
    # visits the dropped ply-2 search put under the move it then played.
    after_a_drop = result.samples[2]
    assert after_a_drop.ply == 3
    assert sum(count for _, count in after_a_drop.sparse_pi) > BUDGET["mid"]


def test_targets_come_from_each_samples_own_mover_across_a_dropped_gap():
    """Stored plies 1 and 4 have different movers and, in a decisive game, opposite z."""
    script = ("fast", "fast", "mid", "fast", "fast", "mid", "fast")  # plies 2-3 dropped
    result = play_game(MICRO, None, d6_cfg(), _scripted(script))

    before, after = result.samples[1], result.samples[2]
    assert (before.ply, after.ply) == (1, 4)
    assert before.mover != after.mover
    assert before.z == -after.z != 0  # decisive: the other mover's targets would be wrong
    for sample in result.samples:
        z, aux = MICRO.training_targets(result.terminal_state, sample.mover)
        assert (sample.z, sample.aux) == (z, tuple(aux))
    assert before.aux != after.aux


@pytest.mark.parametrize("full_plies", [2, 4])
def test_plies_below_the_always_full_boundary_can_never_be_dropped(full_plies):
    """Structural, not statistical: the worst tier stream still drops none of them."""
    cfg = d6_cfg(pcr_full_plies=full_plies)
    result = play_game(TTT, None, cfg, _scripted(["fast"] * 9))

    assert result.plies > full_plies
    assert [(s.ply, s.tier) for s in result.samples] == [(p, "full") for p in range(full_plies)]
    assert result.positions_stored == full_plies
    # ...because a ply below the boundary is a stored tier before any draw.
    for ply in range(full_plies):
        assert draw_tier(cfg, ply, _ScriptedTiers([])).stored


# ``play_game`` on the tree this change was built on (commit 4ca68d7), where
# every ply was stored: Tic-Tac-Toe with the uniform evaluator on the D6
# schedule, actor 3's games 0-7 under run seed 2026. Per ``leaf_batch_size``:
# the plies of each game's non-fast samples, and one digest over every game's
# moves, utilities and every field of those samples. Captured from that tree
# before the append site filtered anything.
D12_GOLDEN_GAMES = 8
D12_GOLDEN = {
    1: (
        [(0, 1, 5, 6), (0, 1, 4, 7), (0, 1, 3, 7), (0, 1, 2), (0, 1, 3, 4), (0, 1, 3, 4)]
        + [(0, 1), (0, 1, 3)],
        "62bb60374973c65766c4bdc9711b28fd9d4e53e81b97bc1cdab9c78f319f663c",
    ),
    4: (
        [(0, 1, 5, 6), (0, 1, 4, 7), (0, 1, 3, 7), (0, 1, 2), (0, 1, 3, 4), (0, 1, 3, 4, 5)]
        + [(0, 1), (0, 1, 3)],
        "1084a756c077a59501dc943c71dbcf74d99384be39585ab2d505873abd05aaae",
    ),
}


def _d6_golden_games(leaf_batch_size):
    """Play the games :data:`D12_GOLDEN` was captured from.

    Args:
        leaf_batch_size: Maximum leaves per evaluation.

    Returns:
        The finished games, in game-index order.
    """
    cfg = d6_cfg(leaf_batch_size=leaf_batch_size)
    return [
        play_game(
            TTT,
            None,
            cfg,
            GameRNGs.for_actor_game(2026, 3, index),
            model_version=5,
            game_id=("run", "3", index),
        )
        for index in range(D12_GOLDEN_GAMES)
    ]


@pytest.mark.parametrize("leaf_batch_size", sorted(D12_GOLDEN))
def test_the_drop_only_filters_the_games_the_all_stored_loop_played(leaf_batch_size):
    """Same moves, same stored samples bit for bit: dropping changes nothing else."""
    digest = hashlib.sha256()
    results = _d6_golden_games(leaf_batch_size)
    for result in results:
        rows = [
            (s.planes, s.sparse_pi, s.ply, s.mover, s.z, s.aux, s.model_version, s.game_id, s.tier)
            for s in result.samples
        ]
        digest.update(repr((result.moves, result.utilities, rows)).encode())

    golden_plies, golden_digest = D12_GOLDEN[leaf_batch_size]
    assert [tuple(s.ply for s in result.samples) for result in results] == golden_plies
    assert digest.hexdigest() == golden_digest


def test_played_and_stored_positions_reconcile_with_the_tier_counts_per_game():
    results = _d6_golden_games(1)
    for result in results:
        counts = Counter(dict(result.tier_counts))
        assert result.positions_stored == len(result.samples) == counts["full"] + counts["mid"]
        assert result.positions_played == result.plies == result.positions_stored + counts["fast"]
    assert all(result.positions_stored < result.positions_played for result in results)
    assert {name for result in results for name, _ in result.tier_counts} == set(BUDGET)


# ==============================================================================
# 4. The consecutive-mover transition count
# ==============================================================================

# The mover at each ply of a scripted game with one legal action per state, so
# the sequence is this one whatever the search does. A player moves again
# straight after their own move at plies 3, 7 and 8.
CHAIN_MOVERS = (0, 1, 0, 0, 1, 0, 1, 1, 1, 0)
CHAIN_TRANSITIONS = 3


def _chain_game() -> PassGame:
    """Build the scripted game whose movers are :data:`CHAIN_MOVERS`.

    Returns:
        A single-line game: state ``i`` has the one action ``0`` leading to
        state ``i + 1``, and the state after the last ply is terminal.
    """
    plies = len(CHAIN_MOVERS)
    return PassGame(
        Scenario(
            start=0,
            to_play=dict(enumerate(CHAIN_MOVERS)),
            edges={state: [(0, state + 1)] for state in range(plies)},
            terminal={plies: (1.0, -1.0)},
        )
    )


@pytest.mark.parametrize(
    "tiers,stored_pairs",
    [
        (None, 3),  # playout-cap randomization off: every ply stored
        (["mid"] * 8, 3),  # on, nothing dropped
        (["fast"] * 8, 0),  # only plies 0-1 stored
        (["mid", "fast"] * 4, 2),  # plies 0, 1, 2, 4, 6, 8 stored
        (["fast", "mid"] * 4, 1),  # plies 0, 1, 3, 5, 7, 9 stored
    ],
    ids=["pcr-off", "all-mid", "all-fast", "mid-fast", "fast-mid"],
)
def test_transitions_are_counted_over_the_played_sequence_whatever_was_dropped(tiers, stored_pairs):
    game = _chain_game()
    if tiers is None:
        result = play_game(game, None, fixed_cfg(), GameRNGs.for_actor_game(*D6_GAME))
    else:
        result = play_game(game, None, d6_cfg(), _scripted(tiers))

    assert result.plies == len(CHAIN_MOVERS)
    assert result.consecutive_mover_transitions == CHAIN_TRANSITIONS
    # The stored samples alone would say something else once plies are dropped:
    # adjacent stored samples sharing a mover is not a consecutive move.
    movers = [sample.mover for sample in result.samples]
    assert sum(a == b for a, b in zip(movers, movers[1:], strict=False)) == stored_pairs


def test_a_forced_consecutive_move_is_one_transition():
    result = play_game(consecutive_trap_game(), None, d6_cfg(), GameRNGs.for_actor_game(*D6_GAME))
    assert [sample.mover for sample in result.samples] == [0, 0]
    assert result.consecutive_mover_transitions == 1


def test_a_strictly_alternating_game_has_no_transitions():
    results = _d6_golden_games(1)
    assert all(result.positions_stored < result.positions_played for result in results)
    assert [result.consecutive_mover_transitions for result in results] == [0] * len(results)
