"""Batched leaf inference through ``core.selfplay.play_game`` (§6.2, §12 M5).

``cfg.leaf_batch_size > 1`` routes every search through the engine's batched
selection, one evaluation per collected leaf batch. Pinned here, at the game
loop rather than the engine (``tests/test_batched_mcts_api.py`` owns the
engine): games complete, a game is a pure function of seed and config, and the
batched evaluator -- not the scalar one -- answers the leaves. No equivalence
with the serial search is claimed: virtual loss steers a batch's descents
apart, so the trajectories legitimately differ.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from core.network import (
    Network,
    NetworkConfig,
    make_batched_network_evaluator,
    make_network_evaluator,
)
from core.runconfig import SelfPlayConfig
from core.seeding import GameRNGs
from core.selfplay import play_game
from games.connect4 import Connect4
from games.tictactoe import TicTacToe

GAMES = {"tictactoe": TicTacToe(), "connect4": Connect4()}


def _cfg(leaf_batch_size: int, sims: int = 48) -> SelfPlayConfig:
    """A noisy, sampling self-play config at the given leaf batch size."""
    return SelfPlayConfig(
        sims=sims,
        k_temp=4,
        dirichlet_eps=0.25,
        dirichlet_alpha_numerator=10.8,
        root_noise=True,
        leaf_batch_size=leaf_batch_size,
    )


def _evaluators(game):
    """Seeded tiny-net ``(scalar, batched)`` evaluators over the same weights."""
    torch.manual_seed(0)
    config = dataclasses.replace(NetworkConfig.from_game(game), trunk_blocks=1, trunk_channels=4)
    net = Network(config)
    return make_network_evaluator(net, game), make_batched_network_evaluator(net, game)


@pytest.mark.parametrize("name", sorted(GAMES))
def test_a_batched_game_completes_and_is_deterministic_for_a_fixed_seed(name):
    game = GAMES[name]
    scalar, batched = _evaluators(game)
    sizes = []

    def spy(game_, states):
        sizes.append(len(states))
        return batched(game_, states)

    def play():
        return play_game(game, scalar, _cfg(4), GameRNGs.for_game(11, 0), evaluate_batch=spy)

    first = play()
    leaves_first = list(sizes)
    sizes.clear()
    again = play()

    assert game.is_terminal(first.terminal_state)
    assert first.plies == len(first.samples) > 0
    assert all(s.z is not None for s in first.samples)
    assert first == again  # moves, stored counts, targets: all of it
    assert sizes == leaves_first  # down to the batches the search formed
    assert max(sizes) > 1 and all(1 <= n <= 4 for n in sizes)


def test_the_batched_evaluator_replaces_the_scalar_one():
    game = GAMES["tictactoe"]
    _, batched = _evaluators(game)

    def scalar_must_not_run(game_, state):
        raise AssertionError("evaluate_batch supplied: the scalar evaluator is not used")

    result = play_game(
        game, scalar_must_not_run, _cfg(4), GameRNGs.for_game(3, 0), evaluate_batch=batched
    )
    assert game.is_terminal(result.terminal_state)


def test_without_a_batched_evaluator_a_batch_maps_the_scalar_one_over_its_leaves():
    game = GAMES["tictactoe"]
    scalar, batched = _evaluators(game)
    calls = []

    def counting(game_, state):
        calls.append(state)
        return scalar(game_, state)

    mapped = play_game(game, counting, _cfg(4), GameRNGs.for_game(3, 0))
    assert calls and game.is_terminal(mapped.terminal_state)
    # The M0 path (no evaluator at all) batches too.
    assert game.is_terminal(play_game(game, None, _cfg(4), GameRNGs.for_game(3, 0)).terminal_state)


def test_leaf_batch_size_one_is_the_serial_search():
    """The default is the pre-batching loop: the scalar evaluator, one leaf per call."""
    game = GAMES["tictactoe"]
    scalar, _ = _evaluators(game)
    serial = play_game(game, scalar, _cfg(1), GameRNGs.for_game(5, 0))
    default = play_game(
        game,
        scalar,
        SelfPlayConfig(
            sims=48, k_temp=4, dirichlet_eps=0.25, dirichlet_alpha_numerator=10.8, root_noise=True
        ),
        GameRNGs.for_game(5, 0),
    )
    assert serial == default
