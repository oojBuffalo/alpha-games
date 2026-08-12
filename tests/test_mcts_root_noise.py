"""D7 root-Dirichlet-noise hook tests (design doc §7 D7; M2.5 task 6.1 / M3 task 8.1).

The hook mixes ``P' = (1-eps)·P + eps·Dir(alpha_numerator/#legal)`` into the *root's*
priors, once per root incarnation — including a subtree promoted to root by
:meth:`MCTS.advance`. It is exploration for self-play only, so the default is off and the
noiseless engine must stay bit-identical (the pre-existing MCTS battery, run unedited, is
the other half of that proof).
"""

from __future__ import annotations

import random

import pytest

from core import MCTS
from games.connect4 import Connect4
from games.tictactoe import TicTacToe

EPS = 0.25  # D7
ALPHA_NUMERATOR = 10.8  # D7


def _priors(mcts: MCTS) -> dict[int, float]:
    """Return the current root's ``{action: prior}`` map."""
    root = mcts.root
    return dict(zip(root.actions, root.P, strict=True))


def _search(game, *, noise_seed: int | None, sims: int = 40) -> MCTS:
    """Run a search from the initial state, with or without root noise."""
    noise = None if noise_seed is None else (EPS, ALPHA_NUMERATOR, random.Random(noise_seed))
    m = MCTS(game, root_noise=noise)
    m.run(sims, root_state=game.initial_state())
    return m


# --- the hook is off by default -------------------------------------------------------


def test_default_is_off_and_priors_are_untouched():
    game = TicTacToe()
    m = _search(game, noise_seed=None)
    assert m.root_noise is None
    # Exactly uniform: the pre-hook engine's priors, to the bit.
    assert m.root.P == [1.0 / 9] * 9


def test_off_path_matches_a_run_with_a_noise_rng_that_is_never_consumed():
    # Constructing the hook must not change search behavior until it is enabled: an
    # explicit root_noise=None instance reproduces the default instance's visit counts.
    game = Connect4(4, 4, 3)
    a = MCTS(game)
    b = MCTS(game, root_noise=None)
    a.run(120, root_state=game.initial_state())
    b.run(120, root_state=game.initial_state())
    assert a.action_visit_counts() == b.action_visit_counts()


# --- support and normalization --------------------------------------------------------


def test_noise_changes_priors_but_never_the_legal_support():
    game = Connect4(5, 5, 4)
    off = _search(game, noise_seed=None)
    on = _search(game, noise_seed=7)

    assert list(on.root.actions) == list(off.root.actions)  # same action set, same order
    assert on.root.P != off.root.P
    assert all(p > 0.0 for p in on.root.P)
    assert abs(sum(on.root.P) - 1.0) < 1e-12


def test_eps_zero_and_eps_one_are_the_convex_combination_endpoints():
    game = TicTacToe()
    at_zero = MCTS(game, root_noise=(0.0, ALPHA_NUMERATOR, random.Random(1)))
    at_zero.run(20, root_state=game.initial_state())
    assert at_zero.root.P == [1.0 / 9] * 9  # (1-0)·P + 0·Dir == P

    at_one = MCTS(game, root_noise=(1.0, ALPHA_NUMERATOR, random.Random(1)))
    at_one.run(20, root_state=game.initial_state())
    assert at_one.root.P != [1.0 / 9] * 9  # pure Dirichlet, not the priors
    assert all(p > 0.0 for p in at_one.root.P)
    assert abs(sum(at_one.root.P) - 1.0) < 1e-12


def test_mixture_matches_the_d7_formula_on_a_known_dirichlet_draw():
    # Reproduce the draw the search consumes from an identically seeded rng and check the
    # engine's arithmetic edge-for-edge: P' = (1-eps)·P + eps·Dir(alpha).
    game = TicTacToe()
    m = _search(game, noise_seed=99)

    n = len(m.root.actions)
    alpha = ALPHA_NUMERATOR / n
    replay = random.Random(99)
    draws = [replay.gammavariate(alpha, 1.0) for _ in range(n)]
    total = sum(draws)
    expected = [(1.0 - EPS) * (1.0 / n) + EPS * (d / total) for d in draws]
    assert all(abs(a - b) < 1e-12 for a, b in zip(m.root.P, expected, strict=True))


# --- root-only ------------------------------------------------------------------------


def test_noise_is_root_only():
    # Every expanded non-root node keeps the untouched uniform priors (D7 is root-only).
    game = TicTacToe()
    m = _search(game, noise_seed=3, sims=200)
    seen = 0
    for child in m.root.children:
        if child is None or not child.is_expanded:
            continue
        seen += 1
        assert child.P == [1.0 / len(child.actions)] * len(child.actions)
    assert seen > 0  # the search really did expand children


# --- one draw per root incarnation ----------------------------------------------------


def test_noise_is_not_re_mixed_across_run_calls_on_the_same_root():
    game = TicTacToe()
    m = _search(game, noise_seed=11, sims=30)
    snapshot = list(m.root.P)
    m.run(30)  # incremental search from the same root
    assert m.root.P == snapshot


def test_reused_subtree_promoted_to_root_is_re_noised():
    # The M3 8.1 acceptance case: a child expanded during the previous search carries clean
    # priors; promoting it to root must draw fresh noise on the next search.
    game = Connect4(5, 5, 4)
    rng = random.Random(2024)
    m = MCTS(game, root_noise=(EPS, ALPHA_NUMERATOR, rng))
    m.run(200, root_state=game.initial_state())

    action = m.best_action()
    child = m.root.children[m.root.actions.index(action)]
    assert child is not None and child.is_expanded
    m.advance(action)
    assert m.root is child  # the subtree object itself is reused
    clean = list(m.root.P)
    assert clean == [1.0 / len(clean)] * len(clean)  # not yet noised

    m.run(50)
    assert m.root.P != clean
    assert all(p > 0.0 for p in m.root.P)
    assert abs(sum(m.root.P) - 1.0) < 1e-12
    assert list(m.root.actions) == list(_priors(m))  # support unchanged


def test_fresh_set_root_is_re_noised():
    game = TicTacToe()
    m = MCTS(game, root_noise=(EPS, ALPHA_NUMERATOR, random.Random(5)))
    m.run(20, root_state=game.initial_state())
    first = list(m.root.P)
    m.run(20, root_state=game.initial_state())  # re-roots on the same state
    assert m.root.P != first  # a new incarnation draws again


def test_advance_on_an_unexpanded_root_still_re_noises():
    game = TicTacToe()
    m = MCTS(game, root_noise=(EPS, ALPHA_NUMERATOR, random.Random(6)))
    m.set_root(game.initial_state())
    m.advance(4)  # never searched: the child is materialized fresh
    m.run(20)
    assert m.root.P != [1.0 / len(m.root.actions)] * len(m.root.actions)


# --- seeded reproducibility -----------------------------------------------------------


def test_same_seed_gives_identical_mixed_priors():
    game = Connect4(5, 5, 4)
    assert _search(game, noise_seed=42).root.P == _search(game, noise_seed=42).root.P


def test_different_seeds_give_different_mixed_priors():
    game = Connect4(5, 5, 4)
    assert _search(game, noise_seed=42).root.P != _search(game, noise_seed=43).root.P


# --- argument validation --------------------------------------------------------------


@pytest.mark.parametrize("eps", [-0.001, 1.5])
def test_eps_outside_the_unit_interval_is_rejected(eps):
    with pytest.raises(ValueError):
        MCTS(TicTacToe(), root_noise=(eps, ALPHA_NUMERATOR, random.Random(0)))


@pytest.mark.parametrize("numerator", [0.0, -1.0])
def test_non_positive_alpha_numerator_is_rejected(numerator):
    with pytest.raises(ValueError):
        MCTS(TicTacToe(), root_noise=(EPS, numerator, random.Random(0)))


# --- the Dirichlet sampler itself -----------------------------------------------------


def test_dirichlet_sample_is_a_normalized_simplex_point():
    rng = random.Random(1234)
    sample = MCTS._dirichlet(828, ALPHA_NUMERATOR / 828, rng)  # the Blokus opening width
    assert len(sample) == 828
    assert all(p >= 0.0 for p in sample)
    assert abs(sum(sample) - 1.0) < 1e-9


def test_dirichlet_falls_back_to_uniform_when_every_draw_underflows():
    # alpha ≈ 0.013 at an 828-wide root: an all-zero gamma vector is representable in
    # principle, and must not divide by zero.
    class _Zero:
        def gammavariate(self, alpha, beta):
            return 0.0

    assert MCTS._dirichlet(4, 0.013, _Zero()) == [0.25] * 4
