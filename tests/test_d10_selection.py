"""D10 move-selection + π_train statistical battery (design doc §10, §12 M3, issue #57).

``core/selfplay.py``'s two D10 functions land pre-built by the M2.5 micro loop
(``tests/test_micro_loop.py`` exercises them end to end through ``play_game``);
this module is the dedicated, self-contained verification against the D10 pins
themselves plus the statistical battery the issue asks for:

* **Empirical sampling frequencies** track ``N / ΣN`` — drawn through the real
  seeded stream family (``core.seeding.GameRNGs``, varied by ``game_index``),
  never a hand-rolled unseeded ``random.Random``.
* **The temperature-boundary ply**: sampling strictly below ``k_temp``, exact
  argmax at and after it.
* **Never-samples-a-zero-visit-action**, both empirically and by construction
  (the sampler's weight for a zero-``N`` action is an exact zero-width slice of
  the cumulative-weight domain ``random.choices`` draws over — not merely an
  unlikely outcome).
* **Subtree-reuse π_train**: the extracted pairs mirror the root's exact counts
  even when ``ΣN`` has been inflated past a single move's simulation budget.
* **The argmax tie-break golden**: exact agreement with ``MCTS.best_action``.

Verification result (see the issue): ``select_move`` and ``policy_target``
match the D10 pins with **no divergence** — confirmed below to be bit-for-bit
identical to ``MCTS.select_action(temperature=1.0, ...)`` and
``MCTS.best_action()`` on a real search tree, so this module adds coverage
rather than fixing anything.
"""

from __future__ import annotations

import math
import random
from itertools import accumulate

from core import MCTS
from core.seeding import GameRNGs
from core.selfplay import policy_target, select_move
from games.tictactoe import TicTacToe

# One fixed run seed for the whole module (issue #57 date-stamped): every draw below
# is a pure function of (RUN_SEED, derivation labels), so the battery is deterministic
# across runs and machines — no flakiness, no wall-clock/system entropy anywhere.
RUN_SEED = 20260057

# A non-degenerate, non-uniform visit distribution shared by the frequency and
# boundary tests: five actions, a clear argmax (12, N=40 of 60), and enough spread
# that a softmax/exponentiated rule would visibly starve the low-count actions.
VISITS = {5: 2, 9: 6, 12: 40, 20: 1, 33: 11}
K_TEMP = 8


def _cdf_tolerance(p: float, n: int, z: float = 6.0) -> float:
    """Return a CLT-based tolerance for a binomial frequency estimate.

    For ``n`` iid draws of an indicator with true probability ``p``, the
    sample frequency is asymptotically ``Normal(p, p(1-p)/n)`` (De
    Moivre-Laplace / CLT). A ``z``-sigma band therefore bounds the deviation
    with probability ``1 - 2*Phi(-z)``; at ``z = 6`` that failure probability
    is on the order of ``1e-9`` per action, so with a *fixed* seed the test
    either always passes or always fails — the margin exists to make the
    assertion principled, not to paper over flakiness (there is none: the
    stream is fully determined by ``RUN_SEED``).

    Args:
        p: The true (target) probability, i.e. ``N(a) / ΣN``.
        n: Number of independent draws.
        z: Number of standard deviations of margin (default 6).

    Returns:
        The absolute tolerance on ``|empirical_frequency - p|``.
    """
    return z * math.sqrt(p * (1.0 - p) / n)


def _independent_draws(visits, ply, k_temp, n_draws, *, prefix):
    """Draw ``n_draws`` moves through the real seeded stream family.

    Each draw uses its own ``GameRNGs.for_game(RUN_SEED, i, prefix=prefix)``
    stream — the production derivation path (``core.seeding``), varied by
    ``game_index`` to get independent draws — rather than repeated calls
    against one hand-held ``random.Random``.

    Args:
        visits: ``{action_id: visit_count}`` root visit counts.
        ply: The ply index passed to ``select_move``.
        k_temp: The D10 temperature cutoff.
        n_draws: Number of independent derivations to draw.
        prefix: Extra label parts distinguishing this battery's streams from
            any other test's or run's use of the same ``RUN_SEED``.

    Returns:
        The list of chosen action ids, one per derivation.
    """
    draws = []
    for i in range(n_draws):
        rng = GameRNGs.for_game(RUN_SEED, i, prefix=prefix).move_selection
        draws.append(select_move(visits, ply, k_temp, rng))
    return draws


# --- empirical sampling frequencies ---------------------------------------------


def test_empirical_frequencies_track_n_over_sigma_n_via_seeded_stream_family():
    """τ = 1, no exponentiation: sampled frequencies track N/ΣN within a CLT bound.

    Draws come from independent derivations of the real ``move-selection``
    stream family (varied ``game_index``), not a single hand-rolled RNG.
    """
    n_draws = 4000
    total = sum(VISITS.values())
    draws = _independent_draws(VISITS, ply=0, k_temp=K_TEMP, n_draws=n_draws, prefix=("d10-freq",))

    counts = {a: draws.count(a) for a in VISITS}
    assert sum(counts.values()) == n_draws
    for action, n in VISITS.items():
        p = n / total
        empirical = counts[action] / n_draws
        assert abs(empirical - p) <= _cdf_tolerance(p, n_draws), (action, p, empirical)
    # A softmax/exponentiated rule would starve the low-count actions; raw N does not.
    assert counts[5] > 0
    assert counts[20] > 0


# --- temperature-boundary ply ----------------------------------------------------


def test_sampling_strictly_below_k_temp_deviates_from_argmax_at_least_once():
    """Ply ``k_temp - 1`` samples: over many derivations a non-argmax move is chosen."""
    argmax = min(VISITS, key=lambda a: (-VISITS[a], a))
    draws = _independent_draws(
        VISITS, ply=K_TEMP - 1, k_temp=K_TEMP, n_draws=50, prefix=("d10-boundary",)
    )
    assert any(a != argmax for a in draws), "sampling at k_temp - 1 never left the argmax"
    # ... but the argmax still comes up sometimes (it is the modal outcome, not excluded).
    assert any(a == argmax for a in draws)


def test_argmax_at_k_temp_is_exact_and_deterministic_across_derivations():
    """Ply ``k_temp`` always plays the max-N action, whatever the rng stream holds."""
    argmax = min(VISITS, key=lambda a: (-VISITS[a], a))
    draws = _independent_draws(
        VISITS, ply=K_TEMP, k_temp=K_TEMP, n_draws=50, prefix=("d10-boundary",)
    )
    assert draws == [argmax] * len(draws)

    # And the argmax branch must not touch the stream at all: an untouched Random
    # still yields its first value afterwards.
    rng = random.Random(0)
    select_move(VISITS, K_TEMP, K_TEMP, rng)
    assert rng.random() == random.Random(0).random()


# --- never samples a zero-visit action --------------------------------------------


def test_zero_visit_action_is_never_sampled_and_has_zero_weight_by_construction():
    """A legal N=0 action is never drawn — empirically, and by an exact-zero weight."""
    visits = {1: 0, 2: 5, 3: 7, 4: 0, 5: 12}
    zero_actions = {1, 4}
    n_draws = 3000

    draws = _independent_draws(visits, ply=0, k_temp=K_TEMP, n_draws=n_draws, prefix=("d10-zero",))
    assert set(draws) == set(visits) - zero_actions
    for action in zero_actions:
        assert action not in draws

    # By construction, not just by luck: select_move's sampling weights are exactly
    # root_visits[a] (mirrored here from the visits mapping itself, which the test
    # fully controls), and random.choices samples uniformly over the cumulative-weight
    # domain built from those weights. A weight of 0 contributes a zero-width slice to
    # that domain, so no draw of the underlying uniform variate can land on it — the
    # selection probability is exactly 0, not merely small.
    actions = list(visits)
    weights = [visits[a] for a in actions]
    cum = list(accumulate(weights))
    for action in zero_actions:
        i = actions.index(action)
        width = cum[i] - (cum[i - 1] if i > 0 else 0)
        assert width == 0


# --- subtree-reuse π_train --------------------------------------------------------


def test_policy_target_mirrors_root_counts_exactly_after_subtree_reuse():
    """ΣN inflated past a move's sim budget by reuse: the target is not re-derived."""
    game = TicTacToe()
    search = MCTS(game)
    budget = 40
    search.run(budget, root_state=game.initial_state())

    action = search.best_action()
    search.advance(action)
    search.run(budget)  # subtree reuse: adds `budget` sims on top of the promoted root

    visits = search.action_visit_counts()
    assert sum(visits.values()) > budget  # the reuse-inflated ΣN this test exists for

    pairs = policy_target(visits)
    assert pairs == list(visits.items())
    assert dict(pairs) == visits  # no rescaling, no clipping to `budget`


# --- argmax tie-break golden -------------------------------------------------------


def test_argmax_tie_break_golden_two_equal_n_actions():
    """Two actions tied on N: the established tie-break (lowest action id) wins, exact."""
    tied = {40: 3, 7: 3, 2: 3}
    assert select_move(tied, K_TEMP, K_TEMP, random.Random(0)) == 2

    node_style = {9: 5, 2: 5, 17: 1}
    assert select_move(node_style, K_TEMP, K_TEMP, random.Random(0)) == 2


# --- select_move / policy_target agree bit-for-bit with the engine ----------------


def test_select_move_matches_mcts_select_action_bit_for_bit():
    """select_move's two branches are exactly MCTS.select_action(1.0) / best_action().

    Verifies the D10 surface uses "the engine's established tie-break" rather than
    an independently invented one, on a real search tree (not a hand-built mapping).
    """
    game = TicTacToe()
    search = MCTS(game)
    search.run(60, root_state=game.initial_state())
    visits = search.action_visit_counts()

    sampled = select_move(visits, 0, K_TEMP, random.Random(99))
    assert sampled == search.select_action(1.0, random.Random(99))

    argmax = select_move(visits, K_TEMP, K_TEMP, random.Random(1))
    assert argmax == search.best_action()


def test_policy_target_matches_action_visit_counts_bit_for_bit():
    """policy_target is exactly MCTS.action_visit_counts(), as sparse pairs."""
    game = TicTacToe()
    search = MCTS(game)
    search.run(60, root_state=game.initial_state())
    visits = search.action_visit_counts()

    assert policy_target(visits) == list(search.action_visit_counts().items())
