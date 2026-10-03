#!/usr/bin/env python3
"""Independently check the precision basis of the 24-pairs-per-cell pin (§9 pin 1).

Deliberately standalone — pure stdlib, no ``core/`` import — so agreement with the
protocol is evidence, not a shared bug. It re-derives the one-agent step of the
pin-6 fit (bracket expansion + bisection with one virtual draw per matchup) and
checks two independent routes against each other: the closed-form
Fisher-information standard error, and a seeded Monte-Carlo through that fit
step. Both rate one candidate's rung-7 form against fixed-rating opponents under
three representative scenarios (an early, a mid-run, and a late member
checkpoint) and three pairs-per-cell values (12, 24, 48). The script also
reproduces the §9 budget arithmetic: the 21,600-game bound and the exact rung-8
schedule's 21,024 games.

Idealizations, stated so the numbers are read correctly: games are independent
Bernoulli draws (no draws, no within-pair correlation); opponents sit at their
true ratings (the large-sample limit for rungs 1–4, which every form of every
member rates); and the §1 Δ standard error treats the ⌈K/3⌉ checkpoints of each
contrast group as independent. The paired bootstrap of §1 is the authoritative
CI; it can only be wider than these numbers.

Run it twice: the output is deterministic and must be byte-identical.

Usage::

    python3 scripts/verify_pairs_per_cell.py          # human-readable report
    python3 scripts/verify_pairs_per_cell.py --json   # machine-readable
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence

# --- the pinned protocol constants this script checks (design doc §6.2, §9) ------

K = 30  # checkpoint count (§6.2)
GROUP = -(-K // 3)  # ⌈K/3⌉ checkpoints per §1 contrast group
PAIRS_GRID = (12, 24, 48)
PINNED_PAIRS = 24  # §9 pin 1
RUNG8_LAG_DIVISOR = 4  # §9 pin 5: u ∈ {v−1, v−⌈K/4⌉, 1}
RUNG8_EARLIEST = 1
N_RUNGS = 4  # frozen network-free rungs 1–4
N_FORMS = 3  # forms 5/6/7
MAX_RUNG8_CELLS = 3
REPS = 4000
SEED = 1
Z_95 = 1.96
ELO_PER_NAT = 400.0 / math.log(10.0)

#: Representative true ratings of frozen rungs 1–4 (rung 1 is the Elo-0 anchor).
RUNG_ELOS = (0.0, 120.0, 250.0, 380.0)

#: (name, candidate's true Elo, opponent true Elos) — the rung-7 form's own cells:
#: the four rungs, plus the rung-8 opponents {v−1, v−8, member 1} where they exist.
SCENARIOS: tuple[tuple[str, float, tuple[float, ...]], ...] = (
    ("early", 100.0, RUNG_ELOS),
    ("mid", 400.0, RUNG_ELOS + (370.0, 250.0, 100.0)),
    ("late", 900.0, RUNG_ELOS + (880.0, 700.0, 100.0)),
)

IDEALIZATIONS = (
    "games are independent Bernoulli draws (no draws, no within-pair correlation)",
    "opponents sit at their true ratings (large-sample limit for rungs 1-4)",
    "delta SE treats the ceil(K/3) checkpoints of each contrast group as independent",
)


# --- the pin-6 fit step, re-derived --------------------------------------------


def expected_score(delta: float) -> float:
    """Return the logistic expected score at Elo difference ``delta``.

    Args:
        delta: Candidate rating minus opponent rating, in Elo points.

    Returns:
        ``1 / (1 + 10^(-delta/400))``.
    """
    return 1.0 / (1.0 + 10.0 ** (-delta / 400.0))


def fit_rating(cells: Sequence[tuple[float, float, int]]) -> float:
    """Fit one agent's rating against fixed-rating opponents (pin-6 step).

    Coordinate-ascent step of the anchored Bradley–Terry fit: bracket
    expansion in 800-point steps, then 80 bisection steps, equating the total
    expected score with the total actual score. The caller folds in the virtual
    draw (score +0.5, games +1 per matchup), exactly as the protocol's fit does.

    Args:
        cells: ``(opponent_elo, score, n_games)`` per matchup, virtual draw
            included.

    Returns:
        The fitted rating, in Elo points.
    """
    actual = sum(score for _, score, _ in cells)

    def expected_at(r: float) -> float:
        return sum(n * expected_score(r - opp) for opp, _, n in cells)

    lo, hi = -800.0, 800.0
    while expected_at(lo) > actual:
        lo -= 800.0
    while expected_at(hi) < actual:
        hi += 800.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if expected_at(mid) < actual:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# --- the two independent precision routes --------------------------------------


def analytic_se(true_elo: float, opponents: Sequence[float], games_per_cell: int) -> float:
    """Return the Fisher-information standard error of the fitted rating.

    A cell of ``n`` games at expected score ``p`` carries information
    ``n·p(1−p)·(ln 10/400)²`` about the rating; the SE is the inverse root of
    the total over the candidate's cells.

    Args:
        true_elo: The candidate's true rating.
        opponents: True ratings of the opponents, one per cell.
        games_per_cell: Games per cell (twice the pairs).

    Returns:
        The standard error, in Elo points.
    """
    info = 0.0
    for opp in opponents:
        p = expected_score(true_elo - opp)
        info += games_per_cell * p * (1.0 - p)
    return ELO_PER_NAT / math.sqrt(info)


def monte_carlo(
    true_elo: float,
    opponents: Sequence[float],
    pairs: int,
    reps: int,
    rng: random.Random,
) -> dict[str, float]:
    """Simulate ``reps`` evaluations and fit each through the pin-6 step.

    Args:
        true_elo: The candidate's true rating.
        opponents: True ratings of the opponents, one per cell.
        pairs: Mirrored pairs per cell (games per cell is twice this).
        reps: Number of simulated evaluations.
        rng: The seeded random stream.

    Returns:
        ``{"sd", "mean", "p2_5", "p97_5", "half_width_95"}`` of the fitted
        ratings, in Elo points.
    """
    n = 2 * pairs
    fits: list[float] = []
    for _ in range(reps):
        cells = []
        for opp in opponents:
            p = expected_score(true_elo - opp)
            wins = sum(1 for _ in range(n) if rng.random() < p)
            cells.append((opp, wins + 0.5, n + 1))  # one virtual draw per matchup
        fits.append(fit_rating(cells))
    fits.sort()
    mean = sum(fits) / reps
    sd = math.sqrt(sum((f - mean) ** 2 for f in fits) / reps)
    lo, hi = fits[int(0.025 * reps)], fits[int(0.975 * reps)]
    return {"sd": sd, "mean": mean, "p2_5": lo, "p97_5": hi, "half_width_95": (hi - lo) / 2}


# --- the §9 budget arithmetic ---------------------------------------------------


def rung8_opponents(v: int, k: int = K) -> tuple[int, ...]:
    """Return member ``v``'s rung-8 opponents under §9 pin 5.

    Args:
        v: The member version (1-based).
        k: The checkpoint count.

    Returns:
        ``{v−1, v−⌈k/4⌉, 1} ∩ [1, v−1]``, deduplicated and sorted.
    """
    lag = -(-k // RUNG8_LAG_DIVISOR)
    return tuple(sorted({u for u in (v - 1, v - lag, RUNG8_EARLIEST) if 1 <= u <= v - 1}))


def budget(pairs: int = PINNED_PAIRS, k: int = K) -> dict[str, int]:
    """Return the per-run cell and game counts, exact and bounded.

    Args:
        pairs: Mirrored pairs per cell.
        k: The checkpoint count.

    Returns:
        Rung cells, exact rung-8 cells, exact and bounded cell/game totals.
    """
    rung_cells = k * N_FORMS * N_RUNGS
    rung8_cells = sum(len(rung8_opponents(v, k)) for v in range(1, k + 1))
    cells_exact = rung_cells + rung8_cells
    cells_bound = k * (N_FORMS * N_RUNGS + MAX_RUNG8_CELLS)
    return {
        "rung_cells": rung_cells,
        "rung8_cells": rung8_cells,
        "cells_exact": cells_exact,
        "games_exact": cells_exact * 2 * pairs,
        "cells_bound": cells_bound,
        "games_bound": cells_bound * 2 * pairs,
    }


# --- driver -----------------------------------------------------------------------


def compute(reps: int = REPS, seed: int = SEED) -> dict:
    """Compute every number the report and the §9 pin-1 text cite.

    Args:
        reps: Monte-Carlo replicates per (scenario, pairs) cell.
        seed: Seed of the single random stream (deterministic output).

    Returns:
        A JSON-serializable dict: idealizations, budget arithmetic, scenarios,
        and per-pairs analytic/Monte-Carlo precision plus the §1 Δ half-width.
    """
    rng = random.Random(seed)
    grid: dict[str, dict] = {}
    for pairs in PAIRS_GRID:
        per_scenario: dict[str, dict[str, float]] = {}
        for name, true_elo, opponents in SCENARIOS:
            mc = monte_carlo(true_elo, opponents, pairs, reps, rng)
            per_scenario[name] = {
                "analytic_se": analytic_se(true_elo, opponents, 2 * pairs),
                "mc_sd": mc["sd"],
                "mc_mean": mc["mean"],
                "mc_p2_5": mc["p2_5"],
                "mc_p97_5": mc["p97_5"],
                "mc_half_width_95": mc["half_width_95"],
            }
        se_early = per_scenario["early"]["analytic_se"]
        se_late = per_scenario["late"]["analytic_se"]
        delta_se = math.sqrt(se_early**2 / GROUP + se_late**2 / GROUP)
        grid[str(pairs)] = {
            "scenarios": per_scenario,
            "delta_se": delta_se,
            "delta_half_width_95": Z_95 * delta_se,
        }
    return {
        "idealizations": list(IDEALIZATIONS),
        "k": K,
        "group": GROUP,
        "pinned_pairs": PINNED_PAIRS,
        "reps": reps,
        "seed": seed,
        "budget": budget(),
        "scenarios": {
            name: {"true_elo": true_elo, "opponents": list(opps), "pairs_form7": len(opps)}
            for name, true_elo, opps in SCENARIOS
        },
        "grid": grid,
    }


def _report(data: dict) -> str:
    """Render ``compute()``'s output as the human-readable report."""
    b = data["budget"]
    lines = [
        "verify_pairs_per_cell — §9 pin 1 precision basis (idealized; the bootstrap CI rules)",
        f"K = {data['k']}, contrast group = ceil(K/3) = {data['group']}, reps = {data['reps']}",
        (
            f"budget @ {data['pinned_pairs']} pairs: {b['rung_cells']} rung cells + "
            f"{b['rung8_cells']} rung-8 cells = {b['cells_exact']} cells = "
            f"{b['games_exact']} games (bound {b['cells_bound']} cells = {b['games_bound']} games)"
        ),
    ]
    for pairs, entry in data["grid"].items():
        lines.append(f"\n### pairs/cell = {pairs}  (games/cell = {2 * int(pairs)})")
        for name, s in entry["scenarios"].items():
            sc = data["scenarios"][name]
            lines.append(
                f"{name:5s} true={sc['true_elo']:5.0f} cells={sc['pairs_form7']}  "
                f"analytic SE={s['analytic_se']:5.1f}  MC SD={s['mc_sd']:5.1f}  "
                f"MC mean={s['mc_mean']:6.1f}  95% band=[{s['mc_p2_5']:6.1f}, "
                f"{s['mc_p97_5']:6.1f}]  half-width={s['mc_half_width_95']:5.1f}"
            )
        lines.append(
            f"   => §1 Δ (mean of {data['group']} late − mean of {data['group']} early): "
            f"SE={entry['delta_se']:.1f}  95% half-width={entry['delta_half_width_95']:.1f} Elo"
        )
    return "\n".join(lines)


def main() -> None:
    """CLI entry: print the report, or JSON with ``--json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    data = compute()
    print(json.dumps(data, indent=2, sort_keys=True) if args.json else _report(data))


if __name__ == "__main__":
    main()
