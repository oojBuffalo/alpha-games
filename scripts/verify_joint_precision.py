#!/usr/bin/env python3
"""Independent full-schedule expected-data BT precision/bias diagnostic.

Uses NumPy, not core.elo. All three forms assume the same linear true trajectory;
rungs assume (0,120,250,380). Games are independent Bernoulli draws. Fisher
half-widths are local Gaussian approximations, not paired-bootstrap intervals.
Expected-data regularized fits isolate shrinkage, not finite-sample MC bias.
"""

from __future__ import annotations

import json
import math

import numpy as np
from verify_pairs_per_cell import GROUP, RUNG_ELOS, K, rung8_opponents


def diagnostic(pairs: int, final_elo: float) -> dict[str, float | int]:
    """Solve joint expected-data fits and Fisher variance on all 438 cells."""
    names = [f"r{r}" for r in range(4)] + [
        f"f{form}v{v}" for form in (5, 6, 7) for v in range(1, K + 1)
    ]
    index = {name: i for i, name in enumerate(names)}
    trajectory = np.linspace(100, final_elo, K)
    truth = np.array([*RUNG_ELOS, *trajectory, *trajectory, *trajectory])
    cells = [
        (f"f{form}v{v}", f"r{r}") for form in (5, 6, 7) for v in range(1, K + 1) for r in range(4)
    ]
    cells += [(f"f7v{v}", f"f7v{u}") for v in range(1, K + 1) for u in rung8_opponents(v)]
    design = np.zeros((len(cells), len(names)))
    for row, (a, b) in enumerate(cells):
        design[row, index[a]] = 1
        design[row, index[b]] = -1
    design = design[:, 1:]  # rung 1 is fixed at zero
    scale = 400 / math.log(10)
    true_logits = truth[1:] / scale
    p = 1 / (1 + np.exp(-(design @ true_logits)))
    games = 2 * pairs
    contrast = np.zeros(len(names) - 1)
    for v in range(1, GROUP + 1):
        contrast[index[f"f7v{v}"] - 1] = -1 / GROUP
    for v in range(K - GROUP + 1, K + 1):
        contrast[index[f"f7v{v}"] - 1] = 1 / GROUP
    information = design.T @ ((games * p * (1 - p))[:, None] * design)
    half_width = 1.96 * scale * math.sqrt(contrast @ np.linalg.solve(information, contrast))
    # Every cell is a unique unordered matchup: one virtual draw per row.
    scores = games * p + 0.5
    logits = true_logits.copy()
    for _ in range(100):
        fitted_p = 1 / (1 + np.exp(-(design @ logits)))
        gradient = design.T @ (scores - (games + 1) * fitted_p)
        hessian = design.T @ (((games + 1) * fitted_p * (1 - fitted_p))[:, None] * design)
        step = np.linalg.solve(hessian, gradient)

        def objective(candidate):
            differences = design @ candidate
            return float(scores @ differences - (games + 1) * np.logaddexp(0, differences).sum())

        fraction = 1.0
        while objective(logits + fraction * step) < objective(logits) - 1e-9:
            fraction *= 0.5
            if fraction < 1e-12:
                raise RuntimeError("joint fit line search failed")
        logits += fraction * step
        if np.max(np.abs(step)) < 1e-10:
            break
    else:
        raise RuntimeError("expected-data fit did not converge")
    delta_true = float(contrast @ truth[1:])
    delta_regularized = float(scale * (contrast @ logits))
    return {
        "cells": len(cells),
        "pairs": pairs,
        "final_elo": final_elo,
        "delta_true": delta_true,
        "delta_regularized": delta_regularized,
        "shrinkage": delta_regularized - delta_true,
        "fisher_half_width_95": half_width,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            [diagnostic(pairs, final) for pairs in (12, 24, 48) for final in (900, 1300, 1800)],
            indent=2,
            sort_keys=True,
        )
    )
