"""Golden for ``scripts/verify_pairs_per_cell.py`` — the §9 pin-1 precision basis.

Closes the loop between the numbers the design doc's pin 1 states, the standalone
script that reproduces them, and the pin-6 fit convention ``core/elo.py`` runs.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from core.elo import fit_elo

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_pairs_per_cell.py"


@pytest.fixture(scope="module")
def vp():
    """Import the standalone script as a module."""
    spec = importlib.util.spec_from_file_location("verify_pairs_per_cell", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_budget_arithmetic_matches_pin_1(vp):
    """Pin 1's 21,600-game bound and the exact rung-8 schedule's 21,024 games."""
    b = vp.budget()
    assert b["rung_cells"] == 30 * 3 * 4 == 360
    assert b["rung8_cells"] == 78
    assert b["cells_exact"] == 438
    assert b["games_exact"] == 21_024
    assert b["cells_bound"] == 450
    assert b["games_bound"] == 21_600


def test_rung8_rule_examples(vp):
    """Pin 5: {v−1, v−8, 1} ∩ [1, v−1], deduplicated, at most 3 cells."""
    assert vp.rung8_opponents(1) == ()
    assert vp.rung8_opponents(2) == (1,)
    assert vp.rung8_opponents(9) == (1, 8)
    assert vp.rung8_opponents(30) == (1, 22, 29)
    assert max(len(vp.rung8_opponents(v)) for v in range(1, 31)) == 3


def test_form7_cell_counts(vp):
    """The rung-7 form's own evidence: 4 rung cells plus ≤3 rung-8 cells."""
    cells = {name: len(opps) for name, _, opps in vp.SCENARIOS}
    assert cells == {"early": 4, "mid": 7, "late": 7}
    assert 24 * min(cells.values()) == 96 and 24 * max(cells.values()) == 168


def test_analytic_delta_half_width_band(vp):
    """Pin 1's Δ 95% half-width: ≈27 Elo at 24 pairs, ≈39 at 12, ≈19 at 48."""
    hw = {
        pairs: 1.96
        * math.sqrt(
            vp.analytic_se(100.0, vp.SCENARIOS[0][2], 2 * pairs) ** 2 / vp.GROUP
            + vp.analytic_se(900.0, vp.SCENARIOS[2][2], 2 * pairs) ** 2 / vp.GROUP
        )
        for pairs in (12, 24, 48)
    }
    assert 26.0 < hw[24] < 29.0
    assert 38.0 < hw[12] < 40.0
    assert 18.5 < hw[48] < 20.5
    # 1/√pairs scaling, exactly, for the closed-form route.
    assert hw[12] / hw[48] == pytest.approx(2.0, rel=1e-9)


def test_analytic_per_checkpoint_band_at_24_pairs(vp):
    """Pin 1's per-checkpoint SE band at 24 pairs: ≈22–35 Elo."""
    ses = [vp.analytic_se(true, opps, 48) for _, true, opps in vp.SCENARIOS]
    assert 21.0 < min(ses) < 23.0
    assert 34.0 < max(ses) < 36.0


def test_monte_carlo_agrees_with_analytic(vp):
    """The two independent routes agree at 24 pairs (reduced reps for speed)."""
    data = vp.compute(reps=600, seed=7)
    for name, s in data["grid"]["24"]["scenarios"].items():
        assert abs(s["mc_sd"] - s["analytic_se"]) < 0.2 * s["analytic_se"], name


def test_fit_step_matches_core_elo(vp):
    """The re-derived step agrees with ``core.elo.fit_elo`` on a one-agent fit."""
    score_a, n = 30.0, 48  # anchor took 30 of 48 points off the candidate
    ratings = fit_elo([("anchor", "cand", score_a, n)], anchor="anchor")
    # Script side: the candidate scored n − score_a; fold in the virtual draw.
    ours = vp.fit_rating([(0.0, (n - score_a) + 0.5, n + 1)])
    assert ours == pytest.approx(ratings["cand"], abs=1e-6)


def test_output_is_deterministic(vp):
    """Same seed, same numbers — the script must be byte-identical across runs."""
    assert vp.compute(reps=150, seed=3) == vp.compute(reps=150, seed=3)
