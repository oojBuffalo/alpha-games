"""D1 value-targets golden: the thin score→z→aux mapping behind every training target.

``z = sign(score_diff)`` including ``sign(0) = 0`` (draws are not losses; §4 has
``z = 0``), ``aux = score_diff / 109``, and the ``|score_diff| <= 109`` range
assertion. Also the declared aux-loss weight: λ_aux was pinned doc-first at M2
(§7), so the spec golden is a hardcoded equality — code drifting from the doc
fails here.
"""

from __future__ import annotations

import pytest

from games.blokus_duo.targets import value_target_spec, value_targets


def test_z_is_sign_of_score_diff():
    assert value_targets(20, -89) == (1, pytest.approx(109 / 109))
    assert value_targets(-89, 20) == (-1, pytest.approx(-109 / 109))
    assert value_targets(-5, -10) == (1, pytest.approx(5 / 109))
    assert value_targets(-10, -5) == (-1, pytest.approx(-5 / 109))


def test_draws_are_zero_not_losses():
    z, aux = value_targets(-7, -7)
    assert z == 0
    assert aux == 0.0


def test_aux_is_normalized_score_diff():
    z, aux = value_targets(15, -30)
    assert aux == pytest.approx(45 / 109)


def test_range_assertion_on_impossible_diffs():
    with pytest.raises(ValueError):
        value_targets(21, -89)  # diff 110 > 109
    with pytest.raises(ValueError):
        value_targets(-90, 20)


def test_value_target_spec_declares_pinned_lambda_aux():
    # λ_aux = 0.25, pinned doc-first in design doc §7 — hardcoded here so code
    # drifting from the doc fails loudly.
    spec = value_target_spec()
    assert spec.primary_name == "z"
    assert spec.aux_names == ("score_diff",)
    assert spec.aux_loss_weights == (0.25,)
