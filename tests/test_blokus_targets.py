"""D1 value-targets golden: the thin score→z→aux mapping behind every training target.

``z = sign(score_diff)`` including ``sign(0) = 0`` (draws are not losses; §4 has
``z = 0``), ``aux = score_diff / 109``, and the ``|score_diff| <= 109`` range
assertion. Also the [F7] NaN sentinel: the aux loss weight λ_aux is unpinned
until M2 pins it doc-first, so the declared weight must poison any loss that
uses it early — 0.0 would silently train the aux head at weight zero.
"""

from __future__ import annotations

import math

import pytest

from games.blokus_duo.targets import value_targets, value_target_spec


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


def test_value_target_spec_declares_nan_sentinel():
    # [F7] λ_aux is pinned doc-first at M2; until then the weight is NaN so any
    # loss computed with it is poisoned rather than silently trained at 0.0.
    spec = value_target_spec()
    assert spec.primary_name == "z"
    assert spec.aux_names == ("score_diff",)
    assert len(spec.aux_loss_weights) == 1
    assert math.isnan(spec.aux_loss_weights[0])  # NaN != NaN — assert via isnan
