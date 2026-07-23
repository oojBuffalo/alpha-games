"""D1 value targets: ``z = sign(score_diff)``, aux ``score_diff/109`` (design doc §4, §10).

Blokus-local by design [F7]: the mapping lives in the adapter package, not on
the ``Game`` ABC — no core additions for one game's auxiliary head.
"""

from __future__ import annotations

from core.game import ValueTargetSpec

# Maximum |score difference|: one player at −89, the other at +20 (§4).
MAX_SCORE_DIFF = 109

# Aux-loss weight λ_aux, pinned doc-first at M2 (§7): keeps the score-diff MSE
# a minority of the value-side gradient. Config scalar, not a D-decision.
AUX_LOSS_WEIGHT = 0.25


def value_targets(score_own: int, score_opp: int) -> tuple[int, float]:
    """Map a terminal score pair to the D1 training targets ``(z, aux)``.

    Args:
        score_own: The player's official score, in ``[-89, 20]``.
        score_opp: The opponent's official score, in ``[-89, 20]``.

    Returns:
        ``(z, aux)`` where ``z = sign(score_own - score_opp)`` (0 on draws) and
        ``aux = (score_own - score_opp) / 109``.

    Raises:
        ValueError: If ``|score_own - score_opp| > 109`` — an impossible score
            pair; every training target flows through this check.
    """
    diff = score_own - score_opp
    if abs(diff) > MAX_SCORE_DIFF:
        raise ValueError(f"|score_diff| = {abs(diff)} exceeds {MAX_SCORE_DIFF} — invalid scores")
    z = (diff > 0) - (diff < 0)
    return z, diff / MAX_SCORE_DIFF


def value_target_spec() -> ValueTargetSpec:
    """Return the declared value-target spec for Blokus Duo (D1, §6.1).

    Returns:
        Spec with primary ``z`` and one aux head ``score_diff`` weighted by
        the §7-pinned ``AUX_LOSS_WEIGHT``.
    """
    return ValueTargetSpec(
        primary_name="z",
        aux_names=("score_diff",),
        aux_loss_weights=(AUX_LOSS_WEIGHT,),
    )
