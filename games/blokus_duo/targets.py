"""D1 value targets: ``z = sign(score_diff)``, aux ``score_diff/109`` (design doc §4, §10).

Blokus-local by design [F7]: the mapping lives in the adapter package, not on
the ``Game`` ABC — no core additions for one game's auxiliary head. The aux
loss weight λ_aux is an unpinned scalar until M2 pins it doc-first, so the
declared weight is a NaN sentinel: any loss computed with it is poisoned
loudly instead of silently training the aux head at weight 0.0.
"""

from __future__ import annotations

from core.game import ValueTargetSpec

# Maximum |score difference|: one player at −89, the other at +20 (§4).
MAX_SCORE_DIFF = 109


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
        Spec with primary ``z`` and one aux head ``score_diff`` whose loss
        weight is the [F7] NaN sentinel until M2 pins λ_aux doc-first.
    """
    return ValueTargetSpec(
        primary_name="z",
        aux_names=("score_diff",),
        aux_loss_weights=(float("nan"),),
    )
