"""D1 value targets: ``z = sign(score_diff)``, aux ``score_diff/max|diff|`` (§4, §10).

Blokus-local by design [F7, revised at M2.5 task 4]: the *mapping* still lives
in the adapter package — core never learns what a score difference is. What
changed is that the ABC now carries a generic ``training_targets(state,
player_id) -> (z, aux)`` surface (§6.1's declared-adapter pattern, alongside
``value_targets`` and ``symmetry_group``) that a game-agnostic self-play loop
calls to materialize both targets without game imports; :class:`BlokusDuo`
overrides it by delegating straight to :func:`value_targets` below. The
original note's intent — "no core additions for one game's auxiliary head" —
holds: the addition is a declared surface, not Blokus's head in core.

The aux divisor is a *derived* per-instance bound, not a constant: the full game
pins it at 109 (a player at −89 against one at +20) and the §5.3 micro instance
at 29 (−9 against +20). :func:`max_score_diff` computes it from the config's
piece set, and :data:`MAX_SCORE_DIFF` keeps the full game's value bound to the
name every M1/M2 caller imports.
"""

from __future__ import annotations

from functools import cache

from core.game import ValueTargetSpec
from games.blokus_duo.config import FULL_CONFIG, BlokusConfig
from games.blokus_duo.pieces import build_pieces

# Scoring bonuses (§4), shared by every instance: all pieces placed, and the
# extra for placing the monomino last.
ALL_PLACED_BONUS = 15
MONOMINO_LAST_BONUS = 5

# Aux-loss weight λ_aux, pinned doc-first at M2 (§7): keeps the score-diff MSE
# a minority of the value-side gradient. Config scalar, not a D-decision — and
# not instance-dependent: the same weight applies at every board size.
AUX_LOSS_WEIGHT = 0.25


@cache
def max_score_diff(config: BlokusConfig = FULL_CONFIG) -> int:
    """Return ``config``'s maximum ``|score difference|`` — the D1 aux divisor.

    The bound is ``best - worst``: the best per-player score is the all-placed
    bonus plus the monomino-last bonus (the latter only if the instance's piece
    set contains the monomino at all), and the worst is one −1 per square of the
    unplaced set. Full game: ``20 - (-89) = 109``; micro (§5.3): ``20 - (-9) =
    29``. Both bounds are pinned, not merely reachable.

    Args:
        config: The instance config; defaults to the full 14×14 game.

    Returns:
        The maximum representable ``|score_own - score_opp|``.
    """
    pieces = build_pieces(config)[0]
    best = ALL_PLACED_BONUS + (MONOMINO_LAST_BONUS if len(pieces[0]) == 1 else 0)
    return best + sum(len(p) for p in pieces)


# Maximum |score difference| for the full game: one player at −89, the other at
# +20 (§4).
MAX_SCORE_DIFF = max_score_diff(FULL_CONFIG)


def value_targets(
    score_own: int, score_opp: int, config: BlokusConfig = FULL_CONFIG
) -> tuple[int, float]:
    """Map a terminal score pair to the D1 training targets ``(z, aux)``.

    Args:
        score_own: The player's official score (full game: ``[-89, 20]``).
        score_opp: The opponent's official score (same range).
        config: The instance the scores come from; defaults to the full game.

    Returns:
        ``(z, aux)`` where ``z = sign(score_own - score_opp)`` (0 on draws) and
        ``aux = (score_own - score_opp) / max_score_diff(config)``.

    Raises:
        ValueError: If ``|score_own - score_opp|`` exceeds the config's bound —
            an impossible score pair; every training target flows through this
            check.
    """
    limit = max_score_diff(config)
    diff = score_own - score_opp
    if abs(diff) > limit:
        raise ValueError(f"|score_diff| = {abs(diff)} exceeds {limit} — invalid scores")
    z = (diff > 0) - (diff < 0)
    return z, diff / limit


def value_target_spec(config: BlokusConfig = FULL_CONFIG) -> ValueTargetSpec:
    """Return the declared value-target spec for a Blokus instance (D1, §6.1).

    The declared shape is instance-independent — one primary ``z`` plus one
    ``score_diff`` aux head at the §7-pinned weight — because the config only
    moves the *divisor* inside :func:`value_targets`, not the head layout. The
    parameter is accepted so adapters can pass their config uniformly.

    Args:
        config: The instance config; defaults to the full 14×14 game.

    Returns:
        Spec with primary ``z`` and one aux head ``score_diff`` weighted by
        ``AUX_LOSS_WEIGHT``.
    """
    del config  # declared shape is the same for every instance (see above)
    return ValueTargetSpec(
        primary_name="z",
        aux_names=("score_diff",),
        aux_loss_weights=(AUX_LOSS_WEIGHT,),
    )
