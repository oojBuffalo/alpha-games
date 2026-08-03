"""Training-side symmetry augmentation (M2, D9/§8).

The M2-owned side of the §12 M1/M2 symmetry ownership boundary: a game-generic
utility applying one declared group element to a ``(state planes, sparse π)``
training sample. Core hardcodes no group — it composes the two callables of an
adapter-declared ``symmetry_group`` element (§6.1: ``SymmetryElement =
(plane_transform, action_permutation)``); Blokus's Klein-4 and Othello's full
D4 flow through the same code path.

Pure stdlib over nested-tuple planes and the D12 sparse ``(action_id,
visit_count)`` pairs, so it tests without torch and works for any adapter;
tensor conversion happens later, at the collate boundary. Which ``g`` to draw
per sample (the D9 sampling strategy) belongs to the M3 training loop — this
module delivers only the transform.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from core.game import Action, Game


def augment_sample(
    game: Game,
    planes: Any,
    sparse_pi: Iterable[tuple[Action, int]],
    g_index: int,
) -> tuple[Any, list[tuple[Action, int]]]:
    """Apply symmetry element ``g_index`` to one ``(planes, sparse π)`` sample.

    Args:
        game: Adapter whose declared ``symmetry_group`` supplies the element.
        planes: ``encode_state`` output (nested-tuple plane tensor).
        sparse_pi: D12-shaped sparse policy target as ``(action_id,
            visit_count)`` pairs.
        g_index: Index into ``game.symmetry_group`` (0 is identity by the
            adapters' element-order pins).

    Returns:
        ``(transformed planes, permuted pairs)`` — visit counts ride along
        untouched: a permutation relabels actions, never redistributes mass.

    Raises:
        IndexError: If ``g_index`` is outside the declared group.
    """
    plane_transform, action_permutation = game.symmetry_group[g_index]
    return (
        plane_transform(planes),
        [(action_permutation[a], n) for a, n in sparse_pi],
    )
