"""Blokus pipeline decode test (M2, §12): plane transform ≡ action permutation.

The §12 M2-named integration check, quoting the spec: "for random ``(state,
sparse π)`` and each declared ``g``: build ``(g·state, g·π)`` via plane
transform + action permutation, decode every ``(action_id, count)`` back to a
cell set, and assert the multiset of ``(cell-set, count)`` equals ``g`` applied
to the original — catches plane/channel wiring mismatches invisible to M1's
table-level golden."

Both halves of an augmented sample are pushed back into board geometry through
one reference map (``symmetry.cell_map``): the permuted policy pairs decode via
``action_cells`` into a *multiset* of ``(cell-set, count)`` pairs — multiset,
not set, since distinct actions may carry equal counts — that must equal the
cell-mapped originals, and the transformed occupancy planes must equal the
cell-mapped original occupancy (guards the named M1 failure mode
``g(anchor) != anchor(g(cells))`` from leaking into the plane wiring).
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from core.augment import augment_sample
from games.blokus_duo import BlokusDuo
from games.blokus_duo.actions import action_cells
from games.blokus_duo.symmetry import GROUP_NAMES, cell_map
from tests.test_blokus_encoding import OPP_OCC, OWN_OCC, plane_cells

GAME = BlokusDuo()

# Game phases as target plies — the ply-0 root (828-wide opening support, empty
# board), midgame, and late game — totalling ~25 unmarked random states, each
# checked against all 4 group elements.
PHASES = [("opening", 0, 5), ("midgame", 8, 10), ("late", 18, 10)]

CASES = [
    pytest.param(plies, 100 * plies + i, id=f"{label}-{i}")
    for label, plies, num_seeds in PHASES
    for i in range(num_seeds)
]


def _random_sample(plies: int, seed: int):
    """Build one seeded random ``(planes, sparse π)`` sample for the decode check.

    Plays ``plies`` random legal actions through the adapter (stopping early if
    a move would end the game, so the sampled state is always nonterminal),
    then draws visit counts over the full legal support. Counts come from a
    small range on purpose: any support wider than the range forces equal
    counts on distinct actions, which is exactly why the assertion below is a
    multiset and not a set.

    Args:
        plies: Target number of random plies to play out.
        seed: RNG seed for both the playout and the visit counts.

    Returns:
        ``(planes, sparse_pi)`` — the ``encode_state`` output and the
        D12-shaped ``(action_id, visit_count)`` pairs.
    """
    rng = random.Random(seed)
    s = GAME.initial_state()
    for _ in range(plies):
        nxt = GAME.apply(s, rng.choice(list(GAME.legal_moves(s))))
        if GAME.is_terminal(nxt):
            break
        s = nxt
    sparse_pi = [(a, rng.randint(1, 6)) for a in GAME.legal_moves(s)]
    return GAME.encode_state(s), sparse_pi


def _assert_pipeline_decodes(planes, sparse_pi):
    """Run the §12 decode check for one sample across every declared element.

    Args:
        planes: ``encode_state`` output for the sampled state.
        sparse_pi: Sparse policy target as ``(action_id, visit_count)`` pairs.
    """
    for g_index, name in enumerate(GROUP_NAMES):
        m = cell_map(name)
        aug_planes, aug_pi = augment_sample(GAME, planes, sparse_pi, g_index)

        # Action side: decode every transformed (action_id, count) back to a
        # cell set; the multiset of (cell-set, count) pairs must equal g
        # applied cell-wise to the original decodes. Counter equality also
        # catches dropped, duplicated, or merged pairs.
        expected = Counter(
            (frozenset(m(r, c) for r, c in action_cells(a)), n) for a, n in sparse_pi
        )
        decoded = Counter((frozenset(action_cells(a)), n) for a, n in aug_pi)
        assert decoded == expected, name

        # Plane side: the same geometry through the plane transform — the
        # transformed occupancy planes must be the cell-mapped originals.
        for p in (OWN_OCC, OPP_OCC):
            mapped = {m(r, c) for r, c in plane_cells(planes[p])}
            assert plane_cells(aug_planes[p]) == mapped, (name, p)


@pytest.mark.parametrize("plies,seed", CASES)
def test_pipeline_decode(plies, seed):
    planes, sparse_pi = _random_sample(plies, seed)
    _assert_pipeline_decodes(planes, sparse_pi)


@pytest.mark.slow
def test_pipeline_decode_sweep():
    # Larger sweep over the whole ply range (random games rarely exceed ~26
    # plies before a side is blocked; the sampler stops early at terminal).
    rng = random.Random(97)
    for _ in range(60):
        planes, sparse_pi = _random_sample(rng.randrange(0, 26), rng.randrange(10**9))
        _assert_pipeline_decodes(planes, sparse_pi)
