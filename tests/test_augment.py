"""Training-side symmetry augmentation: ``core/augment.py`` (M2, D9/§8).

Game-genericity is the point: the same tests run against Blokus Duo (Klein-4,
17,836-id head with off-support identity filler) and Othello (full D4, flat
65-id head with the pass fixed point). ``augment_sample`` composes only the
adapter-declared ``symmetry_group`` callables, so nothing here reaches into
either game's symmetry module — inverses are derived from the declared
permutations themselves.
"""

from __future__ import annotations

import random

import pytest

from core.augment import augment_sample
from games.blokus_duo import BlokusDuo
from games.othello import Othello

# (game, mid-game plies): deep enough that the position is asymmetric under
# every declared element, so identity/round-trip equalities are non-vacuous.
GAMES = [(BlokusDuo(), 6), (Othello(), 7)]
GAME_IDS = ["blokus-duo", "othello"]

BLOKUS, BLOKUS_PLIES = GAMES[0]


def _sample(game, plies, seed):
    """Build one D12-shaped training sample at a seeded random mid-game state.

    Returns:
        ``(planes, sparse_pi)`` with distinct positive visit counts over the
        state's full legal support, so any dropped/duplicated/misrouted pair
        breaks an equality below.
    """
    rng = random.Random(seed)
    s = game.initial_state()
    for _ in range(plies):
        s = game.apply(s, rng.choice(list(game.legal_moves(s))))
    assert not game.is_terminal(s)
    legal = list(game.legal_moves(s))
    counts = rng.sample(range(1, len(legal) + 1), len(legal))
    return game.encode_state(s), [(a, n) for a, n in zip(legal, counts, strict=True)]


def _inverse_index(group, g_index):
    """Find the declared index of the inverse of element ``g_index``.

    Derived purely from the action permutations (a finite group declared in
    full must contain every inverse); the plane side is checked against the
    same pairing by the round-trip test.
    """
    perm_g = group[g_index][1]
    for h, (_, perm_h) in enumerate(group):
        if all(perm_h[perm_g[a]] == a for a in range(len(perm_g))):
            return h
    raise AssertionError(f"declared group has no inverse for element {g_index}")


@pytest.mark.parametrize("game,plies", GAMES, ids=GAME_IDS)
def test_identity_element_is_a_noop(game, plies):
    # Element 0 is identity by both adapters' element-order pins: planes and
    # pairs come back exactly equal (order included).
    planes, sparse_pi = _sample(game, plies, seed=3)
    aug_planes, aug_pi = augment_sample(game, planes, sparse_pi, 0)
    assert aug_planes == planes
    assert aug_pi == sparse_pi


@pytest.mark.parametrize("game,plies", GAMES, ids=GAME_IDS)
def test_visit_count_multiset_preserved_under_every_g(game, plies):
    # A symmetry relabels actions; it never redistributes visit mass. Distinct
    # image ids assert the permutation stays injective on the support.
    planes, sparse_pi = _sample(game, plies, seed=17)
    for g_index in range(len(game.symmetry_group)):
        _, aug_pi = augment_sample(game, planes, sparse_pi, g_index)
        assert len(aug_pi) == len(sparse_pi)
        assert sorted(n for _, n in aug_pi) == sorted(n for _, n in sparse_pi), g_index
        assert len({a for a, _ in aug_pi}) == len(sparse_pi), g_index


@pytest.mark.parametrize("game,plies", GAMES, ids=GAME_IDS)
def test_g_then_inverse_round_trips(game, plies):
    # Applying g then g^-1 (found in the declared group itself) restores the
    # sample exactly — Klein-4 elements are self-inverse; Othello's rot90 and
    # rot270 invert each other, so the pairing is derived, not assumed.
    group = game.symmetry_group
    planes, sparse_pi = _sample(game, plies, seed=29)
    for g_index in range(len(group)):
        h_index = _inverse_index(group, g_index)
        back_planes, back_pi = augment_sample(
            game, *augment_sample(game, planes, sparse_pi, g_index), h_index
        )
        assert back_planes == planes, (g_index, h_index)
        assert back_pi == sparse_pi, (g_index, h_index)


def test_klein_four_elements_are_self_inverse():
    # The Blokus-specific sharpening of the round-trip: every Klein-4 element
    # is its own inverse, so g twice is already the identity.
    group = BLOKUS.symmetry_group
    assert len(group) == 4
    planes, sparse_pi = _sample(BLOKUS, BLOKUS_PLIES, seed=43)
    for g_index in range(len(group)):
        assert _inverse_index(group, g_index) == g_index
        twice = augment_sample(BLOKUS, *augment_sample(BLOKUS, planes, sparse_pi, g_index), g_index)
        assert twice == (planes, sparse_pi), g_index


@pytest.mark.parametrize("game,plies", GAMES, ids=GAME_IDS)
def test_out_of_range_element_fails_loudly(game, plies):
    # No silent fallback to identity: an index outside the declared group is
    # a caller bug and must raise.
    planes, sparse_pi = _sample(game, plies, seed=51)
    with pytest.raises(IndexError):
        augment_sample(game, planes, sparse_pi, len(game.symmetry_group))
