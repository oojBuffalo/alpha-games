"""Run-seeding contract tests (M2.5 task 6.2, built to the M3 task-2 spec).

Three properties carry the whole design: derivation is *stable* (golden ints below fail
loudly if the serialization ever changes, because that silently re-keys every run),
distinct labels give distinct streams, and streams are *independent* — drawing from one
purpose cannot perturb another, which is what lets M3 deepen the keying to durable
coordinates for crash-resume.
"""

from __future__ import annotations

import random

import pytest

from core.seeding import (
    PURPOSE_AUGMENTATION,
    PURPOSE_DIRICHLET,
    PURPOSE_MOVE_SELECTION,
    PURPOSE_NET_INIT,
    PURPOSE_TIE_BREAK,
    PURPOSE_WINDOW_SAMPLING,
    GameRNGs,
    LearnerRNGs,
    component_rng,
    derive_seed,
    net_init_seed,
)

# Frozen golden derivations: sha256 over the canonical serialization, first 8 bytes.
# A change here is a deliberate re-keying of every run, never an incidental refactor.
GOLDEN = {
    (0,): 7616342192435800936,
    (12345,): 2783236798444063219,
    (-1,): 15812938958143454403,
    (12345, "net-init"): 5543978608183432423,
    (12345, "game", 0, "dirichlet"): 7697570542772425456,
    (12345, "game", 1, "dirichlet"): 7117826159696791745,
    (12345, "actor", 3, "game", 0, "move-selection"): 9824953671185842631,
    (12345, "learner", 7, "replay-sampling"): 14003624692365441275,
}


@pytest.mark.parametrize(("args", "expected"), sorted(GOLDEN.items(), key=repr))
def test_derive_seed_golden_values(args, expected):
    run_seed, *labels = args
    assert derive_seed(run_seed, *labels) == expected


def test_derived_seeds_fit_64_unsigned_bits():
    for args in GOLDEN:
        assert 0 <= derive_seed(args[0], *args[1:]) < 2**64


def test_derivation_is_deterministic_across_calls():
    a = derive_seed(999, "actor", 2, "game", 41, PURPOSE_DIRICHLET)
    b = derive_seed(999, "actor", 2, "game", 41, PURPOSE_DIRICHLET)
    assert a == b


# --- distinctness ---------------------------------------------------------------------


def test_distinct_labels_give_distinct_seeds():
    labels = [
        (PURPOSE_NET_INIT,),
        (PURPOSE_DIRICHLET,),
        (PURPOSE_MOVE_SELECTION,),
        (PURPOSE_TIE_BREAK,),
        (PURPOSE_AUGMENTATION,),
        (PURPOSE_WINDOW_SAMPLING,),
        ("game", 0, PURPOSE_DIRICHLET),
        ("game", 1, PURPOSE_DIRICHLET),
        ("actor", 0, "game", 0, PURPOSE_DIRICHLET),
        ("actor", 1, "game", 0, PURPOSE_DIRICHLET),
    ]
    seeds = {derive_seed(4242, *lab) for lab in labels}
    assert len(seeds) == len(labels)


def test_distinct_run_seeds_give_distinct_streams():
    assert derive_seed(1, PURPOSE_DIRICHLET) != derive_seed(2, PURPOSE_DIRICHLET)


def test_label_type_and_order_never_collide():
    # The canonical serialization is type-tagged and length-prefixed, so these three label
    # tuples cannot alias each other (a plain string join would collapse the first two).
    seeds = {
        derive_seed(0, "a", 1),
        derive_seed(0, "a1"),
        derive_seed(0, 1, "a"),
        derive_seed(0, "a", "1"),
    }
    assert len(seeds) == 4


def test_labels_must_be_str_or_int():
    with pytest.raises(TypeError):
        derive_seed(0, 1.5)
    with pytest.raises(TypeError):
        derive_seed(0, ("nested",))
    with pytest.raises(TypeError):
        derive_seed(0, True)  # an int subclass, but a flag — never a stable label
    with pytest.raises(TypeError):
        derive_seed("not-an-int", "x")


# --- streams --------------------------------------------------------------------------


def test_component_rng_is_seeded_from_the_derived_int():
    labels = ("game", 3, PURPOSE_MOVE_SELECTION)
    expected = random.Random(derive_seed(77, *labels))
    got = component_rng(77, *labels)
    assert [got.random() for _ in range(5)] == [expected.random() for _ in range(5)]


def test_re_derivation_reproduces_an_identical_stream():
    # The crash-resume property: a stream is recomputable from (run_seed, labels) alone.
    first = component_rng(77, "actor", 1, "game", 12, PURPOSE_DIRICHLET)
    burn = [first.random() for _ in range(20)]
    again = component_rng(77, "actor", 1, "game", 12, PURPOSE_DIRICHLET)
    assert [again.random() for _ in range(20)] == burn


def test_net_init_seed_matches_its_label():
    assert net_init_seed(31337) == derive_seed(31337, PURPOSE_NET_INIT)


def test_game_bundle_streams_are_distinct():
    rngs = GameRNGs.for_game(5, 0)
    draws = {
        rngs.dirichlet.random(),
        rngs.move_selection.random(),
        rngs.tie_break.random(),
    }
    assert len(draws) == 3


def test_game_bundle_streams_are_independent():
    # Stream independence: burning extra draws on one purpose leaves every other purpose's
    # sequence untouched — adding a Dirichlet draw must never reshuffle move selection.
    baseline = GameRNGs.for_game(5, 9)
    expected = {
        "move_selection": [baseline.move_selection.random() for _ in range(5)],
        "tie_break": [baseline.tie_break.random() for _ in range(5)],
    }

    perturbed = GameRNGs.for_game(5, 9)
    for _ in range(137):
        perturbed.dirichlet.random()
    assert [perturbed.move_selection.random() for _ in range(5)] == expected["move_selection"]
    assert [perturbed.tie_break.random() for _ in range(5)] == expected["tie_break"]


def test_game_bundles_differ_per_game_index_and_prefix():
    a = GameRNGs.for_game(5, 0).dirichlet.random()
    b = GameRNGs.for_game(5, 1).dirichlet.random()
    c = GameRNGs.for_game(5, 0, prefix=("actor", 1)).dirichlet.random()
    assert len({a, b, c}) == 3


def test_game_bundle_labels_match_the_documented_shape():
    rngs = GameRNGs.for_game(5, 4, prefix=("actor", 2))
    expected = component_rng(5, "actor", 2, "game", 4, PURPOSE_DIRICHLET)
    assert rngs.dirichlet.random() == expected.random()


def test_learner_bundle_re_keys_per_step_and_stays_independent():
    step_a, step_b = LearnerRNGs.for_step(8, 0), LearnerRNGs.for_step(8, 1)
    assert step_a.augmentation.random() != step_b.augmentation.random()

    baseline = LearnerRNGs.for_step(8, 3)
    expected = [baseline.window_sampling.random() for _ in range(5)]
    perturbed = LearnerRNGs.for_step(8, 3)
    for _ in range(50):
        perturbed.augmentation.random()
    assert [perturbed.window_sampling.random() for _ in range(5)] == expected


def test_bundles_are_frozen():
    rngs = GameRNGs.for_game(5, 0)
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
        rngs.dirichlet = random.Random(0)
