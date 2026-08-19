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
    (12345, "actor", 3, "game", 0, "dirichlet"): 15804084277158994818,
    (12345, "actor", 3, "game", 0, "tie-break"): 687149822308989655,
    (12345, "learner", 7, "replay-sampling"): 14003624692365441275,
    (12345, "learner", 7, "augmentation"): 7221077250744959150,
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


# --- M3: multi-actor label family + durable-coordinate keying -------------------------
#
# GameRNGs.for_actor_game / LearnerRNGs.for_step are the typed helper surface M3's actor
# and learner components (later issues) build on. Both are thin sugar over derive_seed:
# no RNG state is ever stored, only the durable coordinates (actor_id, game_index; the
# checkpointed learner step) that let any stream be recomputed after a crash.


def test_canonicalization_never_collides_across_a_part_boundary():
    # A length-prefixed, type-tagged encoding is self-delimiting: shifting where one
    # string ends and the next begins, or swapping an int label for its string spelling,
    # must never alias to the same serialized bytes.
    seeds = {
        derive_seed(0, "ab", "c"),
        derive_seed(0, "a", "bc"),
        derive_seed(0, 1),
        derive_seed(0, "1"),
    }
    assert len(seeds) == 4


def test_for_actor_game_matches_the_documented_actor_label_family():
    # The exact label family from issue #53: ("actor", actor_id, "game", game_index,
    # purpose), purpose in {dirichlet, move-selection, tie-break}.
    rngs = GameRNGs.for_actor_game(12345, actor_id=3, game_index=0)
    dirichlet_golden = GOLDEN[(12345, "actor", 3, "game", 0, "dirichlet")]
    move_golden = GOLDEN[(12345, "actor", 3, "game", 0, "move-selection")]
    tie_break_golden = GOLDEN[(12345, "actor", 3, "game", 0, "tie-break")]
    assert rngs.dirichlet.random() == random.Random(dirichlet_golden).random()
    assert rngs.move_selection.random() == random.Random(move_golden).random()
    assert rngs.tie_break.random() == random.Random(tie_break_golden).random()


def test_for_actor_game_is_sugar_over_for_game_with_the_actor_prefix():
    via_helper = GameRNGs.for_actor_game(2026, actor_id=7, game_index=4)
    via_prefix = GameRNGs.for_game(2026, 4, prefix=("actor", 7))
    assert via_helper.dirichlet.random() == via_prefix.dirichlet.random()
    assert via_helper.move_selection.random() == via_prefix.move_selection.random()
    assert via_helper.tie_break.random() == via_prefix.tie_break.random()


def test_two_actors_are_decorrelated_for_the_same_game_index():
    # Parallel actors decorrelate by construction: actor_id folds into every label, so
    # two actors never draw the same stream even when they happen to be on the same
    # durable game_index.
    a = GameRNGs.for_actor_game(2026, actor_id=0, game_index=5)
    b = GameRNGs.for_actor_game(2026, actor_id=1, game_index=5)
    assert a.dirichlet.random() != b.dirichlet.random()
    assert a.move_selection.random() != b.move_selection.random()
    assert a.tie_break.random() != b.tie_break.random()


def test_actor_game_index_and_purpose_each_change_the_stream():
    base = GameRNGs.for_actor_game(2026, actor_id=0, game_index=5)
    other_game = GameRNGs.for_actor_game(2026, actor_id=0, game_index=6)
    assert base.dirichlet.random() != other_game.dirichlet.random()
    draws = {base.dirichlet.random(), base.move_selection.random(), base.tie_break.random()}
    assert len(draws) == 3


def test_actor_game_stream_is_recomputable_after_a_simulated_crash():
    # Crash-resume: derive the stream from durable coordinates (actor_id, game_index),
    # draw from it, then discard the generator entirely — no RNG state is carried across
    # the "restart", only the coordinates that produced it.
    actor_id, game_index = 4, 17
    before_crash = GameRNGs.for_actor_game(999, actor_id=actor_id, game_index=game_index)
    burned = [before_crash.dirichlet.random() for _ in range(30)]
    del before_crash  # simulate the process dying with zero RNG state persisted

    resumed = GameRNGs.for_actor_game(999, actor_id=actor_id, game_index=game_index)
    assert [resumed.dirichlet.random() for _ in range(30)] == burned


def test_learner_step_stream_is_recomputable_after_a_simulated_crash():
    # Same property on the learner side: the only persisted coordinate is the
    # checkpointed step counter, never a stream position.
    learner_step = 42
    before_crash = LearnerRNGs.for_step(999, learner_step)
    burned_aug = [before_crash.augmentation.random() for _ in range(10)]
    burned_window = [before_crash.window_sampling.random() for _ in range(10)]
    del before_crash

    resumed = LearnerRNGs.for_step(999, learner_step)
    assert [resumed.augmentation.random() for _ in range(10)] == burned_aug
    assert [resumed.window_sampling.random() for _ in range(10)] == burned_window


def test_learner_golden_augmentation_stream():
    rngs = LearnerRNGs.for_step(12345, 7)
    expected = random.Random(GOLDEN[(12345, "learner", 7, "augmentation")])
    assert rngs.augmentation.random() == expected.random()
