"""Micro-Blokus symmetry (M2.5 task 4.3/4.4): joint-permutation golden, pipeline decode.

The M1 battery at micro scale. ``tests/test_blokus_symmetry.py`` freezes the
full game's 4 × 13,729 = 54,916-entry ``(g,a)→a′`` table; this file does the
same for the §5.3 micro instance's 4 × 159 = 636 entries against
``tests/fixtures/blokus_micro/symmetry_table.json`` — a *separate* fixture
carrying the micro orientation hash, because micro orientation ids are
re-derived within the piece subset (invariant 4) rather than the full table
restricted.

[F4] independence, as at M1: the cell maps are hardcoded here (vet-verifiable
5×5 formulas) and the images come from the checked-in fixture, so neither side
of the golden is produced by ``symmetry.py``. The named failure mode
``g(anchor) != anchor(g(cells))`` gets its own test in both of its disguises —
transporting the anchor cell, and carrying the orientation id — since a
shortcut that happens to be right for one element (``diag`` *is* a transpose,
so its anchors do transport) is still wrong for the other two.

Also here: M2's §12 pipeline decode check instantiated on micro. Its full-game
version (``tests/test_blokus_pipeline.py``) binds the adapter, group and codec
as module globals, so the two helpers are mirrored rather than imported. And
the write-side byte-stability check for both instances' symmetry fixtures
(task 4.4): regeneration on unchanged code must reproduce the committed bytes.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

from core.augment import augment_sample
from games.blokus_duo import BlokusDuo
from games.blokus_duo.actions import action_codec
from games.blokus_duo.config import MICRO_CONFIG
from games.blokus_duo.pieces import orientation_table_hash
from games.blokus_duo.symmetry import symmetry_group

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "blokus_micro" / "symmetry_table.json"
GEN_SCRIPT = ROOT / "scripts" / "gen_blokus_symmetry_table.py"

CODEC = action_codec(MICRO_CONFIG)
GROUP = symmetry_group(MICRO_CONFIG)
GROUP_NAMES = GROUP.names
GAME = BlokusDuo(config=MICRO_CONFIG)

# §5.3 goldens (independently enumerated by scripts/enumerate_micro_config.py).
BOARD_SIZE = 5
NUM_IN_BOUNDS = 159
NUM_ACTIONS = 225  # 5*5*9
NUM_PLANES = 12  # 2 occ + 2x4 inventory + 2 monomino-last
NUM_OPENINGS = 42  # 21 per start square

# Plane indices in the D3 order (own = side to move).
OWN_OCC, OPP_OCC = 0, 1

# Vet-verified Klein-4 cell maps on a 5×5 board, hardcoded independently of
# symmetry.py [F4]. Klein-4 is the D4 set-stabilizer of {(1,1), (3,3)}; that it
# is *exactly* the stabilizer (the 90° classes move a start square off the pair)
# is enumerated over all eight D4 elements in tests/test_blokus_config.py.
_TEST_MAPS = {
    "identity": lambda r, c: (r, c),
    "rot180": lambda r, c: (4 - r, 4 - c),
    "diag": lambda r, c: (c, r),
    "antidiag": lambda r, c: (4 - c, 4 - r),
}


@pytest.fixture(scope="module")
def fixture_table():
    """The checked-in micro ``(g,a)→a′`` table, keyed ``[g][action_id]``."""
    data = json.loads(FIXTURE.read_text())
    # Provenance: the micro instance's *own* orientation hash and conventions —
    # a fixture generated from the full table (or the full game's flatten) must
    # not silently pass as the micro golden.
    assert data["orientation_hash"] == orientation_table_hash(MICRO_CONFIG)
    assert data["conventions"]["flatten"] == "(r*5+c)*9+o"
    assert data["conventions"]["start_squares"] == [[1, 1], [3, 3]]
    actions = data["actions"]
    assert actions == list(CODEC.in_bounds_actions)
    assert len(actions) == NUM_IN_BOUNDS
    return {g: dict(zip(actions, data["maps"][g], strict=True)) for g in GROUP_NAMES}


# --- the group itself ---------------------------------------------------------------


def test_group_is_klein_four_over_the_micro_start_squares():
    # Element order is the pin: identity is element 0 (D9 counts it among the 4
    # augmentation symmetries), and the names match the full game's — the group
    # is the same abstract Klein-4, over a different board.
    assert GROUP_NAMES == ("identity", "rot180", "diag", "antidiag")
    assert set(GROUP_NAMES) == set(_TEST_MAPS)
    starts = {(1, 1), (3, 3)}
    for g, m in _TEST_MAPS.items():
        assert {m(*sq) for sq in starts} == starts, g


def test_group_laws_on_cells():
    # Klein four-group: every element is an involution; antidiag = diag∘rot180.
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            for g in ("rot180", "diag", "antidiag"):
                assert _TEST_MAPS[g](*_TEST_MAPS[g](r, c)) == (r, c)
            assert _TEST_MAPS["antidiag"](r, c) == _TEST_MAPS["diag"](*_TEST_MAPS["rot180"](r, c))


# --- the joint-permutation golden ---------------------------------------------------


def test_joint_permutation_golden(fixture_table):
    # Every g × every in-bounds micro action: decode → transform cells →
    # re-encode must match the checked-in fixture (4 × 159 = 636 checks).
    for g in GROUP_NAMES:
        m = _TEST_MAPS[g]
        table = fixture_table[g]
        assert len(table) == NUM_IN_BOUNDS
        for a in CODEC.in_bounds_actions:
            expected = CODEC.encode_cells([m(r, c) for r, c in CODEC.action_cells(a)])
            assert table[a] == expected, (g, a)


def test_anchor_is_re_derived_not_transported(fixture_table):
    # Named failure mode, disguise 1: g(anchor) != anchor(g(cells)). The image's
    # anchor is the bbox top-left of the *transformed* cells (D2), so an
    # implementation that maps the anchor cell through g and keeps the shape
    # lands on the wrong id for any piece wider than 1×1 under the 180° classes.
    # diag is a transpose, so its anchors do transport — which is exactly why
    # this check cannot be run on one element and generalized.
    shortcut_wrong = {}
    for g in GROUP_NAMES:
        m = _TEST_MAPS[g]
        wrong = 0
        for a in CODEC.in_bounds_actions:
            r, c, _ = CODEC.decode(a)
            image_cells = [m(rr, cc) for rr, cc in CODEC.action_cells(a)]
            true_anchor = (min(rr for rr, _ in image_cells), min(cc for _, cc in image_cells))
            assert CODEC.decode(fixture_table[g][a])[:2] == true_anchor, (g, a)
            wrong += m(r, c) != true_anchor
        shortcut_wrong[g] = wrong
    assert shortcut_wrong == {"identity": 0, "rot180": 134, "diag": 0, "antidiag": 134}


def test_orientation_id_is_re_derived_not_carried(fixture_table):
    # Named failure mode, disguise 2: the transformed cells are generally a
    # *different* fixed orientation, so carrying the original orientation id
    # (even with a correct anchor) is wrong for most ids under every reflection.
    # Counts are goldens over the micro orientation table {1, 2, 2, 4}: only the
    # monomino and the fully symmetric shapes keep their id.
    changed = {}
    for g in GROUP_NAMES:
        m = _TEST_MAPS[g]
        count = 0
        for a in CODEC.in_bounds_actions:
            image = CODEC.encode_cells([m(r, c) for r, c in CODEC.action_cells(a)])
            assert fixture_table[g][a] == image
            count += CODEC.decode(image)[2] != CODEC.decode(a)[2]
        changed[g] = count
    assert changed == {"identity": 0, "rot180": 64, "diag": 102, "antidiag": 102}


def test_table_closed_and_bijective(fixture_table):
    in_bounds = set(CODEC.in_bounds_actions)
    for g in GROUP_NAMES:
        images = list(fixture_table[g].values())
        assert len(set(images)) == NUM_IN_BOUNDS  # injective on 159 ids
        assert set(images) == in_bounds  # and closed on them, hence a bijection


def test_table_matches_symmetry_module(fixture_table):
    # The module's cached per-config maps must agree with the fixture exactly.
    assert GROUP.action_maps() == fixture_table


def test_openings_permute_within_openings(fixture_table):
    openings = set(CODEC.opening_actions[(1, 1)]) | set(CODEC.opening_actions[(3, 3)])
    assert len(openings) == NUM_OPENINGS
    for g in GROUP_NAMES:
        assert {fixture_table[g][a] for a in openings} == openings


def test_adapter_symmetry_group_shape(fixture_table):
    # The declared surface core/augment.py consumes: per element, the plane
    # transform over the 12-plane micro encoding and the full 225-length head
    # permutation with identity filler on the 66 off-support ids.
    group = GAME.symmetry_group
    in_bounds = set(CODEC.in_bounds_actions)
    planes0 = GAME.encode_state(GAME.initial_state())
    assert len(group) == 4
    for name, (plane_t, perm) in zip(GROUP_NAMES, group, strict=True):
        transformed = plane_t(planes0)
        assert len(transformed) == NUM_PLANES
        # Every initial-state plane is a constant broadcast (empty board, full
        # inventories), so all 4 elements fix it; equivariance on asymmetric
        # states is the pipeline decode test below.
        assert transformed == planes0
        assert len(perm) == NUM_ACTIONS
        for a in in_bounds:
            assert perm[a] == fixture_table[name][a]
        assert all(perm[a] == a for a in range(NUM_ACTIONS) if a not in in_bounds)


# --- M2's §12 pipeline decode check, on the micro instance --------------------------

# Micro games last 2–8 plies (4 pieces a side), so the phases are opening,
# midgame and endgame at 0 / 2 / 4 plies.
PHASES = [("opening", 0, 4), ("midgame", 2, 6), ("late", 4, 6)]

CASES = [
    pytest.param(plies, 100 * plies + i, id=f"{label}-{i}")
    for label, plies, num_seeds in PHASES
    for i in range(num_seeds)
]


def _plane_cells(plane):
    """Return the set of ``(r, c)`` cells set to 1 in a 5×5 plane."""
    return {(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if plane[r][c]}


def _random_sample(plies: int, seed: int):
    """Build one seeded random ``(planes, sparse π)`` micro sample.

    Plays ``plies`` random legal actions through the adapter (stopping early if
    a move would end the game, so the sampled state is always nonterminal), then
    draws visit counts over the full legal support from a small range — narrow
    on purpose, so distinct actions carry equal counts and the assertion below
    has to be a multiset rather than a set.

    Args:
        plies: Target number of random plies to play out.
        seed: RNG seed for both the playout and the visit counts.

    Returns:
        ``(planes, sparse_pi)`` — the ``encode_state`` output and the D12-shaped
        ``(action_id, visit_count)`` pairs.
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
    """Run the §12 decode check for one micro sample across every element.

    Args:
        planes: ``encode_state`` output for the sampled state.
        sparse_pi: Sparse policy target as ``(action_id, visit_count)`` pairs.
    """
    for g_index, name in enumerate(GROUP_NAMES):
        m = _TEST_MAPS[name]
        aug_planes, aug_pi = augment_sample(GAME, planes, sparse_pi, g_index)

        # Action side: decode every transformed (action_id, count) back to a
        # cell set; the multiset of (cell-set, count) pairs must equal g applied
        # cell-wise to the original decodes. Counter equality also catches
        # dropped, duplicated or merged pairs.
        expected = Counter(
            (frozenset(m(r, c) for r, c in CODEC.action_cells(a)), n) for a, n in sparse_pi
        )
        decoded = Counter((frozenset(CODEC.action_cells(a)), n) for a, n in aug_pi)
        assert decoded == expected, name

        # Plane side: the same geometry through the plane transform — the
        # transformed occupancy planes must be the cell-mapped originals (this
        # is where a plane/channel wiring mismatch invisible to the table-level
        # golden would surface).
        for p in (OWN_OCC, OPP_OCC):
            mapped = {m(r, c) for r, c in _plane_cells(planes[p])}
            assert _plane_cells(aug_planes[p]) == mapped, (name, p)


@pytest.mark.parametrize("plies,seed", CASES)
def test_pipeline_decode(plies, seed):
    planes, sparse_pi = _random_sample(plies, seed)
    _assert_pipeline_decodes(planes, sparse_pi)


def test_pipeline_decode_sweep():
    # Sweep over the whole micro ply range; the sampler stops early at terminal.
    rng = random.Random(31)
    for _ in range(40):
        planes, sparse_pi = _random_sample(rng.randrange(0, 8), rng.randrange(10**9))
        _assert_pipeline_decodes(planes, sparse_pi)


# --- write side: fixture regeneration is byte-stable (task 4.4) ---------------------


def _load_generator():
    """Import ``scripts/gen_blokus_symmetry_table.py`` (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("gen_blokus_symmetry_table", GEN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_symmetry_fixtures_regenerate_byte_identically(tmp_path):
    # Both directions of the M2.5 requirement in one check: the new micro
    # fixture is stable across regenerations, and the full game's committed
    # fixture is untouched by the parameterization. Anything that leaks
    # iteration order into the tables (invariant 4) breaks here rather than
    # silently invalidating every checkpoint keyed on the orientation hash.
    generator = _load_generator()
    runs = []
    for i in range(2):
        root = tmp_path / f"run{i}"
        generator.main(root)
        runs.append(
            {
                name: (root / "tests" / "fixtures" / name / "symmetry_table.json").read_bytes()
                for name, _ in generator.INSTANCES
            }
        )
    assert runs[0] == runs[1]  # deterministic on unchanged code
    for name, _ in generator.INSTANCES:
        committed = (ROOT / "tests" / "fixtures" / name / "symmetry_table.json").read_bytes()
        assert runs[0][name] == committed, name
