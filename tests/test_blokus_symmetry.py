"""Klein-4 symmetry: joint-permutation golden, group laws, equivariance, adapter group.

The (g,a)→a′ table is built decode → transform cells → re-encode, which equals
``anchor(g(cells))`` by construction — the doc's named failure mode
(``g(anchor) != anchor(g(cells))``) is real: naive anchor transport is wrong
for most ids under 180°. The checked-in fixture freezes the table; this file
recomputes all 4×13,729 = 54,916 entries against it.

[F4] independence: the equivariance property transforms *states* with the cell
maps hardcoded here (vet-verified formulas, own code path over oracle
frozensets) and *actions* through the checked-in fixture only.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from games.blokus_duo import BlokusDuo
from games.blokus_duo.actions import (
    IN_BOUNDS_ACTIONS,
    NUM_ACTIONS,
    OPENING_ACTIONS,
    action_cells,
    encode_cells,
)
from games.blokus_duo.oracle import OracleEngine
from games.blokus_duo.pieces import orientation_table_hash
from games.blokus_duo.symmetry import GROUP_NAMES, build_action_maps, full_permutation

FIXTURE = Path(__file__).parent / "fixtures" / "blokus" / "symmetry_table.json"

# Vet-verified Klein-4 cell maps, hardcoded independently of symmetry.py [F4].
_TEST_MAPS = {
    "identity": lambda r, c: (r, c),
    "rot180": lambda r, c: (13 - r, 13 - c),
    "diag": lambda r, c: (c, r),
    "antidiag": lambda r, c: (13 - c, 13 - r),
}


@pytest.fixture(scope="module")
def fixture_table():
    data = json.loads(FIXTURE.read_text())
    assert data["orientation_hash"] == orientation_table_hash()
    assert data["conventions"]["flatten"] == "(r*14+c)*91+o"
    assert data["conventions"]["start_squares"] == [[4, 4], [9, 9]]
    actions = data["actions"]
    assert actions == list(IN_BOUNDS_ACTIONS)
    return {
        g: dict(zip(actions, data["maps"][g], strict=True)) for g in GROUP_NAMES
    }


def test_group_is_the_start_square_stabilizer():
    starts = {(4, 4), (9, 9)}
    for g, m in _TEST_MAPS.items():
        assert {m(*sq) for sq in starts} == starts, g


def test_group_laws_on_cells():
    # Klein four-group: every element is an involution; antidiag = diag∘rot180.
    for r in range(14):
        for c in range(14):
            for g in ("rot180", "diag", "antidiag"):
                assert _TEST_MAPS[g](*_TEST_MAPS[g](r, c)) == (r, c)
            assert _TEST_MAPS["antidiag"](r, c) == _TEST_MAPS["diag"](*_TEST_MAPS["rot180"](r, c))


def test_joint_permutation_golden(fixture_table):
    # Every g × every in-bounds action: decode → transform cells → re-encode
    # must match the checked-in fixture (54,916 checks).
    for g in GROUP_NAMES:
        m = _TEST_MAPS[g]
        table = fixture_table[g]
        assert len(table) == 13_729
        for a in IN_BOUNDS_ACTIONS:
            expected = encode_cells([m(r, c) for r, c in action_cells(a)])
            assert table[a] == expected, (g, a)


def test_table_closed_and_bijective(fixture_table):
    in_bounds = set(IN_BOUNDS_ACTIONS)
    for g in GROUP_NAMES:
        image = set(fixture_table[g].values())
        assert image == in_bounds  # closed on in-bounds ids and a bijection


def test_table_matches_symmetry_module(fixture_table):
    # The module's cached maps must agree with the checked-in fixture exactly.
    assert build_action_maps() == fixture_table


def test_openings_permute_within_openings(fixture_table):
    openings = set(OPENING_ACTIONS[(4, 4)]) | set(OPENING_ACTIONS[(9, 9)])
    for g in GROUP_NAMES:
        assert {fixture_table[g][a] for a in openings} == openings


def _transform_oracle_state(state, g):
    """Transform an oracle state with the test's own cell maps (no symmetry.py)."""
    m = _TEST_MAPS[g]
    occ0 = frozenset(m(r, c) for r, c in state[0])
    occ1 = frozenset(m(r, c) for r, c in state[1])
    return (occ0, occ1, *state[2:])


def test_equivariant_move_gen_on_random_states(fixture_table):
    # [F4] legal(g(s)) == g(legal(s)) with states transformed independently of
    # the table under test.
    engine = OracleEngine()
    game = BlokusDuo(engine)
    rng = random.Random(21)
    for seed_game in range(2):
        s = game.initial_state()
        states = [s]
        while not game.is_terminal(s):
            s = game.apply(s, rng.choice(list(game.legal_moves(s))))
            states.append(s)
        samples = [st for st in states[:12] if not game.is_terminal(st)][:8]
        assert len(samples) >= 6, f"game {seed_game} ended too early to sample"
        for st in samples:
            player = game.current_player(st)
            legal = engine.legal_actions(st, player)
            for g in GROUP_NAMES:
                transformed = _transform_oracle_state(st, g)
                mapped = sorted(fixture_table[g][a] for a in legal)
                assert engine.legal_actions(transformed, player) == mapped, g


def test_adapter_symmetry_group_shape():
    # [F6] first slot: state-level transform callable (M2 rebinds the plane
    # side); second slot: full 17,836 permutation with identity filler on
    # off-support ids.
    game = BlokusDuo()
    group = game.symmetry_group
    assert len(group) == 4
    maps = build_action_maps()
    in_bounds = set(IN_BOUNDS_ACTIONS)
    s0 = game.initial_state()
    for name, (state_t, perm) in zip(GROUP_NAMES, group, strict=True):
        assert state_t(s0) == s0  # empty boards are fixed points of every g
        assert len(perm) == NUM_ACTIONS
        for a in IN_BOUNDS_ACTIONS:
            assert perm[a] == maps[name][a]
        assert all(perm[a] == a for a in range(NUM_ACTIONS) if a not in in_bounds)


def test_adapter_state_transform_handles_bitboard_states():
    from games.blokus_duo.bitboard import BitboardEngine, cells_to_bb

    game = BlokusDuo(BitboardEngine())
    state_t, _ = game.symmetry_group[1]  # rot180
    s = (cells_to_bb([(0, 0)]), cells_to_bb([(13, 13)]), *game.initial_state()[2:])
    t = state_t(s)
    assert t[0] == cells_to_bb([(13, 13)])  # rot180 of (0,0)
    assert t[1] == cells_to_bb([(0, 0)])
