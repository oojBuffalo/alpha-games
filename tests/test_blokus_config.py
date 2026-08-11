"""Config parameterization: full-game invariance, micro goldens, micro smoke (M2.5 task 2).

Two claims are under test. First, that :data:`FULL_CONFIG` is genuinely the old
code path — the whole existing battery already proves that, so here it is only
the table/hash identity. Second, that the §5.3 micro instance the *same* package
constructs reproduces the constants ``scripts/enumerate_micro_config.py``
enumerated independently (board, orientations, placements, openings, planes, aux
divisor, orientation hash), and that both engines actually play it: a seeded
random game runs to termination in lockstep through the oracle and the bitboard
engine with identical legal-move lists and identical final scores.

The micro numbers below are literal goldens from the design doc §5.3 table — the
script's output, never read back from the code under test. Deep differential
coverage across configs is M2.5 task 3.
"""

from __future__ import annotations

import random

import pytest

from games.blokus_duo.actions import action_codec
from games.blokus_duo.bitboard import BitboardEngine
from games.blokus_duo.config import FULL_CONFIG, MICRO_CONFIG, BlokusConfig
from games.blokus_duo.game import BlokusDuo
from games.blokus_duo.oracle import OracleEngine
from games.blokus_duo.pieces import (
    BASE_PIECES,
    ORIENTATIONS,
    PIECE_NAMES,
    build_orientation_table,
    build_pieces,
    orientation_table_hash,
)
from games.blokus_duo.targets import max_score_diff, value_targets

FULL_HASH = "4408e7d7b56b3533685ce92e88fd1bc9453ba405f4afacfa96e431974446cb35"
MICRO_HASH = "78ea621ae2d1e27e239ecffa5ff44c793ef15f2884198a0394d394083d3e37e4"

MICRO_CODEC = action_codec(MICRO_CONFIG)


# --- the default config is the full game, bit for bit ------------------------------


def test_default_config_is_the_full_game():
    assert FULL_CONFIG == BlokusConfig()
    assert FULL_CONFIG.board_size == 14
    assert FULL_CONFIG.start_squares == ((4, 4), (9, 9))
    assert FULL_CONFIG.piece_names is None


def test_full_config_reproduces_the_module_piece_and_orientation_tables():
    # The parameterized builders must *be* the M1 tables, not merely agree in
    # aggregate: every fixture, checkpoint and replay dataset is keyed on them.
    assert build_pieces(FULL_CONFIG) == (BASE_PIECES, PIECE_NAMES)
    assert build_orientation_table(FULL_CONFIG) == ORIENTATIONS
    assert orientation_table_hash(FULL_CONFIG) == orientation_table_hash() == FULL_HASH


def test_full_config_codec_matches_the_module_surface():
    codec = action_codec(FULL_CONFIG)
    assert (codec.board_size, codec.num_orientations, codec.num_actions) == (14, 91, 17_836)
    assert len(codec.in_bounds_actions) == 13_729
    assert codec.fixture_conventions["flatten"] == "(r*14+c)*91+o"
    assert max_score_diff(FULL_CONFIG) == 109


def test_full_game_adapter_surface_unchanged_by_construction_paths():
    # BlokusDuo(), BlokusDuo(engine) and BlokusDuo(config=FULL_CONFIG) are the
    # same game — the zero-edit guarantee for every existing call site.
    for game in (BlokusDuo(), BlokusDuo(OracleEngine()), BlokusDuo(config=FULL_CONFIG)):
        assert game.config == FULL_CONFIG
        assert game.policy_shape == (14, 14, 91)
        assert game.input_planes == 46
        assert game.input_shape == (14, 14)


# --- config validation is loud and eager -------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"board_size": 0},
        {"piece_names": ()},  # a set with no pieces
        {"piece_names": ("I1", "I1")},  # duplicate
        {"piece_names": ("I1", "Q9")},  # unknown name
        {"start_squares": ((4, 4),)},  # §4 needs exactly two
        {"start_squares": ((4, 4), (4, 4))},  # duplicate squares
        {"board_size": 5, "start_squares": ((1, 1), (5, 5))},  # off-board
    ],
)
def test_invalid_configs_raise_at_construction(kwargs):
    with pytest.raises(ValueError):
        BlokusConfig(**kwargs)


def test_config_is_hashable_and_value_equal():
    # Hashability is load-bearing: every derived table is cached per config.
    twin = BlokusConfig(
        board_size=5, start_squares=((1, 1), (3, 3)), piece_names=("I1", "I2", "I3", "V3")
    )
    assert hash(MICRO_CONFIG) == hash(twin) and MICRO_CONFIG == twin
    assert action_codec(MICRO_CONFIG) is MICRO_CODEC


# --- micro goldens (design doc §5.3; scripts/enumerate_micro_config.py) ------------


def test_micro_piece_set():
    pieces, names = build_pieces(MICRO_CONFIG)
    assert names == ("I1", "I2", "I3", "V3")  # re-sorted by (size, canonical form)
    assert [len(p) for p in pieces] == [1, 2, 3, 3]
    assert sum(len(p) for p in pieces) == 9  # squares per set


def test_micro_orientation_ids_are_re_derived_within_the_subset():
    table = build_orientation_table(MICRO_CONFIG)
    assert [len(orients) for orients in table] == [1, 2, 2, 4]
    assert MICRO_CODEC.num_orientations == 9
    # Re-derived, not the full table restricted: ids run 0..8 piece-major and
    # the piece column is the subset's own.
    assert MICRO_CODEC.orientation_piece == (0, 1, 1, 2, 2, 3, 3, 3, 3)
    assert orientation_table_hash(MICRO_CONFIG) == MICRO_HASH
    assert orientation_table_hash(MICRO_CONFIG) != orientation_table_hash(FULL_CONFIG)


def test_micro_action_space():
    assert MICRO_CODEC.num_actions == 225  # 5*5*9
    assert len(MICRO_CODEC.in_bounds_actions) == 159
    assert MICRO_CODEC.fixture_conventions["flatten"] == "(r*5+c)*9+o"


def test_micro_openings():
    assert MICRO_CONFIG.start_squares == ((1, 1), (3, 3))
    for sq in MICRO_CONFIG.start_squares:
        covering = [a for a in MICRO_CODEC.in_bounds_actions if sq in MICRO_CODEC.action_cells(a)]
        assert len(covering) == 21
        assert tuple(covering) == MICRO_CODEC.opening_actions[sq]
    both = set(MICRO_CODEC.opening_actions[(1, 1)]) & set(MICRO_CODEC.opening_actions[(3, 3)])
    assert not both
    assert sum(len(v) for v in MICRO_CODEC.opening_actions.values()) == 42


def test_micro_encode_decode_bijection():
    seen = set()
    for a in MICRO_CODEC.in_bounds_actions:
        r, c, o = MICRO_CODEC.decode(a)
        assert MICRO_CODEC.encode(r, c, o) == a
        assert MICRO_CODEC.encode_cells(MICRO_CODEC.action_cells(a)) == a
        assert all(0 <= rr < 5 and 0 <= cc < 5 for rr, cc in MICRO_CODEC.action_cells(a))
        seen.add((r, c, o))
    assert len(seen) == 159


def test_micro_adapter_shapes_and_targets():
    game = BlokusDuo(config=MICRO_CONFIG)
    assert game.policy_shape == (5, 5, 9)
    assert game.input_planes == 12  # 2 occ + 2x4 inventory + 2 monomino-last
    assert game.input_shape == (5, 5)
    assert len(game.encode_state(game.initial_state())) == 12
    assert max_score_diff(MICRO_CONFIG) == 29  # 20 - (-9), the D1 aux divisor
    assert value_targets(20, -9, MICRO_CONFIG) == (1, 1.0)
    assert value_targets(0, 0, MICRO_CONFIG) == (0, 0.0)  # draws are not losses
    with pytest.raises(ValueError):
        value_targets(20, -10, MICRO_CONFIG)


def test_micro_initial_position():
    for engine in (OracleEngine(MICRO_CONFIG), BitboardEngine(MICRO_CONFIG)):
        game = BlokusDuo(engine)
        legal = list(game.legal_moves(game.initial_state()))
        assert len(legal) == 42
        assert set(legal) == set(MICRO_CODEC.opening_actions[(1, 1)]) | set(
            MICRO_CODEC.opening_actions[(3, 3)]
        )


def test_micro_config_is_not_wired_into_the_full_game_symmetry_surface():
    # Per-config symmetry lands in M2.5 task 4; until then the micro adapter
    # must refuse rather than declare the full game's 17,836-length group.
    with pytest.raises(NotImplementedError):
        _ = BlokusDuo(config=MICRO_CONFIG).symmetry_group


def test_mismatched_engine_and_config_fail_loudly():
    with pytest.raises(ValueError):
        BlokusDuo(BitboardEngine(FULL_CONFIG), config=MICRO_CONFIG)


# --- micro smoke: a full seeded game through each engine ---------------------------


def _play_micro_game(seed: int):
    """Play one seeded random micro game in lockstep through both engines.

    Args:
        seed: Seed for the move-choice RNG stream.

    Returns:
        ``(plies, scores)``: the number of placements played and the final
        score pair (identical for both engines, asserted below).
    """
    oracle, bitboard = OracleEngine(MICRO_CONFIG), BitboardEngine(MICRO_CONFIG)
    game_o, game_b = BlokusDuo(oracle), BlokusDuo(bitboard)
    rng = random.Random(seed)
    so, sb = game_o.initial_state(), game_b.initial_state()
    plies = 0
    while not game_o.is_terminal(so):
        assert game_b.is_terminal(sb) is False
        legal_o, legal_b = list(game_o.legal_moves(so)), list(game_b.legal_moves(sb))
        assert legal_o == legal_b  # same sorted legal-id set at every ply
        assert legal_o  # pass invariant: the mover always has a move
        a = rng.choice(legal_o)
        cells = game_o.decode_action(a)
        assert all(0 <= r < 5 and 0 <= c < 5 for r, c in cells)
        so, sb = game_o.apply(so, a), game_b.apply(sb, a)
        assert so[2:] == sb[2:]  # inventories, flags, to_play, terminal agree
        plies += 1
    assert game_b.is_terminal(sb)
    scores_o, scores_b = oracle.scores(so), bitboard.scores(sb)
    assert scores_o == scores_b
    return plies, scores_o


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 17])
def test_micro_random_game_runs_to_termination_through_both_engines(seed):
    plies, scores = _play_micro_game(seed)
    assert 2 <= plies <= 8  # both openings, at most the 4 pieces each
    assert all(-9 <= s <= 20 for s in scores)
    assert abs(scores[0] - scores[1]) <= max_score_diff(MICRO_CONFIG)
    z, aux = value_targets(scores[0], scores[1], MICRO_CONFIG)
    assert z in (-1, 0, 1)
    assert abs(aux) <= 1.0
