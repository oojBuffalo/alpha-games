"""The canonical artifact fingerprint: ``core/artifact_fingerprint.py`` (§12 M3).

Exercises the module standalone (no shard/checkpoint machinery): the
fingerprint's game-agnostic shape, its stability, and every field the reader
must fail loudly on when it disagrees.
"""

from __future__ import annotations

import copy
import json

import pytest

from core.artifact_fingerprint import (
    SCHEMA_VERSION,
    FingerprintMismatchError,
    build_fingerprint,
    canonical_json,
    compare_fingerprints,
    fingerprint_digest,
)
from games.blokus_duo import BlokusDuo
from games.blokus_duo.config import MICRO_CONFIG
from games.connect4 import Connect4
from games.tictactoe import TicTacToe

FULL = BlokusDuo()
MICRO = BlokusDuo(config=MICRO_CONFIG)
TTT = TicTacToe()
C4 = Connect4()


def test_fingerprint_has_the_six_declared_fields_plus_schema_version():
    fp = build_fingerprint(FULL)
    assert set(fp) == {
        "schema_version",
        "game_identity",
        "policy_shape",
        "input_planes",
        "input_shape",
        "encoding_conventions",
        "value_target_scaling",
        "orientation_hash",
    }
    assert fp["schema_version"] == SCHEMA_VERSION


def test_fingerprint_is_json_round_trip_idempotent():
    # build_fingerprint must emit only JSON-native types (never a tuple) so a
    # fingerprint read back from disk compares equal to a freshly-built one.
    fp = build_fingerprint(FULL)
    assert json.loads(canonical_json(fp)) == fp


def test_fingerprint_matches_the_pinned_micro_orientation_hash():
    # design doc §5.3's pinned micro orientation-table hash.
    fp = build_fingerprint(MICRO)
    expected = "78ea621ae2d1e27e239ecffa5ff44c793ef15f2884198a0394d394083d3e37e4"
    assert fp["orientation_hash"] == expected


def test_games_without_an_orientation_table_get_none():
    assert build_fingerprint(TTT)["orientation_hash"] is None
    assert build_fingerprint(C4)["orientation_hash"] is None


def test_games_without_declared_aux_heads_have_empty_value_target_scaling():
    scaling = build_fingerprint(TTT)["value_target_scaling"]
    assert scaling["aux_names"] == []
    assert scaling["aux_loss_weights"] == []


def test_games_without_a_config_attribute_get_none_game_identity_config():
    assert build_fingerprint(TTT)["game_identity"]["config"] is None
    assert build_fingerprint(C4)["game_identity"]["config"] is None


def test_micro_and_full_blokus_have_distinct_fingerprints_on_every_shape_field():
    full_fp = build_fingerprint(FULL)
    micro_fp = build_fingerprint(MICRO)
    assert full_fp["policy_shape"] != micro_fp["policy_shape"]
    assert full_fp["input_shape"] != micro_fp["input_shape"]
    assert full_fp["input_planes"] != micro_fp["input_planes"]
    assert full_fp["orientation_hash"] != micro_fp["orientation_hash"]
    assert full_fp["game_identity"] != micro_fp["game_identity"]


def test_fingerprint_is_stable_across_repeated_calls_and_fresh_instances():
    # Proxy for "same adapter -> identical fingerprint across processes": a
    # brand-new BlokusDuo(config=MICRO_CONFIG) instance, built independently,
    # must fingerprint identically to another one -- no dependence on
    # construction order, instance identity, or set-iteration order.
    a = build_fingerprint(BlokusDuo(config=MICRO_CONFIG))
    b = build_fingerprint(BlokusDuo(config=MICRO_CONFIG))
    assert a == b
    assert canonical_json(a) == canonical_json(b)
    assert fingerprint_digest(a) == fingerprint_digest(b)


def test_canonical_json_does_not_depend_on_key_construction_order():
    fp = build_fingerprint(MICRO)
    reordered = dict(reversed(list(fp.items())))
    assert reordered != list(fp.items())  # sanity: the raw items really differ in order
    assert canonical_json(fp) == canonical_json(reordered)


def test_compare_fingerprints_accepts_an_exact_match():
    fp = build_fingerprint(MICRO)
    compare_fingerprints(fp, fp)  # must not raise


def test_compare_fingerprints_rejects_mismatched_orientation_hash():
    stored = build_fingerprint(MICRO)
    live = build_fingerprint(FULL)
    with pytest.raises(FingerprintMismatchError, match="orientation_hash"):
        compare_fingerprints(stored, live)


def test_compare_fingerprints_rejects_mismatched_policy_shape():
    stored = build_fingerprint(TTT)
    live = build_fingerprint(C4)
    with pytest.raises(FingerprintMismatchError, match="policy_shape"):
        compare_fingerprints(stored, live)


def test_compare_fingerprints_rejects_mismatched_schema_version():
    stored = copy.deepcopy(build_fingerprint(MICRO))
    stored["schema_version"] = SCHEMA_VERSION + 1
    live = build_fingerprint(MICRO)
    with pytest.raises(FingerprintMismatchError, match="schema_version"):
        compare_fingerprints(stored, live)


def test_compare_fingerprints_rejects_an_unknown_older_schema_version_too():
    stored = copy.deepcopy(build_fingerprint(MICRO))
    stored["schema_version"] = 0
    live = build_fingerprint(MICRO)
    with pytest.raises(FingerprintMismatchError, match="schema_version"):
        compare_fingerprints(stored, live)


def test_compare_fingerprints_rejects_mismatched_value_target_scaling():
    stored = copy.deepcopy(build_fingerprint(FULL))
    stored["value_target_scaling"]["aux_loss_weights"] = [0.5]
    live = build_fingerprint(FULL)
    with pytest.raises(FingerprintMismatchError, match="value_target_scaling"):
        compare_fingerprints(stored, live)


def test_compare_fingerprints_names_every_mismatched_field_at_once():
    stored = copy.deepcopy(build_fingerprint(MICRO))
    stored["orientation_hash"] = "not-a-real-hash"
    stored["input_planes"] = 999
    live = build_fingerprint(MICRO)
    with pytest.raises(FingerprintMismatchError) as excinfo:
        compare_fingerprints(stored, live)
    message = str(excinfo.value)
    assert "orientation_hash" in message
    assert "input_planes" in message
