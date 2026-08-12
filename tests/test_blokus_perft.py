"""Perft fixtures per instance: perft(1)/(2) by opening, orbit constancy, deep goldens.

The M1 perft battery with an M2.5 config axis: every check below runs for the
full 14×14 game and for the §5.3 micro instance, against that instance's **own**
fixture (``tests/fixtures/blokus/perft.json`` and
``tests/fixtures/blokus_micro/perft.json``) carrying its own orientation hash —
micro ids are re-derived within the piece subset (invariant 4), never the full
table restricted.

perft(2) reply counts are oracle-generated (exhaustive move-gen per opening);
perft(3) is bitboard-generated with Klein-4 orbit reduction — provenance is
recorded in the fixture, and the oracle differentially spot-checks sampled
perft(2)-frontier states here [F2]. The [F5] shortcut (P2's reply must cover the
other start square) is asserted equivalent to the exhaustive oracle both in the
generator and on a sample below.

Micro-only, because the board is small enough to afford it: the fixture also
freezes the **complete pass-aware game tree** (nodes per ply, complete games,
leaves per opening) walked through the adapter, and the battery re-derives it in
lockstep through *both* engines. That is the strongest differential statement
available — the two independent rules engines agree on every legal-move list in
every reachable micro position — and it pins the §6.1 forced-pass normalization,
which the raw-alternation perft(3) convention cannot see.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from functools import cache
from pathlib import Path
from typing import NamedTuple

import pytest

from games.blokus_duo.actions import action_codec
from games.blokus_duo.bitboard import BitboardEngine
from games.blokus_duo.config import FULL_CONFIG, MICRO_CONFIG, BlokusConfig
from games.blokus_duo.game import BlokusDuo
from games.blokus_duo.oracle import OracleEngine
from games.blokus_duo.pieces import orientation_table_hash

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
GEN_SCRIPT = ROOT / "scripts" / "gen_blokus_perft.py"


class Case(NamedTuple):
    """One instance's perft-battery parameters.

    Attributes:
        config: The Blokus instance under test.
        dirname: Its fixture directory under ``tests/fixtures``.
        flatten: The pinned action-flatten convention string.
        openings: Golden opening count (828 full, 42 micro).
        oracle_sample: Openings to recount with the (slow) oracle.
        bitboard_sample: Openings to recount with the bitboard engine.
    """

    config: BlokusConfig
    dirname: str
    flatten: str
    openings: int
    oracle_sample: int
    bitboard_sample: int


CASES: dict[str, Case] = {
    "full": Case(FULL_CONFIG, "blokus", "(r*14+c)*91+o", 828, 4, 24),
    # Micro is cheap enough to recount every opening with either engine.
    "micro": Case(MICRO_CONFIG, "blokus_micro", "(r*5+c)*9+o", 42, 42, 42),
}
INSTANCES = tuple(CASES)


@cache
def _perft(instance: str) -> dict:
    """Load and validate ``instance``'s perft fixture (hash + conventions).

    Args:
        instance: Key into :data:`CASES`.

    Returns:
        The parsed fixture payload.
    """
    case = CASES[instance]
    data = json.loads((FIXTURES / case.dirname / "perft.json").read_text())
    assert data["orientation_hash"] == orientation_table_hash(case.config)
    assert data["conventions"]["flatten"] == case.flatten
    return data


@cache
def _symmetry_maps(instance: str) -> dict[str, dict[int, int]]:
    """Load ``instance``'s (g,a)→a′ fixture as per-element dicts.

    Args:
        instance: Key into :data:`CASES`.

    Returns:
        Per group element, the in-bounds action map.
    """
    data = json.loads((FIXTURES / CASES[instance].dirname / "symmetry_table.json").read_text())
    actions = data["actions"]
    return {g: dict(zip(actions, m, strict=True)) for g, m in data["maps"].items()}


@cache
def _engines(instance: str) -> tuple[OracleEngine, BitboardEngine]:
    """Return ``instance``'s ``(oracle, bitboard)`` engine pair."""
    config = CASES[instance].config
    return OracleEngine(config), BitboardEngine(config)


@cache
def _openings(instance: str) -> tuple[int, ...]:
    """Return ``instance``'s sorted legal opening action ids."""
    codec = action_codec(CASES[instance].config)
    ids: set[int] = set()
    for actions in codec.opening_actions.values():
        ids.update(actions)
    return tuple(sorted(ids))


def _reply_counts(instance: str) -> dict[int, int]:
    """Return the fixture's per-opening perft(2) reply counts, keyed by int id."""
    return {int(a): n for a, n in _perft(instance)["reply_counts"].items()}


def _n2(instance: str) -> dict[int, int]:
    """Return the fixture's per-opening perft(3) counts, keyed by int id."""
    return {int(a): n for a, n in _perft(instance)["perft3_by_opening"].items()}


def _other_square(instance: str, a: int) -> tuple[int, int]:
    """Return the start square opening ``a`` does not cover."""
    codec = action_codec(CASES[instance].config)
    (sq,) = set(codec.action_cells(a)) & set(codec.start_squares)
    return codec.start_squares[1 - codec.start_squares.index(sq)]


def _place_opening(engine, a):
    """Place opening ``a`` on ``engine``'s initial state."""
    return engine.place(engine.initial_state(), a)


# --- golden counts and orbit constancy (both instances) ----------------------------


@pytest.mark.parametrize("instance", INSTANCES)
def test_perft1_matches_the_opening_golden(instance):
    case = CASES[instance]
    _, bitboard = _engines(instance)
    assert _perft(instance)["perft1"] == case.openings
    assert sorted(_reply_counts(instance)) == list(_openings(instance))
    assert len(bitboard.legal_actions(bitboard.initial_state(), 0)) == case.openings


@pytest.mark.parametrize("instance", INSTANCES)
def test_perft2_total_and_orbit_constancy(instance):
    counts = _reply_counts(instance)
    assert sum(counts.values()) == _perft(instance)["perft2_total"]
    for g, table in _symmetry_maps(instance).items():
        for a in _openings(instance):
            assert counts[a] == counts[table[a]], (instance, g, a)


@pytest.mark.parametrize("instance", INSTANCES)
def test_perft3_total_and_orbit_constancy(instance):
    n2 = _n2(instance)
    assert sorted(n2) == list(_openings(instance))
    assert sum(n2.values()) == _perft(instance)["perft3_total"]
    for g, table in _symmetry_maps(instance).items():
        for a in _openings(instance):
            assert n2[a] == n2[table[a]], (instance, g, a)


# --- differential recounts against the fixture -------------------------------------


@pytest.mark.parametrize("instance", INSTANCES)
def test_oracle_reply_counts_sample_vs_fixture(instance):
    counts = _reply_counts(instance)
    oracle, _ = _engines(instance)
    openings = _openings(instance)
    for a in random.Random(5).sample(openings, CASES[instance].oracle_sample):
        assert len(oracle.legal_actions(_place_opening(oracle, a), 1)) == counts[a]


@pytest.mark.parametrize("instance", INSTANCES)
def test_bitboard_reply_counts_sample_vs_fixture(instance):
    counts = _reply_counts(instance)
    _, bitboard = _engines(instance)
    openings = _openings(instance)
    for a in random.Random(6).sample(openings, CASES[instance].bitboard_sample):
        assert len(bitboard.legal_actions(_place_opening(bitboard, a), 1)) == counts[a]


@pytest.mark.slow
@pytest.mark.parametrize("instance", INSTANCES)
def test_bitboard_reply_counts_all_openings(instance):
    counts = _reply_counts(instance)
    _, bitboard = _engines(instance)
    for a in _openings(instance):
        assert len(bitboard.legal_actions(_place_opening(bitboard, a), 1)) == counts[a]


@pytest.mark.parametrize("instance", INSTANCES)
def test_shortcut_equals_exhaustive_oracle_on_sample(instance):
    # [F5] The other-start-square shortcut (a reply must cover the square the
    # opening did not, minus overlaps) must equal the oracle's exhaustive
    # move-gen — 414 candidates per square at 14×14, 21 at 5×5.
    codec = action_codec(CASES[instance].config)
    oracle, _ = _engines(instance)
    for a in random.Random(7).sample(_openings(instance), 3):
        cells_a = set(codec.action_cells(a))
        candidates = sorted(
            b
            for b in codec.opening_actions[_other_square(instance, a)]
            if not cells_a & set(codec.action_cells(b))
        )
        assert oracle.legal_actions(_place_opening(oracle, a), 1) == candidates


@pytest.mark.parametrize("instance", INSTANCES)
def test_oracle_spot_checks_perft3_frontier(instance):
    # [F2] Differential oracle recount of P1's moves at sampled perft(2)-
    # frontier states (the bitboard generated perft(3); the oracle audits it).
    oracle, bitboard = _engines(instance)
    rng = random.Random(8)
    for a in rng.sample(_openings(instance), 2):
        so = _place_opening(oracle, a)
        sb = _place_opening(bitboard, a)
        replies = oracle.legal_actions(so, 1)
        for b in rng.sample(replies, 2):
            so2 = oracle.place((*so[:6], 1, False), b)
            sb2 = bitboard.place((*sb[:6], 1, False), b)
            assert oracle.legal_actions(so2, 0) == bitboard.legal_actions(sb2, 0)


@pytest.mark.slow
@pytest.mark.parametrize("instance", INSTANCES)
def test_perft3_recompute_orbit_representative(instance):
    # Recompute N2 for one orbit representative from scratch via the bitboard
    # and compare against the fixture (guards against fixture drift without
    # regenerating the whole table).
    _, bitboard = _engines(instance)
    a = _openings(instance)[0]
    sb = _place_opening(bitboard, a)
    total = 0
    for b in bitboard.legal_actions(sb, 1):
        sb2 = bitboard.place((*sb[:6], 1, False), b)
        total += len(bitboard.legal_actions(sb2, 0))
    assert total == _n2(instance)[a]


# --- micro-only: the complete pass-aware game tree ---------------------------------


def _walk_lockstep(max_ply: int | None = None) -> tuple[list[int], dict[int, int]]:
    """Walk the micro game tree through both engines in lockstep.

    Every node is visited with the oracle and the bitboard engine at once and
    their legal-move lists, terminal flags and scores are compared — a full
    differential over the reachable state space rather than a sampled fuzz.

    Args:
        max_ply: Last ply index to tally; the walk stops descending there
            (``None`` walks the complete tree). Leaf counts are only meaningful
            for a complete walk.

    Returns:
        ``(nodes_by_ply, leaves_by_opening)`` in the fixture's convention:
        ``nodes_by_ply[i]`` counts the moves played at ply ``i``, and the leaf
        map counts complete games below each opening.
    """
    oracle, bitboard = _engines("micro")
    game_o, game_b = BlokusDuo(oracle), BlokusDuo(bitboard)
    nodes_by_ply: list[int] = []

    def walk(so, sb, ply: int) -> int:
        terminal = game_o.is_terminal(so)
        assert terminal == game_b.is_terminal(sb)
        assert so[2:] == sb[2:]  # inventories, flags, to_play, terminal agree
        if terminal:
            assert oracle.scores(so) == bitboard.scores(sb)
            return 1
        moves_o = list(game_o.legal_moves(so))
        moves_b = list(game_b.legal_moves(sb))
        assert moves_o == moves_b, (ply, so[2:])
        assert moves_o  # pass invariant: the mover always has a move
        while len(nodes_by_ply) <= ply:
            nodes_by_ply.append(0)
        nodes_by_ply[ply] += len(moves_o)
        if max_ply is not None and ply >= max_ply:
            return 0
        return sum(walk(game_o.apply(so, a), game_b.apply(sb, a), ply + 1) for a in moves_o)

    root_o, root_b = game_o.initial_state(), game_b.initial_state()
    openings = list(game_o.legal_moves(root_o))
    assert openings == list(game_b.legal_moves(root_b))
    nodes_by_ply.append(len(openings))
    leaves = {a: walk(game_o.apply(root_o, a), game_b.apply(root_b, a), 1) for a in openings}
    return nodes_by_ply, leaves


def test_micro_game_tree_block_is_internally_consistent():
    tree = _perft("micro")["game_tree"]
    nodes = tree["nodes_by_ply"]
    assert nodes[0] == _perft("micro")["perft1"] == 42
    assert nodes[1] == _perft("micro")["perft2_total"]
    assert sum(nodes) == tree["total_nodes"]
    leaves = {int(a): n for a, n in tree["leaves_by_opening"].items()}
    assert sorted(leaves) == list(_openings("micro"))
    assert sum(leaves.values()) == tree["complete_games"]
    # Games run 2..8 plies: both openings, then at most 3 further pieces each.
    assert len(nodes) == 8
    assert all(n > 0 for n in nodes)


def test_micro_game_tree_leaf_counts_are_constant_on_klein4_orbits():
    leaves = {int(a): n for a, n in _perft("micro")["game_tree"]["leaves_by_opening"].items()}
    for g, table in _symmetry_maps("micro").items():
        for a in _openings("micro"):
            assert leaves[a] == leaves[table[a]], (g, a)


def test_micro_forced_passes_make_the_tree_wider_than_raw_alternation():
    # The two conventions must *not* coincide at micro scale: perft3_by_opening
    # fixes the ply-3 mover to P1, while the adapter hands the move back to P2
    # when P1 is already blocked (§6.1). If a pass-normalization regression made
    # the adapter alternate blindly, these would become equal.
    tree = _perft("micro")["game_tree"]
    assert tree["nodes_by_ply"][2] > _perft("micro")["perft3_total"]

    # Locate the ply-3 positions where P1 has nothing: the adapter must keep the
    # game alive with P2 to move again, never mark it terminal.
    oracle, bitboard = _engines("micro")
    game = BlokusDuo(bitboard)
    blocked = 0
    for a in _openings("micro"):
        s1 = game.apply(game.initial_state(), a)
        for b in game.legal_moves(s1):
            s2 = game.apply(s1, b)
            if bitboard.legal_actions(s2, 0):
                continue
            blocked += 1
            assert not game.is_terminal(s2)
            assert game.current_player(s2) == 1  # P2 moves twice in a row
            assert not oracle.legal_actions(_to_oracle(s2), 0)  # oracle concurs
    assert blocked > 0


def _to_oracle(state):
    """Convert a bitboard-layout state to the oracle's frozenset layout.

    Args:
        state: Engine state tuple with occupancies as ``H*W``-bit ints.

    Returns:
        The same state with occupancies as frozensets of ``(row, col)`` cells.
    """
    size = MICRO_CONFIG.board_size
    occs = tuple(
        frozenset(divmod(i, size) for i in range(size * size) if occ >> i & 1) for occ in state[:2]
    )
    return (*occs, *state[2:])


def test_micro_shallow_tree_lockstep_matches_fixture():
    # Fast slice of the exhaustive differential: both engines in lockstep over
    # every position down to depth 2 (the perft(3) frontier), against the
    # frozen tally.
    nodes, _ = _walk_lockstep(max_ply=2)
    assert nodes == _perft("micro")["game_tree"]["nodes_by_ply"][:3]


@pytest.mark.slow
def test_micro_complete_tree_lockstep_matches_fixture():
    # The whole micro game tree — every reachable position, both engines in
    # lockstep on move-gen, apply, terminal detection and scoring.
    nodes, leaves = _walk_lockstep()
    tree = _perft("micro")["game_tree"]
    assert nodes == tree["nodes_by_ply"]
    assert sum(leaves.values()) == tree["complete_games"]
    assert {str(a): n for a, n in leaves.items()} == tree["leaves_by_opening"]


# --- write side: fixture regeneration is byte-stable -------------------------------


def _load_generator():
    """Import ``scripts/gen_blokus_perft.py`` (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("gen_blokus_perft", GEN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_micro_perft_fixture_regenerates_byte_identically(tmp_path):
    # The micro fixture must be reproducible from unchanged code, byte for
    # byte — anything that leaks iteration order into the tables (invariant 4)
    # breaks here instead of silently invalidating every artifact keyed on the
    # orientation hash. The full game's fixture is checked the same way, but
    # offline: its generation is minutes-long (perft(3) over 828 openings), so
    # regenerating it belongs in the fixture-generation workflow, not the
    # battery.
    generator = _load_generator()
    (spec,) = [s for s in generator.INSTANCES if s.name == "blokus_micro"]
    runs = [generator.write_fixture(spec, tmp_path / f"run{i}").read_bytes() for i in range(2)]
    assert runs[0] == runs[1]  # deterministic on unchanged code
    assert runs[0] == (FIXTURES / "blokus_micro" / "perft.json").read_bytes()
