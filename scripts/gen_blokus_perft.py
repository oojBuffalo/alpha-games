"""Generate the checked-in perft fixtures, one per pinned instance (§5.1, §5.3).

Writes ``tests/fixtures/blokus/perft.json`` (full 14×14 game) and
``tests/fixtures/blokus_micro/perft.json`` (the §5.3 micro instance), each with
**its own** orientation-table hash and encoding conventions embedded (micro ids
are re-derived within the piece subset, so its digest differs by construction).

perft(2) reply counts come from the exhaustive oracle, one move-gen per opening
(the doc's reference path). The [F5] start-square shortcut (a P2 reply must
cover the other start square and not overlap) is asserted equivalent to the
exhaustive oracle on a sampled subset, and its counts are asserted equal on
every opening. perft(3) is computed by the bitboard engine (the oracle is
infeasible at depth 3 in Python at 14×14) with Klein-4 orbit reduction; the
method is recorded as provenance in the fixture, and the test battery
differentially spot-checks perft(2)-frontier states with the oracle.

perft(3) fixes the ply-3 mover to P1 (raw alternation) — the frozen full-game
convention. The micro instance is small enough to afford the *pass-aware* truth
as well: its fixture carries an extra ``game_tree`` block enumerating the
complete micro game tree through the adapter (which realizes forced passes by
skipping the blocked player, §6.1). The two conventions disagree at micro scale
— there are ply-3 positions where P1 is already blocked and P2 moves again — so
freezing both pins the pass normalization itself, not just move generation.

Deterministic: re-running on unchanged code must be byte-identical.

Usage:
    python3 scripts/gen_blokus_perft.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from games.blokus_duo.actions import ActionCodec, action_codec  # noqa: E402
from games.blokus_duo.bitboard import (  # noqa: E402
    BitboardEngine,
    build_piece_action_tables,
    cells_to_bb,
)
from games.blokus_duo.config import FULL_CONFIG, MICRO_CONFIG, BlokusConfig  # noqa: E402
from games.blokus_duo.game import BlokusDuo  # noqa: E402
from games.blokus_duo.oracle import OracleEngine  # noqa: E402
from games.blokus_duo.pieces import build_pieces, orientation_table_hash  # noqa: E402
from games.blokus_duo.symmetry import symmetry_group  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Openings sampled for the [F5] shortcut set-equality check (seed 0).
SHORTCUT_SAMPLE = 12


class Instance(NamedTuple):
    """One pinned instance's generation spec.

    Attributes:
        name: Fixture directory name under ``tests/fixtures``.
        config: The Blokus instance to enumerate.
        openings: Expected opening count — a golden guard, so a config or
            encoding regression fails here instead of silently rewriting the
            fixture (828 full, 42 micro).
        game_tree: Whether to enumerate the complete pass-aware game tree;
            affordable only on the micro board.
    """

    name: str
    config: BlokusConfig
    openings: int
    game_tree: bool


INSTANCES: tuple[Instance, ...] = (
    Instance("blokus", FULL_CONFIG, 828, game_tree=False),
    Instance("blokus_micro", MICRO_CONFIG, 42, game_tree=True),
)


def opening_actions(codec: ActionCodec) -> list[int]:
    """Return every legal opening action id of ``codec``'s instance, sorted.

    Args:
        codec: The instance's action codec.

    Returns:
        The sorted union of the per-start-square opening ids.
    """
    ids: set[int] = set()
    for actions in codec.opening_actions.values():
        ids.update(actions)
    return sorted(ids)


def _other_square(codec: ActionCodec, action: int) -> tuple[int, int]:
    """Return the start square that opening ``action`` does *not* cover.

    Args:
        codec: The instance's action codec.
        action: An opening action id (covers exactly one start square).

    Returns:
        The other start square.
    """
    (sq,) = set(codec.action_cells(action)) & set(codec.start_squares)
    return codec.start_squares[1 - codec.start_squares.index(sq)]


def _shortcut_replies(codec: ActionCodec, action: int) -> list[int]:
    """Return the [F5] candidate P2 replies to ``action``.

    A reply must cover the other start square and not overlap the opening.

    Args:
        codec: The instance's action codec.
        action: An opening action id.

    Returns:
        The sorted candidate reply ids.
    """
    cells = set(codec.action_cells(action))
    return sorted(
        b
        for b in codec.opening_actions[_other_square(codec, action)]
        if not cells & set(codec.action_cells(b))
    )


def _count_ply3(config: BlokusConfig, action: int, replies: list[int]) -> int:
    """Count P1's legal moves summed over all P2 replies to opening ``action``.

    Bitboard legality inline (piece available ∧ no overlap ∧ no own edge contact
    ∧ some own corner contact), with the ply-3 mover fixed to P1.

    Args:
        config: The instance config.
        action: P1's opening action id.
        replies: P2's legal replies to ``action``.

    Returns:
        The number of ply-3 P1 placements over the whole reply fan.
    """
    codec = action_codec(config)
    tables = build_piece_action_tables(config)
    own = cells_to_bb(codec.action_cells(action), config.board_size)
    played = codec.action_piece(action)
    inv = [p for p in range(len(build_pieces(config)[0])) if p != played]
    total = 0
    for b in replies:
        occ = own | cells_to_bb(codec.action_cells(b), config.board_size)
        for piece in inv:
            for _, placement, ortho, diag in tables[piece]:
                if placement & occ == 0 and ortho & own == 0 and diag & own:
                    total += 1
    return total


def _game_tree(config: BlokusConfig) -> dict:
    """Enumerate the complete pass-aware game tree of ``config``.

    Walks every line through the :class:`~games.blokus_duo.game.BlokusDuo`
    adapter, so forced passes are realized as the §6.1 skip rather than assumed
    away — the honest game-tree perft, unlike the raw-alternation ``perft3``.

    Args:
        config: The instance config (micro-sized only; the full game's tree is
            astronomically large).

    Returns:
        Dict with ``nodes_by_ply`` (moves played at each ply over the whole
        tree; entry 0 is perft(1), entry 1 perft(2), …), ``total_nodes``,
        ``complete_games`` (leaf count) and ``leaves_by_opening``.
    """
    game = BlokusDuo(BitboardEngine(config))
    nodes_by_ply: list[int] = []

    def walk(state, ply: int) -> int:
        """Return the number of complete games below ``state``, tallying nodes."""
        if game.is_terminal(state):
            return 1
        moves = game.legal_moves(state)
        while len(nodes_by_ply) <= ply:
            nodes_by_ply.append(0)
        nodes_by_ply[ply] += len(moves)
        return sum(walk(game.apply(state, a), ply + 1) for a in moves)

    root = game.initial_state()
    openings = game.legal_moves(root)
    nodes_by_ply.append(len(openings))
    leaves = {a: walk(game.apply(root, a), 1) for a in openings}
    return {
        "nodes_by_ply": nodes_by_ply,
        "total_nodes": sum(nodes_by_ply),
        "complete_games": sum(leaves.values()),
        "leaves_by_opening": {str(a): leaves[a] for a in openings},
    }


def build_payload(spec: Instance) -> dict:
    """Build one instance's perft fixture payload.

    Args:
        spec: The instance to enumerate.

    Returns:
        The fixture dict: orientation hash, encoding conventions, perft(1),
        per-opening perft(2) reply counts and total, per-opening perft(3) and
        total, provenance, and — for small instances — the complete pass-aware
        ``game_tree`` block.

    Raises:
        AssertionError: If the opening count, the [F5] shortcut equivalence or
            the bitboard/oracle reply-count agreement fails.
    """
    config = spec.config
    codec = action_codec(config)
    oracle = OracleEngine(config)
    init = oracle.initial_state()
    openings = opening_actions(codec)
    assert len(openings) == spec.openings, (spec.name, len(openings))

    # perft(2): exhaustive oracle move-gen per opening.
    reply_counts = {a: len(oracle.legal_actions(oracle.place(init, a), 1)) for a in openings}
    print(f"[{spec.name}] perft(2) = {sum(reply_counts.values())}")

    # [F5] shortcut equivalence: exact sets on a sample, counts on every opening.
    sample = random.Random(0).sample(openings, min(SHORTCUT_SAMPLE, len(openings)))
    for a in sample:
        assert oracle.legal_actions(oracle.place(init, a), 1) == _shortcut_replies(codec, a), a
    for a in openings:
        assert len(_shortcut_replies(codec, a)) == reply_counts[a], a

    # perft(3): bitboard, Klein-4 orbit-reduced (equivariance gives equal
    # subtree counts across an orbit; representatives computed exhaustively).
    maps = symmetry_group(config).action_maps()
    bitboard = BitboardEngine(config)
    n2: dict[int, int] = {}
    for a in openings:
        orbit = sorted({maps[g][a] for g in maps})
        rep = orbit[0]
        if rep not in n2:
            sb = bitboard.place(bitboard.initial_state(), rep)
            replies = bitboard.legal_actions(sb, 1)
            assert len(replies) == reply_counts[rep], rep
            n2[rep] = _count_ply3(config, rep, replies)
        if a != rep:
            n2[a] = n2[rep]
    perft3_total = sum(n2.values())
    print(f"[{spec.name}] perft(3) = {perft3_total}")

    payload = {
        "orientation_hash": orientation_table_hash(config),
        "conventions": codec.fixture_conventions,
        "perft1": len(openings),
        "reply_counts": {str(a): reply_counts[a] for a in openings},
        "perft2_total": sum(reply_counts.values()),
        "perft3_by_opening": {str(a): n2[a] for a in openings},
        "perft3_total": perft3_total,
        "provenance": {
            "perft2": "oracle, exhaustive move-gen per opening",
            "perft3": "bitboard count, Klein-4 orbit-reduced",
            "shortcut_check": (
                f"oracle replies == other-square candidates minus overlaps on {len(sample)} "
                f"sampled openings (seed 0); counts equal on all {len(openings)}"
            ),
        },
    }
    if spec.game_tree:
        payload["game_tree"] = _game_tree(config)
        payload["provenance"]["game_tree"] = (
            "bitboard, complete pass-aware tree through the BlokusDuo adapter "
            "(forced passes skip the blocked player, §6.1); nodes_by_ply[i] counts "
            "the moves played at ply i, so [0] is perft(1) and [1] is perft(2). "
            "perft3_by_opening above fixes the ply-3 mover to P1 (raw alternation) "
            "and is therefore <= nodes_by_ply[2]"
        )
        print(f"[{spec.name}] game tree = {payload['game_tree']['nodes_by_ply']}")
    return payload


def write_fixture(spec: Instance, out_dir: Path) -> Path:
    """Write one instance's perft fixture as canonical JSON.

    Args:
        spec: The instance to enumerate.
        out_dir: Directory to write ``perft.json`` into (created if missing).

    Returns:
        The path written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "perft.json"
    path.write_text(json.dumps(build_payload(spec), sort_keys=True, separators=(",", ":")) + "\n")
    return path


def main(root: Path = ROOT) -> None:
    """Write every pinned instance's perft fixture.

    Args:
        root: Repo root to write under; overridden by the byte-stability test,
            which regenerates into a temp tree and diffs against the committed
            fixtures.
    """
    for spec in INSTANCES:
        path = write_fixture(spec, root / "tests" / "fixtures" / spec.name)
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
