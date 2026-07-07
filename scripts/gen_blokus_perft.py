"""Generate the checked-in perft fixture (perft(1)/(2) by opening + perft(3) golden).

perft(2) reply counts come from the exhaustive oracle, one move-gen per opening
(the doc's reference path). The [F5] start-square shortcut (a P2 reply must
cover the other start square and not overlap) is asserted equivalent to the
exhaustive oracle on a sampled subset, and its counts are asserted equal on all
828 openings. perft(3) is computed by the bitboard engine (the oracle is
infeasible at depth 3 in Python) with Klein-4 orbit reduction; the method is
recorded as provenance in the fixture, and the test battery differentially
spot-checks perft(2)-frontier states with the oracle.

Deterministic: re-running on unchanged code must be byte-identical.

Usage:
    python3 scripts/gen_blokus_perft.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from games.blokus_duo.actions import (  # noqa: E402
    FIXTURE_CONVENTIONS,
    OPENING_ACTIONS,
    START_SQUARES,
    action_cells,
    decode,
)
from games.blokus_duo.bitboard import (  # noqa: E402
    PIECE_ACTION_TABLES,
    BitboardEngine,
    cells_to_bb,
)
from games.blokus_duo.oracle import OracleEngine  # noqa: E402
from games.blokus_duo.pieces import ORIENTATION_PIECE, orientation_table_hash  # noqa: E402
from games.blokus_duo.symmetry import build_action_maps  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "blokus"


def _other_square(a: int) -> tuple[int, int]:
    (sq,) = set(action_cells(a)) & set(START_SQUARES)
    return START_SQUARES[1 - START_SQUARES.index(sq)]


def _shortcut_replies(a: int) -> list[int]:
    """Candidate P2 replies to opening ``a``: cover the other square, no overlap."""
    cells_a = set(action_cells(a))
    return sorted(
        b for b in OPENING_ACTIONS[_other_square(a)] if not cells_a & set(action_cells(b))
    )


def _count_ply3(a: int, replies: list[int]) -> int:
    """Count P1's legal moves summed over all P2 replies to opening ``a`` (bitboard)."""
    own = cells_to_bb(action_cells(a))
    inv = [p for p in range(21) if p != ORIENTATION_PIECE[decode(a)[2]]]
    total = 0
    for b in replies:
        occ = own | cells_to_bb(action_cells(b))
        for piece in inv:
            for _, placement, ortho, diag in PIECE_ACTION_TABLES[piece]:
                if placement & occ == 0 and ortho & own == 0 and diag & own:
                    total += 1
    return total


def main() -> None:
    """Build the perft fixture and write it as canonical JSON."""
    oracle = OracleEngine()
    init = oracle.initial_state()
    openings = sorted(set(OPENING_ACTIONS[(4, 4)]) | set(OPENING_ACTIONS[(9, 9)]))
    assert len(openings) == 828

    # perft(2): exhaustive oracle move-gen per opening.
    reply_counts: dict[int, int] = {}
    for a in openings:
        s = oracle.place(init, a)
        reply_counts[a] = len(oracle.legal_actions(s, 1))
    print(f"perft(2) = {sum(reply_counts.values())}")

    # [F5] shortcut equivalence: exact sets on a sample, counts on all 828.
    for a in random.Random(0).sample(openings, 12):
        s = oracle.place(init, a)
        assert oracle.legal_actions(s, 1) == _shortcut_replies(a), a
    for a in openings:
        assert len(_shortcut_replies(a)) == reply_counts[a], a

    # perft(3): bitboard, Klein-4 orbit-reduced (equivariance gives equal
    # subtree counts across an orbit; representatives computed exhaustively).
    maps = build_action_maps()
    bitboard = BitboardEngine()
    n2: dict[int, int] = {}
    for a in openings:
        orbit = sorted({maps[g][a] for g in maps})
        rep = orbit[0]
        if rep not in n2:
            sb = bitboard.place(bitboard.initial_state(), rep)
            replies = bitboard.legal_actions(sb, 1)
            assert len(replies) == reply_counts[rep], rep
            n2[rep] = _count_ply3(rep, replies)
        if a != rep:
            n2[a] = n2[rep]
    perft3_total = sum(n2.values())
    print(f"perft(3) = {perft3_total}")

    payload = {
        "orientation_hash": orientation_table_hash(),
        "conventions": FIXTURE_CONVENTIONS,
        "perft1": len(openings),
        "reply_counts": {str(a): reply_counts[a] for a in openings},
        "perft2_total": sum(reply_counts.values()),
        "perft3_by_opening": {str(a): n2[a] for a in openings},
        "perft3_total": perft3_total,
        "provenance": {
            "perft2": "oracle, exhaustive move-gen per opening",
            "perft3": "bitboard count, Klein-4 orbit-reduced",
            "shortcut_check": (
                "oracle replies == other-square candidates minus overlaps on 12 sampled "
                "openings (seed 0); counts equal on all 828"
            ),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "perft.json"
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
