"""Slow cell-grid reference engine for Blokus Duo (design doc §12 M1, oracle first).

Deliberately naive and exhaustive: every legality decision is made by scanning
sets of ``(row, col)`` cells. Independence contract: this module shares only the
base-piece data and the action-encoding surface with the bitboard engine — it
runs its **own** D4 transforms (different rotation/reflection code from
``pieces.py``), its own legality scan, and its own scoring.

State layout (shared tuple convention across engines and the adapter):
``(occ0, occ1, inv0, inv1, mono_last0, mono_last1, to_play, terminal)`` where
``occ`` are frozensets of cells, ``inv`` frozensets of piece indices. Engines
never touch ``to_play``/``terminal`` beyond carrying them: the adapter owns the
pass-invariant normalization (§6.1).
"""

from __future__ import annotations

from games.blokus_duo.actions import (
    BOARD_SIZE,
    START_SQUARES,
    action_cells,
    decode,
    encode_cells,
)
from games.blokus_duo.pieces import BASE_PIECES, ORIENTATION_PIECE

# The size-1 piece sorts first (§5.1 piece order), so the monomino is index 0.
MONOMINO = 0

_ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))


class OracleEngine:
    """Exhaustive cell-grid rules engine; the reference the bitboard is fuzzed against."""

    def __init__(self):
        self._piece_shapes = [self._own_orientations(p) for p in BASE_PIECES]

    @staticmethod
    def _own_orientations(cells) -> list[frozenset[tuple[int, int]]]:
        """Generate the piece's distinct orientations with the oracle's own transforms.

        Uses a different rotation direction and reflection axis from
        ``pieces.d4_orientations`` — same D4 orbit, independent code path.

        Args:
            cells: The piece's canonical cell tuple.

        Returns:
            The distinct origin-normalized orientations as frozensets.
        """
        shapes: list[frozenset[tuple[int, int]]] = []
        seen = set()
        for flip in (False, True):
            pts = [(-r, c) for r, c in cells] if flip else list(cells)
            for _ in range(4):
                pts = [(-c, r) for r, c in pts]
                mr = min(r for r, _ in pts)
                mc = min(c for _, c in pts)
                shape = frozenset((r - mr, c - mc) for r, c in pts)
                if shape not in seen:
                    seen.add(shape)
                    shapes.append(shape)
        return shapes

    def initial_state(self):
        """Return the Blokus Duo start state (empty board, full inventories, P1 to move)."""
        full = frozenset(range(len(BASE_PIECES)))
        return (frozenset(), frozenset(), full, full, False, False, 0, False)

    def legal_actions(self, state, player: int) -> list[int]:
        """Return the sorted legal action ids for ``player`` at ``state``.

        Opening (own occupancy empty): the placement must cover a start square
        the opponent has not covered (§4: P1 either, P2 the other) and overlap
        nothing. Post-opening: piece available, in-bounds, no overlap, no own
        edge contact, at least one own corner contact; opponent contact is free.

        Args:
            state: Engine state tuple.
            player: 0 or 1 (independent of ``state[6]``).

        Returns:
            Sorted list of legal flat action ids (possibly empty).
        """
        own = state[player]
        opp = state[1 - player]
        occupied = own | opp
        opening = not own
        targets = [sq for sq in START_SQUARES if sq not in opp] if opening else ()
        out = []
        for piece in state[2 + player]:
            for shape in self._piece_shapes[piece]:
                h = 1 + max(r for r, _ in shape)
                w = 1 + max(c for _, c in shape)
                for ar in range(BOARD_SIZE - h + 1):
                    for ac in range(BOARD_SIZE - w + 1):
                        cells = {(ar + r, ac + c) for r, c in shape}
                        if cells & occupied:
                            continue
                        if opening:
                            if not any(sq in cells for sq in targets):
                                continue
                        else:
                            if any((r + dr, c + dc) in own for r, c in cells for dr, dc in _ORTH):
                                continue
                            if not any(
                                (r + dr, c + dc) in own for r, c in cells for dr, dc in _DIAG
                            ):
                                continue
                        out.append(encode_cells(cells))
        return sorted(out)

    def place(self, state, action: int):
        """Apply ``action`` for the mover ``state[6]``; no to_play/terminal normalization.

        The monomino-last flag is set iff the placed piece is the monomino AND
        the inventory empties on this placement (§4 scoring-state caveat); it
        can never be unset afterwards because an empty inventory allows no
        further placements.

        Args:
            state: Engine state tuple.
            action: A legal flat action id for the mover.

        Returns:
            The successor state tuple, with ``to_play`` and ``terminal``
            carried unchanged (the adapter normalizes them).
        """
        player = state[6]
        piece = ORIENTATION_PIECE[decode(action)[2]]
        own = state[player] | frozenset(action_cells(action))
        inv = state[2 + player] - {piece}
        mono_last = piece == MONOMINO and not inv
        parts = list(state)
        parts[player] = own
        parts[2 + player] = inv
        parts[4 + player] = mono_last
        return tuple(parts)

    def scores(self, state) -> tuple[int, int]:
        """Return the official scores ``(score_p0, score_p1)`` (§4).

        −1 per unplaced square; +15 if all 21 pieces placed; +5 more if the
        monomino-last flag is set (only reachable with all placed).

        Args:
            state: Engine state tuple.

        Returns:
            Per-player integer scores.
        """
        out = []
        for p in (0, 1):
            inv = state[2 + p]
            if inv:
                out.append(-sum(len(BASE_PIECES[i]) for i in inv))
            else:
                out.append(20 if state[4 + p] else 15)
        return tuple(out)
