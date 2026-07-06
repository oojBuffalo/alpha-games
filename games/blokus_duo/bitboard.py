"""Bitboard rules engine: 196-bit ints, precomputed per-action masks (design doc §6.3).

Bit index = ``r*14 + c`` (the §5.1 cell flatten). Per in-bounds action, three
precomputed masks: ``placement_bb`` (the piece cells), ``ortho_halo_bb``
(orthogonal neighbors, in-bounds, excluding placement cells), ``diag_halo_bb``
(diagonal neighbors, likewise). Post-opening legality is four integer ops:
``piece available ∧ placement & occ == 0 ∧ ortho & own == 0 ∧ diag & own != 0``;
the opening substitutes start-square coverage for the diagonal-contact clause
(orthogonal contact is vacuous on an empty own board, but the mask check is
kept — it is free and identical).

Independence contract: shares only piece data and the action-encoding surface
with the oracle; masks are derived from ``action_cells`` (the golden-tested
decode side), not from the oracle's transforms. State layout matches the shared
tuple convention with occupancies as ints instead of frozensets.
"""

from __future__ import annotations

from games.blokus_duo.actions import (
    BOARD_SIZE,
    IN_BOUNDS_ACTIONS,
    START_SQUARES,
    action_cells,
    decode,
)
from games.blokus_duo.pieces import BASE_PIECES, ORIENTATION_PIECE

MONOMINO = 0

_ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def cells_to_bb(cells) -> int:
    """Pack ``(row, col)`` cells into a 196-bit occupancy int (bit = r*14+c).

    Args:
        cells: Iterable of on-board ``(row, col)`` tuples.

    Returns:
        The occupancy bitboard.
    """
    bb = 0
    for r, c in cells:
        bb |= 1 << (r * BOARD_SIZE + c)
    return bb


def _halo(cells, offsets) -> int:
    placement = set(cells)
    halo = 0
    for r, c in cells:
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and (nr, nc) not in placement:
                halo |= 1 << (nr * BOARD_SIZE + nc)
    return halo


def _build_tables():
    per_piece: list[list[tuple[int, int, int, int]]] = [[] for _ in BASE_PIECES]
    for a in IN_BOUNDS_ACTIONS:
        cells = action_cells(a)
        piece = ORIENTATION_PIECE[decode(a)[2]]
        per_piece[piece].append(
            (a, cells_to_bb(cells), _halo(cells, _ORTH), _halo(cells, _DIAG))
        )
    return tuple(tuple(rows) for rows in per_piece)


# Per piece: (action_id, placement_bb, ortho_halo_bb, diag_halo_bb) for every
# in-bounds placement of that piece — availability prunes whole blocks at once.
PIECE_ACTION_TABLES = _build_tables()

_START_BITS = {sq: 1 << (sq[0] * BOARD_SIZE + sq[1]) for sq in START_SQUARES}


class BitboardEngine:
    """Mask-based rules engine over the shared state tuple (occupancies as ints)."""

    def initial_state(self):
        """Return the start state (empty boards, full inventories, P1 to move)."""
        full = frozenset(range(len(BASE_PIECES)))
        return (0, 0, full, full, False, False, 0, False)

    def legal_actions(self, state, player: int) -> list[int]:
        """Return the sorted legal action ids for ``player`` at ``state``.

        Args:
            state: Engine state tuple (occupancies as ints).
            player: 0 or 1 (independent of ``state[6]``).

        Returns:
            Sorted list of legal flat action ids (possibly empty).
        """
        own = state[player]
        opp = state[1 - player]
        occ = own | opp
        out = []
        if own == 0:
            targets = 0
            for sq, bit in _START_BITS.items():
                if opp & bit == 0:
                    targets |= bit
            for piece in state[2 + player]:
                for a, placement, _ortho, _diag in PIECE_ACTION_TABLES[piece]:
                    if placement & occ == 0 and placement & targets:
                        out.append(a)
        else:
            for piece in state[2 + player]:
                for a, placement, ortho, diag in PIECE_ACTION_TABLES[piece]:
                    if placement & occ == 0 and ortho & own == 0 and diag & own != 0:
                        out.append(a)
        out.sort()
        return out

    def place(self, state, action: int):
        """Apply ``action`` for the mover ``state[6]``; no normalization (adapter's job).

        Args:
            state: Engine state tuple.
            action: A legal flat action id for the mover.

        Returns:
            The successor state tuple with ``to_play``/``terminal`` carried
            unchanged; the monomino-last flag is set iff the monomino empties
            the inventory on this placement.
        """
        player = state[6]
        piece = ORIENTATION_PIECE[decode(action)[2]]
        inv = state[2 + player] - {piece}
        parts = list(state)
        parts[player] = state[player] | cells_to_bb(action_cells(action))
        parts[2 + player] = inv
        parts[4 + player] = piece == MONOMINO and not inv
        return tuple(parts)

    def scores(self, state) -> tuple[int, int]:
        """Return the official scores ``(score_p0, score_p1)`` (§4).

        Args:
            state: Engine state tuple.

        Returns:
            Per-player integer scores: −1 per unplaced square, +15 all placed,
            +5 more with the monomino-last flag.
        """
        out = []
        for p in (0, 1):
            inv = state[2 + p]
            if inv:
                out.append(-sum(len(BASE_PIECES[i]) for i in inv))
            else:
                out.append(20 if state[4 + p] else 15)
        return tuple(out)
