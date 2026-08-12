"""Bitboard rules engine: ``H*W``-bit ints, precomputed per-action masks (design doc §6.3).

Bit index = ``r*W + c`` (the §5.1 cell flatten) — 196 bits for the full 14×14
game, 25 for the §5.3 micro instance; nothing here assumes either. Per in-bounds
action, three precomputed masks: ``placement_bb`` (the piece cells),
``ortho_halo_bb`` (orthogonal neighbors, in-bounds, excluding placement cells),
``diag_halo_bb`` (diagonal neighbors, likewise) — the in-bounds clipping in the
halos is what replaces an explicit sentinel/halo column scheme, and it clips at
the configured dims. Post-opening legality is four integer ops:
``piece available ∧ placement & occ == 0 ∧ ortho & own == 0 ∧ diag & own != 0``;
the opening substitutes start-square coverage for the diagonal-contact clause
(orthogonal contact is vacuous on an empty own board, but the mask check is
kept — it is free and identical).

Independence contract: shares only piece data and the action-encoding surface
with the oracle; masks are derived from the codec's ``action_cells`` (the
golden-tested decode side), not from the oracle's transforms. State layout
matches the shared tuple convention with occupancies as ints instead of
frozensets.
"""

from __future__ import annotations

from functools import cache

from games.blokus_duo.actions import action_codec
from games.blokus_duo.config import FULL_CONFIG, BlokusConfig
from games.blokus_duo.pieces import build_pieces

MONOMINO = 0

_ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def cells_to_bb(cells, board_size: int = FULL_CONFIG.board_size) -> int:
    """Pack ``(row, col)`` cells into an occupancy int (bit = ``r*board_size + c``).

    Args:
        cells: Iterable of on-board ``(row, col)`` tuples.
        board_size: Row stride of the layout; defaults to the full game's 14
            (a 196-bit board).

    Returns:
        The occupancy bitboard.
    """
    bb = 0
    for r, c in cells:
        bb |= 1 << (r * board_size + c)
    return bb


def _halo(cells, offsets, board_size: int) -> int:
    placement = set(cells)
    halo = 0
    for r, c in cells:
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < board_size and 0 <= nc < board_size and (nr, nc) not in placement:
                halo |= 1 << (nr * board_size + nc)
    return halo


@cache
def build_piece_action_tables(
    config: BlokusConfig = FULL_CONFIG,
) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
    """Build ``config``'s per-piece mask table, once per config.

    Args:
        config: The instance config; defaults to the full 14×14 game.

    Returns:
        Per piece (in the config's piece order), the tuple of
        ``(action_id, placement_bb, ortho_halo_bb, diag_halo_bb)`` rows for
        every in-bounds placement of that piece — piece availability prunes
        whole blocks at once.
    """
    codec = action_codec(config)
    board_size = config.board_size
    per_piece: list[list[tuple[int, int, int, int]]] = [[] for _ in build_pieces(config)[0]]
    for a in codec.in_bounds_actions:
        cells = codec.action_cells(a)
        per_piece[codec.action_piece(a)].append(
            (
                a,
                cells_to_bb(cells, board_size),
                _halo(cells, _ORTH, board_size),
                _halo(cells, _DIAG, board_size),
            )
        )
    return tuple(tuple(rows) for rows in per_piece)


# The full game's table — the M1 module surface (used by the perft generator).
PIECE_ACTION_TABLES = build_piece_action_tables(FULL_CONFIG)


class BitboardEngine:
    """Mask-based rules engine over the shared state tuple (occupancies as ints).

    Board dims, start squares, the action codec and the per-piece mask tables are
    instance state derived from the config, so one class serves the full game and
    any reduced instance (§5.3).

    Args:
        config: The instance config; defaults to the full 14×14 game.

    Attributes:
        config: The config this engine plays.
    """

    def __init__(self, config: BlokusConfig = FULL_CONFIG):
        self.config = config
        self._codec = action_codec(config)
        self._board_size = config.board_size
        self._pieces = build_pieces(config)[0]
        self._tables = build_piece_action_tables(config)
        # Piece index of the monomino, or None if this instance has no size-1
        # piece (pieces sort by size, so it can only be index 0).
        self._monomino = MONOMINO if len(self._pieces[0]) == 1 else None
        self._start_bits = {
            sq: 1 << (sq[0] * self._board_size + sq[1]) for sq in config.start_squares
        }

    def initial_state(self):
        """Return the start state (empty boards, full inventories, P1 to move)."""
        full = frozenset(range(len(self._pieces)))
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
            for bit in self._start_bits.values():
                if opp & bit == 0:
                    targets |= bit
            for piece in state[2 + player]:
                for a, placement, _ortho, _diag in self._tables[piece]:
                    if placement & occ == 0 and placement & targets:
                        out.append(a)
        else:
            for piece in state[2 + player]:
                for a, placement, ortho, diag in self._tables[piece]:
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
        piece = self._codec.action_piece(action)
        inv = state[2 + player] - {piece}
        parts = list(state)
        parts[player] = state[player] | cells_to_bb(
            self._codec.action_cells(action), self._board_size
        )
        parts[2 + player] = inv
        parts[4 + player] = piece == self._monomino and not inv
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
                out.append(-sum(len(self._pieces[i]) for i in inv))
            else:
                out.append(20 if state[4 + p] else 15)
        return tuple(out)
