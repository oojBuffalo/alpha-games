"""The Blokus instance config: board dims, start squares, piece subset (§5.1, §5.3).

The M2.5 micro instance is **not** a new game package — it is this same
``games/blokus_duo/`` package constructed from a different config, so "adding a
game touches only ``games/`` + ``configs/``" holds trivially and no ``core/``
file changes for it. :data:`FULL_CONFIG` is the pinned 14×14 game and the
default everywhere; :data:`MICRO_CONFIG` is the design doc §5.3 instance.

The config is frozen and hashable so every derived table (piece set, orientation
table, action codec, bitboard masks) can be built once per config behind
``functools.cache`` keyed on the config value itself. Validation is loud and
eager: a config that names an unknown piece or puts a start square off the board
raises at construction, never at first move generation.

This module deliberately imports only :mod:`games.blokus_duo.pieces` (for the
known piece names) — every other module in the package imports *it*, so the
dependency stays acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass

from games.blokus_duo.pieces import PIECE_NAMES


@dataclass(frozen=True)
class BlokusConfig:
    """One Blokus instance, as a hashable construction argument.

    Attributes:
        board_size: Edge length of the square board (14 for the full game).
        start_squares: The two 0-indexed opening squares (§5.1; the doc's
            1-indexed (5,5)/(10,10) is the display convention). P1's opening
            covers either, P2's the other — the rule needs exactly two.
        piece_names: Names of the pieces in this instance's set (keys of the
            canonical 21), or ``None`` for all of them. Order is irrelevant:
            pieces are re-sorted by ``(size, canonical form)`` and renumbered
            from 0 *within the subset* (invariant 4), so a subset's piece and
            orientation ids are re-derived, never the full table restricted.
    """

    board_size: int = 14
    start_squares: tuple[tuple[int, int], ...] = ((4, 4), (9, 9))
    piece_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the instance and normalize the sequence fields to tuples.

        Raises:
            ValueError: If ``board_size`` is not positive; if the start squares
                are not exactly two distinct on-board cells; or if
                ``piece_names`` is empty, has duplicates, or names a piece
                outside the canonical 21.
        """
        if self.board_size < 1:
            raise ValueError(f"board_size must be >= 1, got {self.board_size}")

        squares = tuple((int(r), int(c)) for r, c in self.start_squares)
        if len(squares) != 2:
            raise ValueError(f"expected exactly 2 start squares (§4 opening), got {len(squares)}")
        if len(set(squares)) != 2:
            raise ValueError(f"start squares must be distinct, got {squares}")
        for square in squares:
            if not all(0 <= x < self.board_size for x in square):
                raise ValueError(
                    f"start square {square} is off a {self.board_size}x{self.board_size} board"
                )
        object.__setattr__(self, "start_squares", squares)

        if self.piece_names is None:
            return
        names = tuple(self.piece_names)
        if not names:
            raise ValueError("piece_names must name at least one piece (use None for all 21)")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate piece names in {names}")
        unknown = tuple(n for n in names if n not in PIECE_NAMES)
        if unknown:
            raise ValueError(f"unknown piece names {unknown}; known names are {PIECE_NAMES}")
        object.__setattr__(self, "piece_names", names)


# The pinned full game (§4, §5.1) — the default instance of every engine, the
# adapter, and the action codec.
FULL_CONFIG = BlokusConfig()

# The pinned M2.5 micro instance (§5.3): 5×5, all free polyominoes of order <= 3,
# start squares (1,1)/(3,3). Its derived constants — 9 orientations, (5,5,9)
# head, 159 in-bounds placements, 42 openings, 12 planes, aux divisor 29, and its
# own orientation hash — are independently enumerated by
# ``scripts/enumerate_micro_config.py``.
MICRO_CONFIG = BlokusConfig(
    board_size=5,
    start_squares=((1, 1), (3, 3)),
    piece_names=("I1", "I2", "I3", "V3"),
)
