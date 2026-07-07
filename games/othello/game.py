"""Othello (8×8): the second real game behind the generic ``Game`` interface (M1.5).

The abstraction test of design doc §12 M1.5: Othello realizes the §6.1 pass
invariant with an **explicit pass action** (id 64 in a flat 64+1 head), the
opposite realization from Blokus's auto-skip — and its passing is non-monotone
(a passed player can regain a move), so it exercises exactly what Blokus can't.

Conventions (§12 M1.5 pin block): 0-indexed row-major ``(r, c)``; placement
``action_id = r*8 + c`` (0–63), pass = 64; player 0 = Black moves first from the
standard start (White (3,3),(4,4); Black (3,4),(4,3)). State is the immutable,
hashable tuple ``(board, to_play)`` with ``board`` a length-64 tuple over
``{EMPTY, 0, 1}`` so the reference solver can memoize on it.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.game import Action, PlayerId, State, ValueTargetSpec
from core.game import Game as _Game

BOARD_SIZE = 8
EMPTY = -1
PASS = 64

_DIRS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


class Othello(_Game):
    """Standard 8×8 Othello with an explicit pass action (flat 64+1 head)."""

    @property
    def num_players(self) -> int:
        return 2

    @property
    def is_stochastic(self) -> bool:
        return False

    @property
    def is_perfect_information(self) -> bool:
        return True

    @property
    def symmetry_group(self) -> Sequence:
        return ()

    @property
    def value_targets(self) -> ValueTargetSpec:
        return ValueTargetSpec(primary_name="z")

    def initial_state(self) -> State:
        board = [EMPTY] * 64
        board[3 * 8 + 3] = 1
        board[4 * 8 + 4] = 1
        board[3 * 8 + 4] = 0
        board[4 * 8 + 3] = 0
        return (tuple(board), 0)

    def current_player(self, state: State) -> PlayerId:
        return state[1]

    def legal_moves(self, state: State) -> Sequence[Action]:
        """Return legal placements, or ``[PASS]`` when blocked but nonterminal.

        Pass is legal iff the mover has no placement and the opponent has one
        (§12 M1.5 pin) — so the pass invariant holds at every nonterminal state
        and a terminal state offers no action at all.
        """
        board, to_play = state
        moves = _placements(board, to_play)
        if moves:
            return moves
        return [PASS] if _placements(board, 1 - to_play) else []

    def apply(self, state: State, action: Action) -> State:
        """Place (flipping flanked lines) and hand the move to the opponent.

        Strict alternation at the action level (§12 M1.5 pin): ``apply`` always
        flips ``to_play``, for placements and passes alike — the *placement*
        sequence is what goes consecutive when a blocked player passes.

        Args:
            state: A nonterminal state whose mover has ``action`` legal.
            action: Placement id (0–63) or ``PASS`` (64). Behavior is undefined
                for illegal actions (core only ever applies ids from
                ``legal_moves``).

        Returns:
            The successor ``(board, to_play)`` state.
        """
        board, to_play = state
        if action == PASS:
            return (board, 1 - to_play)
        cells = _flips(board, to_play, action)
        cells.append(action)
        new_board = list(board)
        for i in cells:
            new_board[i] = to_play
        return (tuple(new_board), 1 - to_play)

    def is_terminal(self, state: State) -> bool:
        return not _placements(state[0], 0) and not _placements(state[0], 1)

    def terminal_utility(self, state: State, player_id: PlayerId) -> float:
        """Return ``sign(disc_diff)`` for ``player_id`` (§12 M1.5 pin).

        Raw disc counts, empties unassigned — the sign (including draws) is
        invariant to the empties-to-winner scoring convention.
        """
        board = state[0]
        diff = board.count(player_id) - board.count(1 - player_id)
        return float((diff > 0) - (diff < 0))

    # --- encoding surface (M1.5-carried, §12 M1.5 scope note) --------------------

    def encode_state(self, state: State):
        """Encode ``state`` as 2 mover-relative occupancy planes (own, opponent).

        Nested 8×8 tuples over {0, 1} — stdlib-only until NumPy arrives at M2.
        Mover-relative per §5.2's own/opponent convention: no side-to-move plane.
        """
        board, to_play = state
        return tuple(
            tuple(
                tuple(1 if board[r * BOARD_SIZE + c] == player else 0 for c in range(BOARD_SIZE))
                for r in range(BOARD_SIZE)
            )
            for player in (to_play, 1 - to_play)
        )

    def encode_action(self, move) -> Action:
        """Encode ``(r, c)`` as ``r*8 + c``, or ``"pass"`` as 64."""
        if move == "pass":
            return PASS
        r, c = move
        return r * BOARD_SIZE + c

    def decode_action(self, action: Action):
        """Decode an action id into ``(r, c)``, or 64 into ``"pass"``."""
        if action == PASS:
            return "pass"
        return divmod(action, BOARD_SIZE)

    @property
    def policy_shape(self) -> tuple[int, ...]:
        return (65,)

    @property
    def input_planes(self) -> int:
        return 2

    # --- test convenience ------------------------------------------------------

    def from_grid(self, rows: Sequence[str], to_play: PlayerId) -> State:
        """Build a state from an 8-row ASCII grid (``B`` -> 0, ``W`` -> 1, ``.`` -> empty).

        Args:
            rows: Eight strings of eight chars each.
            to_play: The player to move in the constructed position.

        Returns:
            The corresponding ``(board, to_play)`` state.
        """
        mapping = {"B": 0, "W": 1, ".": EMPTY}
        cells = [mapping[ch] for row in rows for ch in row]
        if len(cells) != 64:
            raise ValueError("grid must be 8x8")
        return (tuple(cells), to_play)


def _flips(board: tuple, player: PlayerId, cell: int) -> list[int]:
    """Return the board indices flipped by ``player`` placing on empty ``cell``."""
    r0, c0 = divmod(cell, BOARD_SIZE)
    flips: list[int] = []
    for dr, dc in _DIRS:
        r, c = r0 + dr, c0 + dc
        line: list[int] = []
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            v = board[r * BOARD_SIZE + c]
            if v == 1 - player:
                line.append(r * BOARD_SIZE + c)
            elif v == player:
                flips.extend(line)
                break
            else:
                break
            r, c = r + dr, c + dc
    return flips


def _placements(board: tuple, player: PlayerId) -> list[Action]:
    """Return the sorted placement action ids legal for ``player`` (pass excluded)."""
    return [i for i in range(64) if board[i] == EMPTY and _flips(board, player, i)]
