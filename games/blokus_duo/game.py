"""BlokusDuo adapter: the ``Game`` contract over an interchangeable rules engine.

The adapter owns the §6.1 pass-invariant normalization — engines only place
pieces and report legality/scores. Forced passes are realized by *skipping* the
blocked player (no pass action in the ``H×W×K`` head): ``apply`` hands the move
to the first of [opponent, mover] with a legal action, else marks the state
terminal. Blocking is monotone in Blokus, but that is an adapter-level fact —
core never assumes it.

The engine (oracle or bitboard) is injected so the whole contract battery and
the differential fuzz can run against either implementation [F8]. Since M2.5 the
adapter also takes a :class:`~games.blokus_duo.config.BlokusConfig`, defaulting
to the full 14×14 game: ``policy_shape``, ``input_planes``, ``input_shape``, the
``encode_state`` plane layout, the declared ``symmetry_group`` and the D1 aux
divisor are all derived from it (§5.2's plane count is a formula, not a
constant; §8's group is the computed start-square stabilizer), so the §5.3 micro
instance is a construction argument rather than a fork.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from core.game import Action, Game, PlayerId, State, SymmetryElement, ValueTargetSpec
from games.blokus_duo.actions import action_codec
from games.blokus_duo.bitboard import BitboardEngine
from games.blokus_duo.config import FULL_CONFIG, BlokusConfig
from games.blokus_duo.pieces import build_pieces
from games.blokus_duo.symmetry import symmetry_group
from games.blokus_duo.targets import value_target_spec, value_targets

_TO_PLAY, _TERMINAL = 6, 7


class BlokusDuo(Game):
    """Blokus Duo (2-player) behind the generic ``Game`` interface.

    Args:
        engine: Rules engine providing ``initial_state`` / ``legal_actions`` /
            ``place`` / ``scores`` over the shared state tuple. Defaults to the
            production bitboard engine (~9x faster search) for ``config``;
            reference tests inject the cell-grid oracle explicitly.
        config: The instance to play; defaults to the injected engine's config,
            or the full 14×14 game when no engine is given.

    Raises:
        ValueError: If both an engine and a config are given and they disagree —
            a mismatched pair would silently mix two action spaces.
    """

    def __init__(self, engine=None, config: BlokusConfig | None = None):
        engine_config = getattr(engine, "config", None)
        if config is None:
            config = engine_config if engine_config is not None else FULL_CONFIG
        elif engine_config is not None and engine_config != config:
            raise ValueError(
                f"engine config {engine_config} does not match adapter config {config}"
            )
        self._config = config
        self._engine = engine if engine is not None else BitboardEngine(config)
        self._codec = action_codec(config)
        self._symmetry = symmetry_group(config)
        self._board_size = config.board_size
        pieces = build_pieces(config)[0]
        self._num_pieces = len(pieces)
        # §5.2: the two monomino-last completion planes exist only if this
        # instance's piece set contains the monomino (pieces sort by size).
        self._has_monomino = len(pieces[0]) == 1
        # Constant broadcast planes for the inventory/flag channels (D3) —
        # immutable, so the shared tuples are safely reused across every
        # encoded state.
        self._zeros = tuple((0,) * self._board_size for _ in range(self._board_size))
        self._ones = tuple((1,) * self._board_size for _ in range(self._board_size))

    @property
    def config(self) -> BlokusConfig:
        """The Blokus instance this adapter plays."""
        return self._config

    # --- declared capabilities ---------------------------------------------------

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
    def symmetry_group(self) -> Sequence[SymmetryElement]:
        # The instance's group (§8, D9), *computed* as the D4 set-stabilizer of
        # this config's start squares — Klein-4 for both pinned instances, but
        # never hardcoded. Each element pairs the plane transform over the
        # encode_state output (46 planes full, 12 micro) with the full
        # H*W*K-length head permutation, identity filler on off-support ids
        # [F6, revised per PR #2 review].
        group = self._symmetry
        return tuple((group.plane_transform(g), group.full_permutation(g)) for g in group.names)

    @property
    def value_targets(self) -> ValueTargetSpec:
        return value_target_spec(self._config)

    # --- core contract -------------------------------------------------------------

    def initial_state(self) -> State:
        return self._engine.initial_state()

    def current_player(self, state: State) -> PlayerId:
        return state[_TO_PLAY]

    def legal_moves(self, state: State) -> Sequence[Action]:
        return self._engine.legal_actions(state, state[_TO_PLAY])

    def apply(self, state: State, action: Action) -> State:
        """Place for the mover, then normalize ``to_play``/``terminal`` (§6.1).

        Args:
            state: A nonterminal state whose mover has ``action`` legal.
            action: Flat action id to play. Behavior is undefined for illegal
                actions (core only ever applies ids from ``legal_moves``).

        Returns:
            The successor state; its mover is guaranteed >= 1 legal action
            unless the state is terminal (pass invariant).
        """
        mover = state[_TO_PLAY]
        nxt = self._engine.place(state, action)
        parts = list(nxt)
        for player in (1 - mover, mover):
            if self._engine.legal_actions(nxt, player):
                parts[_TO_PLAY] = player
                parts[_TERMINAL] = False
                return tuple(parts)
        parts[_TERMINAL] = True
        return tuple(parts)

    def is_terminal(self, state: State) -> bool:
        return state[_TERMINAL]

    def _score_targets(self, state: State, player_id: PlayerId) -> tuple[int, float]:
        """Return the D1 ``(z, aux)`` pair for a terminal state, mover-relative.

        The single place the adapter turns engine scores into targets — both
        :meth:`terminal_utility` and :meth:`training_targets` route through it,
        so the search value and the training value can never disagree.

        Args:
            state: A terminal state.
            player_id: The player whose perspective the targets are stated in.

        Returns:
            ``(z, aux)`` from :func:`~games.blokus_duo.targets.value_targets`,
            with the aux divisor derived from this instance's config (109 full,
            29 micro).
        """
        scores = self._engine.scores(state)
        return value_targets(scores[player_id], scores[1 - player_id], self._config)

    def terminal_utility(self, state: State, player_id: PlayerId) -> float:
        return float(self._score_targets(state, player_id)[0])

    def training_targets(
        self, state: State, player_id: PlayerId
    ) -> tuple[float, tuple[float, ...]]:
        """Return ``(z, (score_diff_aux,))`` — the D1 targets the loop stores.

        Overrides the ABC default (which supplies no aux) because Blokus
        declares one auxiliary head: core cannot derive ``score_diff /
        max|score_diff|`` from the declared spec's names and weights.

        Args:
            state: A terminal state.
            player_id: The player whose perspective the targets are stated in.

        Returns:
            ``(z, aux)`` with ``z = sign(score_diff)`` and a one-element aux
            tuple parallel to ``value_targets.aux_names``.
        """
        z, aux = self._score_targets(state, player_id)
        return float(z), (aux,)

    # --- encoding surface (action side owned by M1; plane side by M2) ---------------

    def _occupancy_plane(self, occ):
        """Project one occupancy onto an ``H×W`` plane of ``{0, 1}``.

        Handles both engine representations — occupancy ints (bitboard, bit
        ``r*W + c``) and frozensets of ``(r, c)`` cells (oracle) — the same dual
        dispatch as ``symmetry.state_transform``.

        Args:
            occ: One player's occupancy from the shared engine state tuple.

        Returns:
            Nested ``H×W`` tuples over ``{0, 1}``.
        """
        size = self._board_size
        if isinstance(occ, int):
            return tuple(tuple(occ >> (r * size + c) & 1 for c in range(size)) for r in range(size))
        return tuple(tuple(1 if (r, c) in occ else 0 for c in range(size)) for r in range(size))

    def encode_state(self, state: State):
        """Encode ``state`` as the D3 planes (§5.2), mover-relative.

        Plane order (pinned by D3/§5.2): own occupancy, opponent occupancy, one
        own-inventory plane per piece, one opponent-inventory plane per piece
        (piece order fixed by the config's piece set, §5.1), then — iff the
        instance has a monomino — own and opponent monomino-last flags.
        Inventory and flag planes are constant broadcast planes (all 1s iff the
        piece is in inventory / the flag is set). "Own" is the side to move —
        no side-to-move plane (§5.2). The full game instantiates the formula at
        46 planes, the §5.3 micro instance at 12.

        Args:
            state: Engine state tuple; occupancies as ints (bitboard) or
                frozensets of cells (oracle) — both handled.

        Returns:
            ``input_planes`` nested ``H×W`` tuples over ``{0, 1}`` — stdlib-pure;
            the training boundary converts with ``numpy.asarray``.
        """
        mover = state[_TO_PLAY]
        planes = [self._occupancy_plane(state[p]) for p in (mover, 1 - mover)]
        for p in (mover, 1 - mover):
            inv = state[2 + p]
            planes.extend(
                self._ones if piece in inv else self._zeros for piece in range(self._num_pieces)
            )
        if self._has_monomino:
            planes.extend(self._ones if state[4 + p] else self._zeros for p in (mover, 1 - mover))
        return tuple(planes)

    def encode_action(self, move: Any) -> Action:
        """Encode absolute placement cells as a flat action id."""
        return self._codec.encode_cells(move)

    def decode_action(self, action: Action) -> Any:
        """Decode a flat action id into its absolute placement cells."""
        return self._codec.action_cells(action)

    @property
    def policy_shape(self) -> tuple[int, ...]:
        return (self._board_size, self._board_size, self._codec.num_orientations)

    @property
    def input_planes(self) -> int:
        # §5.2 formula: 2 occupancy + 2 x #pieces inventory + 2 monomino-last
        # flags iff the monomino is in the set. Full game: 46 (D3); micro: 12.
        return 2 + 2 * self._num_pieces + (2 if self._has_monomino else 0)

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self._board_size, self._board_size)
