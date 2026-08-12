"""The ``Game`` interface, capability declarations, and the v1-envelope assertion.

Design doc §6.1. Adapters *declare* their capabilities; ``core/`` asserts the v1
engine envelope (2-player, zero-sum, perfect-information, deterministic) and fails
loudly if a game exceeds it — the scope boundary is encoded in code, not just prose.

State is opaque to core: it is any adapter-defined value. The reference games use
immutable, hashable states (tuples) so the reference solver can memoize on them, but
the search tree itself does not require hashability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

State = Any
Action = int
PlayerId = int


class EnvelopeError(Exception):
    """Raised when an adapter declares capabilities outside the v1 engine envelope.

    The v1 envelope is 2-player, non-stochastic, perfect-information. Seams for
    N-player (M7) and stochastic transitions are documented in the design doc but
    not built; imperfect information is permanently out of scope for this engine.
    """


@dataclass(frozen=True)
class ValueTargetSpec:
    """Declares how a terminal outcome maps to training targets (design doc §6.1, D1).

    M0 games do not train; this is the declared contract surface, exercised for real
    at M1 (Blokus ``value_targets`` golden) and M2 (network heads). The primary target
    is the scalar ``z``; adapters may declare auxiliary heads (e.g. Blokus's normalized
    score-difference), each carrying its own loss weight.

    Attributes:
        primary_name: Label of the primary (scalar ``z``) value target.
        aux_names: Labels of optional auxiliary heads, in head order.
        aux_loss_weights: Loss weight per auxiliary head, parallel to ``aux_names``.
    """

    primary_name: str = "z"
    aux_names: tuple[str, ...] = ()
    aux_loss_weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.aux_names) != len(self.aux_loss_weights):
            raise ValueError("aux_names and aux_loss_weights must be parallel")


# A single symmetry element: (state-plane transform, action-index permutation).
# Declared per adapter; core contains no game-specific symmetry logic. M0 games
# declare the empty group (identity only); the real groups land with Blokus (M1)
# and Othello (M1.5).
SymmetryElement = tuple[Callable[[Any], Any], Sequence[int]]


class Game(ABC):
    """Stateless rules object behind which a single game hides (design doc §6.1).

    The object carries no per-position mutable state: every method takes an opaque
    ``state`` value and returns a new one. This keeps states cheap to store as MCTS
    tree nodes and trivially reusable across searches.

    The **pass invariant** (never violated by core): at every nonterminal state,
    ``current_player`` returns a player with >= 1 legal action. Adapters realize
    forced passes either as an explicit pass action in the action space (Othello:
    64+1 head) or by skipping inactive players (Blokus: no pass action). Core assumes
    *only* this invariant — never strict alternation, never monotone blocking.
    """

    # --- declared capabilities (design doc §6.1) -------------------------------

    @property
    @abstractmethod
    def num_players(self) -> int:
        """Number of players. Asserted ``== 2`` by the v1 envelope."""

    @property
    @abstractmethod
    def is_stochastic(self) -> bool:
        """Whether transitions are stochastic. Asserted ``False`` by the v1 envelope."""

    @property
    @abstractmethod
    def is_perfect_information(self) -> bool:
        """Whether the game is perfect-information. Asserted ``True`` by the v1 envelope."""

    @property
    @abstractmethod
    def symmetry_group(self) -> Sequence[SymmetryElement]:
        """Declared symmetry group as ``(plane_transform, action_permutation)`` pairs.

        Empty means "identity only" (no augmentation). Declared, never hardcoded in core.
        """

    @property
    @abstractmethod
    def value_targets(self) -> ValueTargetSpec:
        """Declared value/aux target specification (design doc §6.1, D1)."""

    # --- core contract ---------------------------------------------------------

    @abstractmethod
    def initial_state(self) -> State:
        """Return the game's start state."""

    @abstractmethod
    def current_player(self, state: State) -> PlayerId:
        """Return the player to move.

        Guaranteed meaningful only at nonterminal states, where — per the pass
        invariant — the returned player has >= 1 legal action. Behavior at terminal
        states is unspecified; core never calls this on a terminal state.
        """

    @abstractmethod
    def legal_moves(self, state: State) -> Sequence[Action]:
        """Return the sparse list of legal action ids. Nonempty at nonterminal states."""

    @abstractmethod
    def apply(self, state: State, action: Action) -> State:
        """Return the successor state. Deterministic in v1.

        Stochastic seam (not built): may generalize to a distribution over next states
        or explicit chance nodes, with ``apply`` as the point-mass special case.
        """

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Return whether ``state`` is terminal (game over)."""

    @abstractmethod
    def terminal_utility(self, state: State, player_id: PlayerId) -> float:
        """Return the exact utility of a terminal ``state`` for ``player_id``.

        v1: in ``{-1, 0, +1}`` and zero-sum across the two players. Player-parameterized
        (an exact value, not an expected outcome). N-player seam: generalizes to a
        per-player utility vector.
        """

    def training_targets(
        self, state: State, player_id: PlayerId
    ) -> tuple[float, tuple[float, ...]]:
        """Return the terminal training targets ``(z, aux)`` for ``player_id``.

        The generic terminal-target surface a game-agnostic self-play loop
        consumes (M2.5 task 4; supersedes the "no core additions" note in
        ``games/blokus_duo/targets.py``). It is a *declared* adapter surface in
        the §6.1 pattern — the same shape as ``value_targets``, ``symmetry_group``
        and the encoding methods — not one game's head leaking into core: core
        cannot materialize an aux value itself, because ``ValueTargetSpec``
        carries head names and loss weights but no mapping from a terminal state
        to a number (Blokus's ``score_diff / max|diff|`` lives in its adapter).

        Both components are **mover-relative**, i.e. stated from ``player_id``'s
        perspective, exactly like :meth:`terminal_utility`. The default is the
        no-aux case — the primary ``z`` is the terminal utility and the aux tuple
        is empty — so adapters declaring no auxiliary heads (TTT, Connect 4,
        Othello) need no override, while an adapter that declares aux heads and
        forgets to override fails loudly here instead of silently training
        against missing targets.

        Args:
            state: A terminal state.
            player_id: The player whose perspective the targets are stated in.

        Returns:
            ``(z, aux)``: the primary scalar target and one value per declared
            entry of ``value_targets.aux_names``, in head order.

        Raises:
            NotImplementedError: If the game declares auxiliary heads but does
                not override this method — the default cannot produce them.
        """
        aux: tuple[float, ...] = ()
        declared = self.value_targets.aux_names
        if len(aux) != len(declared):
            raise NotImplementedError(
                f"{type(self).__name__} declares aux heads {declared} but does not "
                f"override training_targets; the default supplies the primary z only"
            )
        return self.terminal_utility(state, player_id), aux

    # --- encoding surface (abstract since M2, per the M0 seam note) --------------
    # M0 declared these as concrete NotImplementedError stubs (no network → no
    # tensors); with the network landed, every game entering the training stack
    # must encode, so a missing surface now fails at instantiation, not first use.

    @abstractmethod
    def encode_state(self, state: State):
        """Encode ``state`` as a plane tensor of shape ``(input_planes, *input_shape)``."""

    @abstractmethod
    def encode_action(self, move: Any) -> Action:
        """Map a move to its action id."""

    @abstractmethod
    def decode_action(self, action: Action) -> Any:
        """Map an action id back to a move."""

    @property
    @abstractmethod
    def policy_shape(self) -> tuple[int, ...]:
        """Shape of the policy head."""

    @property
    @abstractmethod
    def input_planes(self) -> int:
        """Number of input planes."""

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, int]:
        """``(height, width)`` of every input plane.

        The full input tensor is ``(input_planes, *input_shape)``. Declared, not
        derived: a flat policy head (Othello's ``(65,)``) carries no board
        geometry to recover ``(H, W)`` from.
        """


def assert_v1_envelope(game: Game) -> None:
    """Assert ``game`` lies within the v1 engine envelope, or raise ``EnvelopeError``.

    The v1 envelope is 2-player, non-stochastic, perfect-information. This is the
    "asserted in code, not just prose" scope boundary (design doc §2, §6.1): a
    mis-declared adapter must fail loudly here rather than silently producing
    corrupt search or training signal.

    Args:
        game: The adapter to validate.

    Raises:
        EnvelopeError: If the game declares more than two players, stochastic
            transitions, or imperfect information.
    """
    if game.num_players != 2:
        raise EnvelopeError(
            f"v1 engine is 2-player; {type(game).__name__} declares "
            f"num_players={game.num_players} (N-player is the M7 seam, not built)"
        )
    if game.is_stochastic:
        raise EnvelopeError(
            f"v1 engine is deterministic; {type(game).__name__} declares "
            f"is_stochastic=True (stochastic transitions are a documented seam, not built)"
        )
    if not game.is_perfect_information:
        raise EnvelopeError(
            f"v1 engine is perfect-information; {type(game).__name__} declares "
            f"is_perfect_information=False (imperfect information is permanently "
            f"out of scope for this engine)"
        )
