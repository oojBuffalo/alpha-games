"""Game-generic self-play: one game to terminal, the replay window, the run record.

The M2.5 minimal loop's self-play half (§12 M2.5, task 6.3/6.4). Everything here
is game-agnostic — ``core/selfplay.py`` imports nothing from ``games/``; the
micro instance arrives only as the ``game`` argument — and stdlib-pure: torch
stays confined to ``core/network.py`` / ``core/losses.py`` / ``core/train.py``
(the pyproject pin), so the learner half of the loop lives in
``scripts/run_micro.py`` and consumes the plain records produced here.

The three pinned decisions this module realizes:

* **D10 move selection.** :func:`select_move` samples ∝ *raw* N (τ = 1, no
  exponentiation) at plies below ``k_temp`` and plays argmax N at and after it,
  with ``MCTS.best_action``'s tie-break (most visits, ties to the lowest action
  id). :func:`policy_target` mirrors the root's raw counts **verbatim** at every
  stored ply — including when subtree reuse has inflated ΣN past the sim budget,
  which is the point of storing counts rather than a normalized vector.
* **D7 root noise.** Wired through ``MCTS(root_noise=...)`` — the hook already in
  ``core/mcts.py`` — from the per-game Dirichlet stream, self-play only, and
  re-drawn by the engine whenever a node becomes root (subtree reuse included).
* **D12 sample shape.** Every stored sample carries sparse ``(action_id,
  visit_count)`` pairs (Invariant 3 — never a dense policy vector), plus ``z``
  and the declared aux targets backfilled at game end. Fast/low-sim positions are
  dropped entirely at M5; M2.5 searches every ply at one fixed sim count, so
  every ply is stored.

``z`` and aux are **mover-relative**: each sample is backfilled from
``game.training_targets(terminal_state, sample.mover)`` — the declared ABC
surface (task 4), never a game import or a private helper — matching the
mover-relative own/opponent encoding the network is trained on.

Seeding is per-purpose (``core.seeding``): ``play_game`` takes a
:class:`~core.seeding.GameRNGs` bundle, so a change in how many Dirichlet draws a
search makes cannot shift the move-selection sequence.

M3 (issue #59) extends :class:`Sample`/:func:`play_game` backward-compatibly with
the two fields the full on-disk sample record (``core.replay_shard.PendingSample``)
needs beyond what this module already produced: ``model_version`` (the pinned
weight version the whole game's search ran against, §6.2) and ``game_id`` (the
durable ``(run_id, actor_id, game_index)`` triple). Both are optional keyword-only
parameters on ``play_game``, stamped verbatim onto every sample of the game;
omitted, they default to ``None`` and every existing call site (the micro loop,
the exit gate) is unaffected. Assigning a *real* ``game_id`` is never this
module's job — that identity comes from exactly one place,
``core.replay_shard.ShardWriter``'s persisted state — ``play_game`` only carries
whatever label its caller (``core.actor.ActorDriver``) already derived from it.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from core.game import Action, Game, PlayerId, State
from core.mcts import MCTS, Evaluator
from core.runconfig import SelfPlayConfig
from core.seeding import GameRNGs

# Schema tag stamped into every persisted run record. Task 7's exit gate reads
# the persisted file, so the shape it reads is versioned from the start; M3's
# hardened run record supersedes it rather than silently reshaping it.
RUN_RECORD_SCHEMA = "alpha-games/run-record/v1"


def select_move(
    root_visits: Mapping[Action, int], ply: int, k_temp: int, rng: random.Random
) -> Action:
    """Choose the move to play from the root's visit counts (D10).

    Below ``k_temp`` the move is sampled ∝ *raw* N — τ = 1 with **no**
    exponentiation, exactly ``MCTS.select_action(temperature=1.0)``. At and after
    ``k_temp`` the most-visited move is played, ties broken by lowest action id,
    exactly ``MCTS.best_action``. (π_train is unaffected: counts are stored
    verbatim at *every* ply — see :func:`policy_target`.)

    Args:
        root_visits: ``{action_id: visit_count}`` over the root's legal actions,
            i.e. ``MCTS.action_visit_counts()``.
        ply: Zero-based ply index within the game.
        k_temp: The D10 temperature cutoff ply (``SelfPlayConfig.k_temp``).
        rng: The move-selection stream (``GameRNGs.move_selection``); consumed
            only on the sampling branch.

    Returns:
        The chosen action id.

    Raises:
        ValueError: If ``root_visits`` is empty — the pass invariant guarantees
            at least one legal action at every nonterminal state, so an empty
            root is a caller bug, not a position to shrug at.
    """
    actions = list(root_visits)
    if not actions:
        raise ValueError("no root visit counts to select from")
    if ply >= k_temp:
        return _argmax_visits(root_visits)
    weights = [root_visits[a] for a in actions]
    if sum(weights) <= 0:
        # Degenerate: no simulation reached an edge (sims == 1 expands the root
        # and stops). Uniform, mirroring MCTS.select_action, never a zero-weight
        # sample that would raise from inside random.choices.
        return rng.choice(actions)
    return rng.choices(actions, weights=weights, k=1)[0]


def _argmax_visits(root_visits: Mapping[Action, int]) -> Action:
    """Return the most-visited action, ties broken by lowest action id.

    The tie-break is ``MCTS.best_action``'s, restated over a counts mapping so
    the two can never drift: most visits wins; equal visits go to the lower
    action id, never to adapter or dict-insertion order.

    Args:
        root_visits: ``{action_id: visit_count}`` over the root's legal actions.

    Returns:
        The chosen action id.
    """
    return min(root_visits, key=lambda a: (-root_visits[a], a))


def policy_target(root_visits: Mapping[Action, int]) -> list[tuple[Action, int]]:
    """Return the sparse D12 policy target: the root's raw counts, verbatim.

    Sparse ``(action_id, visit_count)`` pairs over the root's legal set only
    (Invariant 3), in the root's edge order, with the counts **unmodified** —
    not normalized, not rescaled to the sim budget. Subtree reuse legitimately
    leaves ΣN above ``cfg.sims`` (the promoted child arrives with its share of
    the previous search's visits), and the extra visits are real search effort:
    the loss normalizes ``π_train(a) = N(a)/ΣN`` at collate time, so any
    rescaling here would only lose information.

    Args:
        root_visits: ``{action_id: visit_count}`` over the root's legal actions,
            i.e. ``MCTS.action_visit_counts()``.

    Returns:
        The full legal set as ``(action_id, visit_count)`` pairs; zero-count
        legal actions are kept, since they shape the legal-set renormalization
        in ``core.losses.sparse_policy_loss``.
    """
    return list(root_visits.items())


@dataclass(frozen=True)
class Sample:
    """One stored self-play position — the base sample shape (D12).

    ``(planes, sparse_pi, ply)`` is the base triple M3's versioned
    ``SampleRecord`` extends backward-compatibly; ``mover`` is carried because
    the terminal backfill is mover-relative, and ``z``/``aux`` are ``None`` until
    that backfill runs (:func:`backfill_targets`). ``model_version``/``game_id``
    are the two fields the M3 actor layer adds on top (see the module docstring);
    both are ``None`` unless ``play_game`` was called with them set.

    Attributes:
        planes: The adapter's ``encode_state`` output (nested tuples; the
            stdlib→tensor conversion happens at ``core.train.collate``).
        sparse_pi: The D12 sparse policy target, ``(action_id, visit_count)``
            pairs over the position's full legal set.
        ply: Zero-based ply index within the game.
        mover: The player to move at this position — the perspective ``planes``,
            ``z`` and ``aux`` are all stated in.
        z: The D1 primary target, or ``None`` before backfill.
        aux: The declared auxiliary targets in head order, or ``None`` before
            backfill (a game declaring no aux heads backfills an empty tuple).
        model_version: The pinned weight version the search ran against (§6.2),
            or ``None`` when the caller did not supply one (every non-actor
            caller, e.g. the micro loop).
        game_id: The durable ``(run_id, actor_id, game_index)`` triple this
            sample's game was played under, or ``None`` when the caller did not
            supply one. Never assigned by this module — only ever a label
            carried through from the caller (``core.actor.ActorDriver``, which
            reads it from ``core.replay_shard.ShardWriter``'s persisted state).
    """

    planes: Any
    sparse_pi: tuple[tuple[Action, int], ...]
    ply: int
    mover: PlayerId
    z: float | None = None
    aux: tuple[float, ...] | None = None
    model_version: int | None = None
    game_id: tuple[str, str, int] | None = None

    def training_row(self, num_aux: int) -> tuple[Any, ...]:
        """Return the sample in ``core.train.collate``'s spec-driven row shape.

        Args:
            num_aux: Number of declared aux heads
                (``len(game.value_targets.aux_names)``).

        Returns:
            ``(planes, sparse_pi, z, aux)`` when ``num_aux`` is positive,
            ``(planes, sparse_pi, z)`` otherwise — collate rejects the other
            arity loudly, and a no-aux game carries no zero-filled slot.

        Raises:
            ValueError: If the sample has not been backfilled, or its aux width
                disagrees with ``num_aux``.
        """
        if self.z is None or self.aux is None:
            raise ValueError(f"sample at ply {self.ply} was never backfilled with (z, aux)")
        if len(self.aux) != num_aux:
            raise ValueError(
                f"sample at ply {self.ply} carries {len(self.aux)} aux target(s), "
                f"but the game declares {num_aux}"
            )
        if num_aux == 0:
            return (self.planes, self.sparse_pi, self.z)
        return (self.planes, self.sparse_pi, self.z, self.aux)


@dataclass(frozen=True)
class GameResult:
    """One finished self-play game.

    Attributes:
        samples: The backfilled per-ply samples, in play order.
        moves: The actions played, in order.
        terminal_state: The game's terminal state — kept so a caller can re-derive
            every target through the public ``Game`` surface (the z-consistency
            test does exactly that).
        utilities: Terminal utility per player id, zero-sum in v1.
    """

    samples: tuple[Sample, ...]
    moves: tuple[Action, ...]
    terminal_state: State
    utilities: tuple[float, ...]

    @property
    def plies(self) -> int:
        """Number of actions played from the initial state to terminal."""
        return len(self.moves)


def backfill_targets(game: Game, terminal: State, samples: Iterable[Sample]) -> tuple[Sample, ...]:
    """Fill each sample's ``(z, aux)`` from the terminal state, mover-relative.

    Every sample is stated from *its own* mover's perspective — the same
    perspective its encoded planes use — so the targets are read per sample
    rather than once per game and sign-flipped by parity: Blokus lets one player
    move consecutively once the other is blocked, so a parity rule would be
    wrong, not merely fragile.

    Args:
        game: The adapter, queried only through its public
            ``training_targets`` surface (task 4).
        terminal: The game's terminal state.
        samples: The un-backfilled samples.

    Returns:
        The samples with ``z`` and ``aux`` set, in input order.
    """
    cache: dict[PlayerId, tuple[float, tuple[float, ...]]] = {}
    filled = []
    for sample in samples:
        targets = cache.get(sample.mover)
        if targets is None:
            z, aux = game.training_targets(terminal, sample.mover)
            targets = (float(z), tuple(float(a) for a in aux))
            cache[sample.mover] = targets
        filled.append(replace(sample, z=targets[0], aux=targets[1]))
    return tuple(filled)


def play_game(
    game: Game,
    evaluator: Evaluator | None,
    cfg: SelfPlayConfig,
    rngs: GameRNGs,
    *,
    model_version: int | None = None,
    game_id: tuple[str, str, int] | None = None,
) -> GameResult:
    """Play one complete self-play game, returning its stored samples.

    One :class:`~core.mcts.MCTS` instance drives the whole game, so the subtree
    under the played move is reused (§6.2) and the D7 hook re-draws root noise on
    the promoted root. Every ply is searched at the fixed ``cfg.sims`` (M2.5 runs
    no playout-cap randomization — that is D6/M5), every ply is stored, and the
    targets are backfilled once the game ends.

    Args:
        game: The adapter to play. Game-specific behavior enters *only* here —
            this module never imports ``games/``.
        evaluator: Leaf evaluator for MCTS (``core.network.make_network_evaluator``
            bridges the batch-1 network in). ``None`` runs the M0 engine: value
            0.0 and uniform priors, which is what the stdlib-only tests use.
        cfg: The shared self-play scalars (``core.runconfig.SelfPlayConfig``);
            ``root_noise`` False disables the D7 hook entirely.
        rngs: The per-game stream bundle (``GameRNGs.for_game(run_seed, index)``);
            ``dirichlet`` goes to the D7 hook and ``move_selection`` to D10.
            ``tie_break`` is deliberately unconsumed here: at M2.5 both the PUCT
            tie-break and the D10 argmax tie-break are deterministic, and the
            stream stays reserved so introducing a randomized tie-break later
            cannot shift the move-selection sequence.
        model_version: Stamped verbatim onto every sample's ``Sample.model_version``
            (§6.2 version pinning — one model version per game). ``None`` (the
            default) leaves every sample's ``model_version`` unset, which is what
            every caller through M2.5 does; the M3 actor layer is the first
            caller to pass one.
        game_id: Stamped verbatim onto every sample's ``Sample.game_id``. This
            function never derives or validates the id — it is only ever a
            label the caller already owns (``core.actor.ActorDriver``, from
            ``core.replay_shard.ShardWriter``'s persisted state).

    Returns:
        The finished :class:`GameResult`, its samples already backfilled.
    """
    root_noise = None
    if cfg.root_noise:
        root_noise = (cfg.dirichlet_eps, cfg.dirichlet_alpha_numerator, rngs.dirichlet)
    search = MCTS(game, evaluate=evaluator, root_noise=root_noise)
    state = game.initial_state()
    search.set_root(state)

    samples: list[Sample] = []
    moves: list[Action] = []
    ply = 0
    while not game.is_terminal(state):
        search.run(cfg.sims)
        visits = search.action_visit_counts()
        samples.append(
            Sample(
                planes=game.encode_state(state),
                sparse_pi=tuple(policy_target(visits)),
                ply=ply,
                mover=game.current_player(state),
                model_version=model_version,
                game_id=game_id,
            )
        )
        action = select_move(visits, ply, cfg.k_temp, rngs.move_selection)
        search.advance(action)
        state = game.apply(state, action)
        moves.append(action)
        ply += 1

    return GameResult(
        samples=backfill_targets(game, state, samples),
        moves=tuple(moves),
        terminal_state=state,
        utilities=tuple(game.terminal_utility(state, p) for p in range(game.num_players)),
    )


class ReplayWindow:
    """A fixed-capacity ring buffer of samples with uniform batch sampling.

    The minimal M2.5 stand-in for D5's 250k on-disk replay (M3/M5 scope): oldest
    samples are evicted once the window is full, and batches are drawn uniformly
    **with replacement** — the window is routinely smaller than a few batches at
    micro budgets, and D5's "2–4 samples per stored position" is a rate over the
    run, not a per-batch constraint.

    Args:
        capacity: Maximum samples retained (``TrainingConfig.replay_window``).

    Raises:
        ValueError: If ``capacity`` is not positive.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"replay window capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._items: list[Sample] = []
        self._total_added = 0

    @property
    def capacity(self) -> int:
        """The window's maximum size, in samples."""
        return self._capacity

    @property
    def total_added(self) -> int:
        """Samples ever added, including those since evicted."""
        return self._total_added

    def __len__(self) -> int:
        """Return the number of samples currently retained."""
        return len(self._items)

    def extend(self, samples: Iterable[Sample]) -> None:
        """Append samples, evicting the oldest beyond ``capacity``.

        Args:
            samples: The samples to add, in order.
        """
        for sample in samples:
            self._items.append(sample)
            self._total_added += 1
        if len(self._items) > self._capacity:
            self._items = self._items[-self._capacity :]

    def sample_batch(self, batch_size: int, rng: random.Random) -> list[Sample]:
        """Draw a uniform batch, with replacement.

        Args:
            batch_size: Number of samples to draw.
            rng: The window-sampling stream
                (``LearnerRNGs.window_sampling``).

        Returns:
            ``batch_size`` samples drawn uniformly from the window.

        Raises:
            ValueError: If ``batch_size`` is not positive, or the window is
                empty — a learner step on an empty window is a pacing bug and
                must fail loudly rather than train on nothing.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not self._items:
            raise ValueError("cannot sample from an empty replay window")
        return rng.choices(self._items, k=batch_size)


@dataclass
class RunRecord:
    """The persisted evidence one run produces (§12 M2.5).

    Task 7's exit gate reads its loss predicates from the *written file* — never
    from a number recomputed ad hoc — so this shape is the contract between the
    loop and the gate. The JSON layout is::

        {
          "schema": "alpha-games/run-record/v1",
          "run_name": str,
          "run_seed": int,
          "config": {...},              # RunConfig.to_dict()
          "game_identity": {"game": str, "game_config": str,
                            "orientation_hash": str},
          "device": str,
          "steps": [{"step": int, "policy_loss": float, "value_loss": float,
                     "aux_loss": float|null, "total_loss": float,
                     "learning_rate": float, "window_size": int,
                     "games_played": int}, ...],
          "games": [{"game_index": int, "plies": int, "samples": int,
                     "utilities": [float, float], "moves": [int, ...]}, ...],
          "checkpoints": [{"step": int, "kind": str, "path": str}, ...],
          "timing": {"self_play_seconds": float, "train_seconds": float,
                     "total_seconds": float}
        }

    ``steps`` is in learner-step order and ``step`` is the zero-based step index,
    so the predicates' head window is ``steps[:head_window_steps]`` and the tail
    window ``steps[-tail_window_steps:]``. ``aux_loss`` is ``null`` for a game
    declaring no aux heads — absent, never zero-filled, mirroring the batch
    convention.

    Attributes:
        run_name: The run config's ``name``.
        run_seed: The single recorded run seed every stream derives from.
        config: The run config as nested plain dicts (``RunConfig.to_dict()``).
        game_identity: Game, instance-config name, and the instance's
            orientation-table hash (Invariant 4 — the same digest stamped into
            the checkpoint, which M3 validates on load).
        device: The torch device the learner ran on.
        steps: One entry per learner step (see the layout above).
        games: One entry per completed self-play game.
        checkpoints: One entry per written checkpoint.
        timing: Coarse wall-clock totals — observational only, and the one part
            of the record that is *not* reproducible across runs.
    """

    run_name: str
    run_seed: int
    config: dict[str, Any]
    game_identity: dict[str, Any]
    device: str = "cpu"
    steps: list[dict[str, Any]] = field(default_factory=list)
    games: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)

    def record_step(
        self,
        step: int,
        *,
        policy_loss: float,
        value_loss: float,
        aux_loss: float | None,
        total_loss: float,
        learning_rate: float,
        window_size: int,
        games_played: int,
    ) -> None:
        """Append one learner step's component losses.

        Args:
            step: Zero-based learner-step index.
            policy_loss: The step's policy cross-entropy (predicate 2 reads it).
            value_loss: The step's value MSE (predicate 3 reads it).
            aux_loss: The step's aux loss, or ``None`` for a no-aux game.
            total_loss: The §7 composite total actually optimized.
            learning_rate: The LR in force at this step (warmup+cosine).
            window_size: Replay-window occupancy when the batch was drawn.
            games_played: Self-play games completed before this step.
        """
        self.steps.append(
            {
                "step": step,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "aux_loss": aux_loss,
                "total_loss": total_loss,
                "learning_rate": learning_rate,
                "window_size": window_size,
                "games_played": games_played,
            }
        )

    def record_game(self, game_index: int, result: GameResult) -> None:
        """Append one completed self-play game.

        Args:
            game_index: The game's durable index within the run — the same index
                its ``GameRNGs`` were derived from.
            result: The finished game.
        """
        self.games.append(
            {
                "game_index": game_index,
                "plies": result.plies,
                "samples": len(result.samples),
                "utilities": list(result.utilities),
                "moves": list(result.moves),
            }
        )

    def record_checkpoint(self, step: int, kind: str, path: str) -> None:
        """Append one written checkpoint.

        Args:
            step: Learner steps completed when it was written.
            kind: ``"periodic"`` or ``"final"`` — the gate reads the ``"final"``
                one (``training.checkpoint_selection``).
            path: Path to the checkpoint file, as written.
        """
        self.checkpoints.append({"step": step, "kind": kind, "path": path})

    def loss_series(self, name: str) -> list[float | None]:
        """Return one loss component across all recorded steps, in step order.

        Args:
            name: ``"policy_loss"``, ``"value_loss"``, ``"aux_loss"`` or
                ``"total_loss"``.

        Returns:
            The per-step values (``None`` entries only for ``aux_loss`` on a
            no-aux game).

        Raises:
            KeyError: If ``name`` is not a recorded loss component.
        """
        if name not in ("policy_loss", "value_loss", "aux_loss", "total_loss"):
            raise KeyError(f"unknown loss component {name!r}")
        return [step[name] for step in self.steps]

    def to_dict(self) -> dict[str, Any]:
        """Return the record in its persisted JSON layout.

        Returns:
            A plain nested ``dict``, schema tag first.
        """
        return {
            "schema": RUN_RECORD_SCHEMA,
            "run_name": self.run_name,
            "run_seed": self.run_seed,
            "config": self.config,
            "game_identity": dict(self.game_identity),
            "device": self.device,
            "steps": list(self.steps),
            "games": list(self.games),
            "checkpoints": list(self.checkpoints),
            "timing": dict(self.timing),
        }

    def write(self, path: Path | str) -> Path:
        """Write the record as JSON, creating parent directories.

        Args:
            path: Destination file.

        Returns:
            The path written.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return out


def load_run_record(path: Path | str) -> dict[str, Any]:
    """Read a persisted run record, checking its schema tag.

    The gate-side reader: task 7 evaluates its predicates on *this* dict, never
    on a number recomputed from a live loop.

    Args:
        path: The run-record file.

    Returns:
        The parsed record.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the document is not an object or carries an unknown
            schema tag.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(f"run record: expected an object, got {type(raw).__name__}")
    schema = raw.get("schema")
    if schema != RUN_RECORD_SCHEMA:
        raise ValueError(f"unknown run-record schema {schema!r} (expected {RUN_RECORD_SCHEMA!r})")
    return dict(raw)
