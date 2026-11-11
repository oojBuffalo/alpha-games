"""The M3 self-play actor driver (design doc §6.2/§7, §12 M3, issue #59).

``core.selfplay.play_game`` is the reusable, game-generic *mechanics*: one
game, root Dirichlet noise, D10 move selection, mover-relative terminal
backfill. Everything actor/process-specific — which config the actor is
pinned to, where a game's durable identity comes from, when weights are
allowed to change, and how production is paced — lives here, one layer up,
and is deliberately kept out of ``core.selfplay`` so the shared loop stays a
plain function any caller (the micro loop, evaluation, this module) can drive
without inheriting actor concerns it doesn't have.

**The D6 budgets are an actor-layer assertion, not a game-loop one.**
``core.selfplay.play_game`` stays fully parameterizable over its config —
the micro loop runs it at a fixed 64, evaluation at whatever
``EvaluationConfig.sims`` names. :class:`ActorDriver` is the caller that is
*pinned* to D6 (§7), in exactly two shapes: the validate tier (fixed 128
sims, playout-cap randomization off) and the playout-cap schedule (plies 0-1
at 512; later plies 25% at 256, 75% at 64). So that is exactly where the
check belongs: :func:`validate_actor_self_play_config` runs once, at
construction, and raises loudly on anything else — a hybrid of the two
included — rather than letting a misconfigured actor silently search at the
wrong budget for an entire run.

**Durable identity has exactly one source: the writer's persisted state.**
``core.replay_shard.ShardWriter`` already owns a crash-safe
``(next_shard_seq, next_game_index)`` pair, durably advanced before a shard
publishes. The actor never keeps a second, in-process game counter — every
game's index is read from ``writer.state.next_game_index`` immediately before
play (fixing the seed and the label a game will carry) and the *same* value
is independently re-derived by ``ShardWriter.write_shard`` when it publishes,
so the two can never disagree while exactly one writer is live for a given
``(run_id, actor_id)`` (``ShardWriter``'s own contract). A crash between
those two reads-of-the-same-fact burns the index — the shard is simply never
published — but can never reissue it to a real game on restart, because a
fresh :class:`ActorDriver` reloads the same persisted state.

**Per-game seeding is a pure function of durable coordinates.**
``core.seeding.GameRNGs.for_actor_game(run_seed, actor_id, game_index)``
takes exactly the triple above (minus ``run_id``, which the seed stream
doesn't need), so a game replayed after a restart draws bit-identical
Dirichlet/move-selection streams to the one a continuous run would have
played at that index.

**Weight refresh and pacing are both between-game hooks, never mid-search.**
``refresh`` is called once per iteration of :meth:`ActorDriver.run`'s loop,
strictly before the game it pins is played, and its returned
``model_version`` is stamped onto every sample of that game
(``core.selfplay.play_game``'s new optional parameter) and never changes
until the next call -- within-search leaf batching (``self_play.leaf_batch_size``, §12
M5) never crosses a game boundary, so a batch is always evaluated by the one
version its game is pinned to. ``pacing``, when supplied, is polled the same way,
before ``refresh`` — a "hold" response defers to an injectable ``wait``
strategy (real backoff in production; instrumented and sleep-free in tests)
until it clears. Neither hook's cadence depends on this module knowing
anything about replay ratios, checkpoint files, or IPC — those are the
concerns of the seams issue #60 (pacing math) and issue #61 (real weight
publication) build against.

**Observability is opt-in and additive (issue #62).** ``metrics_writer`` /
``position_counter`` are both ``None`` by default -- a caller that never
supplies them (every test predating this issue) gets byte-identical
behavior. When both are given, every finished game flushes three ``delta``
records to ``metrics_writer`` at the game boundary: ``games_completed``
(always ``1``), ``sims_run`` (``GameResult.sims_run`` -- the exact sum of
the budgets the game's plies were searched at, which under playout-cap
randomization is not ``plies * self_play.sims``), and
``positions_evaluated`` (drained from ``position_counter``, which the
caller is responsible for wiring into the actual evaluator via
``core.observability.count_positions`` -- this module never constructs an
evaluator itself). ``positions_evaluated`` is omitted entirely when no
``position_counter`` was supplied, never written as a fabricated zero. An
optional ``batch_counter``, wired into the evaluator the same way, adds the
inference-batch accounting deltas to the same flush (``inference_batches``
plus the non-zero size buckets -- ``core.observability``), under the same
omitted-when-absent rule. A game played under playout-cap randomization also
flushes its per-tier move counts (``moves_full_512`` / ``moves_mid_256`` /
``moves_fast_64``, each only when non-zero); a validate-tier game writes none.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from core.game import Game
from core.mcts import BatchEvaluator, Evaluator
from core.metrics import EpochMetricsWriter
from core.observability import (
    SERIES_GAMES_COMPLETED,
    SERIES_MOVES_FAST_64,
    SERIES_MOVES_FULL_512,
    SERIES_MOVES_MID_256,
    SERIES_POSITIONS_EVALUATED,
    SERIES_SIMS_RUN,
    InferenceBatchCounter,
    PositionCounter,
    delta_record,
)
from core.replay_shard import PendingSample, ShardWriter
from core.runconfig import SelfPlayConfig
from core.seeding import GameRNGs
from core.selfplay import TIER_FAST, TIER_FULL, TIER_MID, GameResult, Sample, play_game

# D6: validate the loop at a fixed simulation count first (§7); the actor is
# pinned to it, the shared game loop is not.
VALIDATE_TIER_SIMS = 128

# D6: then the playout-cap schedule (§7) -- plies 0-1 always at the full
# budget; every later ply at the mid budget with this probability, else fast.
PCR_FULL_SIMS = 512
PCR_MID_SIMS = 256
PCR_FAST_SIMS = 64
PCR_MID_PROBABILITY = 0.25
PCR_FULL_PLIES = 2

# The delta series each playout-cap tier's plies are counted under. The fixed
# tier (playout-cap randomization off) has none: ``sims_run / sims`` is its
# ply count.
_TIER_SERIES = {
    TIER_FULL: SERIES_MOVES_FULL_512,
    TIER_MID: SERIES_MOVES_MID_256,
    TIER_FAST: SERIES_MOVES_FAST_64,
}

# One weight refresh: the evaluator to search with (None = the M0 uniform/zero
# path), the model version pinned for every sample of the next game, and --
# optionally -- the same weights as a batched evaluator, which the driver
# searches with when ``self_play.leaf_batch_size > 1``. A refresh that has no
# batched evaluator returns the two-element form.
RefreshResult = tuple[Evaluator | None, int] | tuple[Evaluator | None, int, BatchEvaluator | None]
RefreshFn = Callable[[], RefreshResult]

# True = hold production (the learner is behind the replay-ratio floor,
# issue #60's concern, not this module's); False = proceed.
PacingFn = Callable[[], bool]
WaitFn = Callable[[], None]
StopFn = Callable[[], bool]


def validate_actor_self_play_config(cfg: SelfPlayConfig) -> None:
    """Assert ``cfg`` is one of the two D6 shapes an :class:`ActorDriver` runs.

    The validate tier is ``sims == 128`` with playout-cap randomization off
    (the ``pcr_*`` scalars are unused there and not checked). The playout-cap
    schedule is the flag on with ``sims`` and all four ``pcr_*`` scalars at
    the D6 pins. Nothing in between is accepted; ``leaf_batch_size`` is free
    in both.

    Args:
        cfg: The self-play scalars an actor is about to be built with.

    Raises:
        ValueError: If ``cfg`` is neither shape.
    """
    schedule = (
        cfg.sims,
        cfg.pcr_mid_sims,
        cfg.pcr_fast_sims,
        cfg.pcr_mid_probability,
        cfg.pcr_full_plies,
    )
    d6_schedule = (PCR_FULL_SIMS, PCR_MID_SIMS, PCR_FAST_SIMS, PCR_MID_PROBABILITY, PCR_FULL_PLIES)
    if cfg.playout_cap_randomization:
        if schedule == d6_schedule:
            return
    elif cfg.sims == VALIDATE_TIER_SIMS:
        return
    raise ValueError(
        "actor self-play config must be one of the two D6 shapes: the validate tier "
        f"(sims {VALIDATE_TIER_SIMS}, playout-cap randomization off) or the playout-cap "
        f"schedule (playout-cap randomization on, sims {PCR_FULL_SIMS}, pcr_mid_sims "
        f"{PCR_MID_SIMS}, pcr_fast_sims {PCR_FAST_SIMS}, pcr_mid_probability "
        f"{PCR_MID_PROBABILITY}, pcr_full_plies {PCR_FULL_PLIES}); got "
        f"playout_cap_randomization={cfg.playout_cap_randomization}, sims={cfg.sims}, "
        f"pcr_mid_sims={cfg.pcr_mid_sims}, pcr_fast_sims={cfg.pcr_fast_sims}, "
        f"pcr_mid_probability={cfg.pcr_mid_probability}, pcr_full_plies={cfg.pcr_full_plies}"
    )


def _default_wait() -> None:
    """The production pacing backoff.

    A real (short) sleep — every test supplies its own ``wait`` instead, so
    this is never exercised off the actor's default construction path.
    """
    time.sleep(1.0)


def _to_pending_sample(sample: Sample) -> PendingSample:
    """Convert one backfilled, actor-stamped :class:`Sample` to a :class:`PendingSample`.

    Args:
        sample: A sample from a game played by :func:`core.selfplay.play_game`
            with ``model_version`` set — every actor-driven game supplies one.

    Returns:
        The equivalent :class:`~core.replay_shard.PendingSample`, still
        missing its durable game id (:class:`~core.replay_shard.ShardWriter`
        assigns that on publish — see the module docstring).

    Raises:
        ValueError: If ``sample`` was never backfilled with ``(z, aux)``, or
            was played without a pinned ``model_version``.
    """
    if sample.z is None or sample.aux is None:
        raise ValueError(f"sample at ply {sample.ply} was never backfilled with (z, aux)")
    if sample.model_version is None:
        raise ValueError(f"sample at ply {sample.ply} carries no pinned model_version")
    return PendingSample(
        planes=sample.planes,
        sparse_pi=sample.sparse_pi,
        z=sample.z,
        aux=sample.aux,
        mover=sample.mover,
        model_version=sample.model_version,
        ply=sample.ply,
    )


class ActorDriver:
    """Drives whole self-play games for one actor, publishing every position.

    Owns the pieces that are specific to being *an actor* rather than to
    playing one game: the D6 config assertion, the durable
    ``(run_id, actor_id, game_index)`` identity sourced only from
    :class:`~core.replay_shard.ShardWriter`'s persisted state, per-game
    seeding, and the between-games refresh/pacing discipline. See the module
    docstring for the crash-safety and determinism arguments.

    Args:
        game: The adapter to play. Never imported by this module — the
            caller constructs it, exactly as ``core.selfplay.play_game``
            expects.
        self_play: The self-play scalars every game is played with; must be
            one of the two D6 shapes (:func:`validate_actor_self_play_config`).
        run_id: This run's identity (constant across every actor in the run).
        actor_id: This actor's durable identifier within the run — an
            ``int`` because ``core.seeding.GameRNGs.for_actor_game`` requires
            one; stringified for ``ShardWriter``'s filename/``game_id`` use.
        out_dir: Directory shards and this actor's writer-state file are
            published into (``ShardWriter``'s ``shard_dir``).
        run_seed: The run's recorded root seed every game's
            :class:`~core.seeding.GameRNGs` derives from.
        refresh: Called once per game, strictly before that game is played,
            never while a search is in flight. Returns the evaluator to
            search with and the model version pinned for every sample of the
            next game, plus optionally a batched evaluator over the same
            weights (:data:`RefreshResult`). The batched one is used only
            when ``self_play.leaf_batch_size > 1``: at ``1`` the search is the
            serial one through the scalar evaluator, exactly as before.
        pacing: Optional. Polled the same way as ``refresh``, before it, on
            every iteration of :meth:`run`'s loop; while it returns ``True``
            the driver calls ``wait`` and polls again rather than starting a
            game. ``None`` (the default) never pauses.
        wait: The backoff strategy invoked while ``pacing`` holds. Defaults
            to a short real sleep — every test supplies its own no-sleep
            fake.
        max_games: Optional cap on games played by one :meth:`run` call —
            the test-facing stop condition; a production actor leaves this
            unset and relies on ``should_stop`` (issue #63 wires the real
            lifecycle).
        should_stop: Optional callable polled alongside ``max_games``; ``run``
            stops as soon as either says to.
        metrics_writer: Optional. When given, every finished game flushes
            ``games_completed`` / ``sims_run`` / ``positions_evaluated``
            deltas to it, plus the per-tier move counts under playout-cap
            randomization (module docstring). ``None`` (the default) disables
            all metrics flushing — backward compatible with every existing
            caller.
        position_counter: Optional. Drained once per game to populate the
            ``positions_evaluated`` delta; only consulted when
            ``metrics_writer`` is also given. The caller wires this counter
            into the actual evaluator (``core.observability.count_positions``)
            — this module only drains it.
        batch_counter: Optional. Drained once per game to populate the
            inference-batch accounting deltas; only consulted when
            ``metrics_writer`` is also given, and wired by the caller exactly
            like ``position_counter``.

    Raises:
        ValueError: If ``self_play`` is neither D6 shape, or ``max_games``
            is not positive.
    """

    def __init__(
        self,
        *,
        game: Game,
        self_play: SelfPlayConfig,
        run_id: str,
        actor_id: int,
        out_dir: Path | str,
        run_seed: int,
        refresh: RefreshFn,
        pacing: PacingFn | None = None,
        wait: WaitFn | None = None,
        max_games: int | None = None,
        should_stop: StopFn | None = None,
        metrics_writer: EpochMetricsWriter | None = None,
        position_counter: PositionCounter | None = None,
        batch_counter: InferenceBatchCounter | None = None,
    ) -> None:
        validate_actor_self_play_config(self_play)
        if max_games is not None and max_games <= 0:
            raise ValueError(f"max_games must be positive, got {max_games}")

        self.game = game
        self.self_play = self_play
        self.run_id = run_id
        self.actor_id = actor_id
        self.run_seed = run_seed
        self.refresh = refresh
        self.pacing = pacing
        self.wait = wait if wait is not None else _default_wait
        self.max_games = max_games
        self.should_stop = should_stop
        self.metrics_writer = metrics_writer
        self.position_counter = position_counter
        self.batch_counter = batch_counter
        self.writer = ShardWriter(Path(out_dir), game, run_id, str(actor_id))

    def run(self) -> list[Path]:
        """Play games until the stop condition triggers.

        Each iteration: check the stop condition, wait out a pacing hold,
        re-check the stop condition (a hold can span an arbitrarily long
        wait -- issue #61's signal-shutdown wiring needs the loop to notice
        ``should_stop`` firing *during* the hold, not only at the top of the
        next iteration, or a paused actor would never exit), refresh
        weights, play one whole game at the pinned version, and publish it
        as its own shard. A production actor (no ``max_games`` /
        ``should_stop``) loops forever, exactly like the design doc's
        actor–learner split assumes.

        Returns:
            The shard paths published during this call, in play order.
        """
        published: list[Path] = []
        while not self._stop_requested(len(published)):
            self._await_pacing()
            # Only re-check when a pacing hook is actually installed: with
            # none, ``_await_pacing`` is a guaranteed no-op and re-checking
            # here would poll ``should_stop`` an extra, observable time per
            # iteration for no behavioral gain.
            if self.pacing is not None and self._stop_requested(len(published)):
                break
            evaluator, model_version, *batched = self.refresh()
            evaluate_batch = batched[0] if batched and self.self_play.leaf_batch_size > 1 else None
            published.append(
                self._play_and_publish_one_game(evaluator, model_version, evaluate_batch)
            )
        return published

    def _stop_requested(self, games_this_call: int) -> bool:
        """Return whether ``run`` should stop before starting another game.

        Args:
            games_this_call: Games already published by the current ``run``
                call.

        Returns:
            ``True`` if ``max_games`` has been reached or ``should_stop``
            says so.
        """
        if self.max_games is not None and games_this_call >= self.max_games:
            return True
        return self.should_stop is not None and self.should_stop()

    def _await_pacing(self) -> None:
        """Block on the pacing hook, via ``wait``, until it clears (or is absent).

        Also returns early if ``should_stop`` fires while held -- without
        this, an actor paused by a learner permanently below the D5 replay
        floor could never observe a shutdown signal (issue #61): the pacing
        hold has no other exit condition, so a signal-driven ``should_stop``
        must be polled from inside it, not only between games.
        """
        if self.pacing is None:
            return
        while self.pacing():
            if self.should_stop is not None and self.should_stop():
                return
            self.wait()

    def _play_and_publish_one_game(
        self,
        evaluator: Evaluator | None,
        model_version: int,
        evaluate_batch: BatchEvaluator | None = None,
    ) -> Path:
        """Play one game at the pinned version and publish it as its own shard.

        The durable game index is read from the writer's persisted state
        immediately before play (fixing the seed and the ``game_id`` label the
        game carries) and re-derived independently by
        ``ShardWriter.write_shard`` at publish time — the module docstring's
        "exactly one source" argument for why the two can never disagree.

        Args:
            evaluator: This game's leaf evaluator, from ``refresh``.
            model_version: This game's pinned model version, from ``refresh``.
            evaluate_batch: This game's batched leaf evaluator, or ``None`` to
                evaluate leaves one at a time through ``evaluator``.

        Returns:
            The published shard's path.
        """
        game_index = self.writer.state.next_game_index
        game_id = (self.run_id, str(self.actor_id), game_index)
        rngs = GameRNGs.for_actor_game(self.run_seed, self.actor_id, game_index)
        result = play_game(
            self.game,
            evaluator,
            self.self_play,
            rngs,
            model_version=model_version,
            game_id=game_id,
            evaluate_batch=evaluate_batch,
        )
        pending = [_to_pending_sample(sample) for sample in result.samples]
        path = self.writer.write_shard([pending])
        # ShardWriter is the only source of durable game indices; if this ever
        # disagrees, something else has started writing under our actor_id.
        assert self.writer.state.next_game_index == game_index + 1
        self._flush_game_metrics(result)
        return path

    def _flush_game_metrics(self, result: GameResult) -> None:
        """Append this game's delta series at the between-game flush boundary.

        A no-op unless ``metrics_writer`` was supplied at construction
        (module docstring: optional, off by default). ``games_completed`` is
        always exactly ``1`` -- one flush per finished game. ``sims_run`` is
        ``result.sims_run``, the exact sum of the budgets the game's plies
        were searched at. Each playout-cap tier the game played adds its ply
        count under that tier's series; a validate-tier game played none, so
        it writes none. ``positions_evaluated`` is drained from
        ``position_counter`` and omitted entirely when none was supplied --
        never a fabricated zero. The inference-batch deltas are drained from
        ``batch_counter`` under the same rule.

        Args:
            result: The just-finished game.
        """
        if self.metrics_writer is None:
            return
        now = time.time()
        self.metrics_writer.append(delta_record(SERIES_GAMES_COMPLETED, 1, timestamp=now))
        self.metrics_writer.append(delta_record(SERIES_SIMS_RUN, result.sims_run, timestamp=now))
        tier_counts = dict(result.tier_counts)
        for tier, series in _TIER_SERIES.items():
            if tier in tier_counts:
                self.metrics_writer.append(delta_record(series, tier_counts[tier], timestamp=now))
        if self.position_counter is not None:
            positions = self.position_counter.drain()
            self.metrics_writer.append(
                delta_record(SERIES_POSITIONS_EVALUATED, positions, timestamp=now)
            )
        if self.batch_counter is not None:
            for series, value in self.batch_counter.drain().items():
                self.metrics_writer.append(delta_record(series, value, timestamp=now))
