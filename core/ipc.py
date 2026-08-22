"""Actor–learner filesystem IPC on one GPU (design doc §12 M3, issue #61).

Wires ``core.actor.ActorDriver`` and ``core.learner.LearnerDriver`` -- both
already built, both already sparse/idempotent/atomic on their own artifacts --
into separate OS processes that coordinate **only** through the filesystem
artifacts those two modules already define:

* **Weight publication (learner → actors).** :func:`build_actor_refresh`
  reads ``core.checkpoint``'s atomically-updated ``latest`` pointer and loads
  the published checkpoint it names (``core.checkpoint.load_checkpoint``,
  full fingerprint validation included), caching the built evaluator by
  version so an unchanged ``latest`` never re-hits the disk or rebuilds a
  net. This *is* ``ActorDriver``'s ``refresh`` seam -- issue #59 built the
  hook, this issue is the real implementation behind it.
* **Sample flow (actors → learner).** Unchanged from issue #59/#60:
  ``ShardWriter`` publishes temp-then-renamed shards; ``ReplayWindow``
  rescans and picks them up. This module adds no new step to that path --
  the same ``shard_dir`` is simply the directory both processes' drivers are
  pointed at.
* **Pacing (learner → actors).** :func:`build_actor_pacing` reads the
  learner's ``pacing.json`` (``core.learner.read_pacing_file``) and is the
  real implementation behind ``ActorDriver``'s ``pacing`` seam; a missing
  file (fresh start, or still in warm-up) reads as "go", never "hold" --
  exactly ``read_pacing_file``'s documented contract.

**No locks, no queues, no pipes carry data.** Atomic replace
(temp-name-then-``os.replace``, already how every artifact above is written)
is the only cross-process synchronization primitive on either path;
collision-freedom, idempotent rescans, and durable ordering come from the
shard-naming/state-file protocol and the replay manifest (``core.replay_shard``/
``core.replay_window``), never from anything this module adds. The only
thing this module's processes exchange directly with each other is a Unix
signal (below) -- a stop notification, never a payload.

**Process model.** :func:`run_actor_process` / :func:`run_learner_process`
are the picklable, module-level ``multiprocessing.Process`` targets;
:func:`launch_run` starts ``N`` actors + one learner under
``multiprocessing.get_context("spawn")`` **explicitly** -- CUDA forbids
``fork`` (a forked child inherits a CUDA context it cannot safely reuse), so
this module never relies on the platform default (``fork`` on Linux). Every
entrypoint takes a ``game_factory: Callable[[], Game]`` rather than a
``Game`` instance or a game name: this keeps ``core/`` importing nothing from
``games/`` (the repo-wide rule), and defers construction to *inside* each
spawned process rather than pickling a possibly-heavy adapter (Blokus's
precomputed bitboard tables) across the process boundary. ``device`` is
always an explicit argument -- CPU in CI, a caller-chosen CUDA device in
production; nothing here inspects ``torch.cuda.is_available()`` to decide.

**Clean shutdown.** :class:`ShutdownFlag` installs a ``SIGTERM``/``SIGINT``
handler that only ever sets an in-process flag -- it is the producer behind
both drivers' existing ``should_stop`` seam, never a new shutdown mechanism.
All actual shutdown *behavior* already lives in the drivers: ``ActorDriver.run``
checks ``should_stop`` between games (never mid-search), so a signaled actor
finishes its in-flight game and flushes its shard before exiting;
``LearnerDriver.run`` checks between steps, so a signaled learner finishes
its in-flight step (train, advance, publish-if-due, write the rolling resume
snapshot) before exiting -- and **never** performs a publish as a
shutdown-specific action, because the only shutdown-specific write this
module ever makes is setting a boolean. A shutdown landing strictly between
two publish-interval boundaries therefore writes ``resume.pt`` and nothing
under ``ckpt-*.pt``. This issue additionally closes three gaps real
concurrent-process wiring exposed in the M3 drivers (all minimal,
backward-compatible, default-``None``-preserving extensions):

1. A pacing hold has no other exit condition, so
   ``ActorDriver._await_pacing`` now polls ``should_stop`` from inside the
   hold too, not only between games.
2. The D5 replay ceiling's wait has the same gap, so
   ``LearnerDriver._enforce_ceiling`` now does the same.
3. A learner started concurrently with (or before) its actors has no other
   guarantee any shard exists yet -- ``core.replay_window.ReplayWindow.sample_batch``
   raises against a genuinely empty window -- so ``LearnerDriver.run`` now
   blocks (``_await_first_data``, also ``should_stop``-aware) until the
   window holds at least one position before training its first step,
   rather than assuming a caller always seeds the window first.

``LaunchedRun.shutdown`` sends ``SIGTERM`` (``Process.terminate()``) to every
live process and joins them -- never ``SIGKILL``, so every process always
runs its own clean path.

**GPU memory envelope (documented, not engineered around -- issue #61's own
constraint).** The D5 trunk is 8 residual blocks × 128 channels; the
learner's training batch (256, AMP) peak-allocates 0.52 of 16.0 GiB (~3%) of
the RTX 4060 Ti's device memory (``docs/bench/m2-train-step.md``). Each
actor process built here runs only batch-1 leaf inference through
``core.network.make_network_evaluator`` (M5 batches inference across actors;
out of scope here) -- a strictly smaller resident footprint than the
learner's own training step, and the learner's headroom above already
leaves ample room for several such actors to share the device concurrently.
No memory scheduler, cgroup, or MPS configuration is built here; §3's
"single consumer GPU" framing is satisfied by this arithmetic, not by code.
"""

from __future__ import annotations

import multiprocessing
import signal
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any

from core.actor import ActorDriver, PacingFn, RefreshFn, RefreshResult, WaitFn
from core.checkpoint import load_checkpoint, published_checkpoint_path, read_latest_pointer
from core.game import Game
from core.learner import PACING_HOLD, LearnerDriver, pacing_file_path, read_pacing_file
from core.metrics import EpochMetricsWriter
from core.network import Network, NetworkConfig, make_network_evaluator
from core.observability import (
    PositionCounter,
    count_positions,
    segment_end_record,
    segment_start_record,
)
from core.runconfig import RunConfig, SelfPlayConfig

# A zero-argument, picklable callable that builds a fresh adapter instance.
# Module-level functions, classes (``TicTacToe`` itself, called with no args),
# and ``functools.partial`` wrappers around either all satisfy this and are
# picklable under the ``spawn`` context; a lambda or closure is not (see the
# module docstring's process-model section).
GameFactory = Callable[[], Game]

# The signals a process-local ShutdownFlag catches by default: SIGTERM is
# what LaunchedRun.shutdown sends; SIGINT covers Ctrl-C on a single process
# run directly (e.g. under a future #63 CLI, or by hand in development).
_DEFAULT_SHUTDOWN_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGTERM, signal.SIGINT)


def _default_wait(poll_interval: float) -> None:
    """The production poll backoff -- a real sleep; every test injects its own.

    Args:
        poll_interval: Seconds to sleep.
    """
    time.sleep(poll_interval)


# --- signal-based clean shutdown --------------------------------------------


class ShutdownFlag:
    """A signal-settable flag realizing both drivers' ``should_stop`` seam.

    The installed handler only ever *sets* an in-process boolean -- no
    cleanup, no driver calls, nothing that could itself fail or re-enter a
    search/training step -- so it is safe to fire at any point in a process's
    lifetime. Every actual shutdown behavior (finishing in-flight work,
    flushing a shard, writing the resume snapshot, never publishing) lives in
    ``core.actor.ActorDriver`` / ``core.learner.LearnerDriver`` themselves;
    this class is only the seam's producer.

    Never pickle an instance across a process boundary -- construct and
    :meth:`install` one *inside* the target process's own entrypoint
    (:func:`run_actor_process` / :func:`run_learner_process`), never in the
    launching process, so the handler is installed in whichever process will
    actually receive the signal.
    """

    def __init__(self) -> None:
        self._flag = False

    def __call__(self) -> bool:
        """Return whether a handled signal has fired -- the ``StopFn`` seam."""
        return self._flag

    def _handle(self, signum: int, frame: Any) -> None:
        """The installed handler: set the flag and do nothing else.

        Args:
            signum: The signal number received (unused; every installed
                signal has the same effect).
            frame: The interrupted stack frame (unused; required by
                ``signal.signal``'s handler contract).
        """
        del signum, frame
        self._flag = True

    def install(self, signals: Sequence[int] = _DEFAULT_SHUTDOWN_SIGNALS) -> ShutdownFlag:
        """Install this flag's handler for ``signals`` in the current process.

        Args:
            signals: Signals to catch. Defaults to ``SIGTERM`` and ``SIGINT``
                (module docstring).

        Returns:
            ``self`` -- so construction and installation chain at process
            start: ``stop = ShutdownFlag().install()``.
        """
        for sig in signals:
            signal.signal(sig, self._handle)
        return self


# --- actor-side wiring: refresh (checkpoint IO) and pacing (pacing-file IO) --


def build_actor_refresh(
    *,
    game: Game,
    ckpt_dir: Path | str,
    device: str = "cpu",
    network_config: NetworkConfig | None = None,
    poll_interval: float = 1.0,
    wait: WaitFn | None = None,
    position_counter: PositionCounter | None = None,
) -> RefreshFn:
    """Build the real ``refresh`` seam: the learner's ``latest`` pointer, live.

    A pure closure over ``ckpt_dir`` -- no process, no signal, no driver
    dependency -- so it is directly unit-testable and reusable by anything
    that wants ``ActorDriver``'s ``RefreshFn`` shape (``core.actor``) without
    the process layer this module also builds.

    Each call rereads :func:`core.checkpoint.read_latest_pointer`. While no
    checkpoint has been published yet, it blocks on ``wait`` and rereads --
    the learner's mandatory version-0 publish at construction
    (``core.learner.LearnerDriver.__init__``) means this is normally a single
    short wait at the very start of a run, not a steady-state condition. Once
    a version is named, its checkpoint is loaded and fingerprint-validated
    via :func:`core.checkpoint.load_checkpoint` -- a mismatch raises
    ``core.artifact_fingerprint.FingerprintMismatchError`` straight through,
    uncaught, exactly the "loud failure" the checkpoint module already
    promises. The built evaluator is cached by version, so repeated calls
    naming the same ``latest`` (the common case -- most games are played
    between publishes, not across one) neither touch the disk again nor
    rebuild a ``Network``.

    Args:
        game: The adapter every loaded checkpoint's fingerprint is validated
            against, and every built evaluator is wired for.
        ckpt_dir: The checkpoint directory (``core.checkpoint``) the learner
            publishes into -- shared with it, read-only from here.
        device: Torch device for inference (``"cpu"`` in CI; module
            docstring's GPU-envelope note).
        network_config: The network architecture to build. **Must match
            whatever the learner this run's checkpoints came from was
            constructed with** (``core.network.NetworkConfig.from_game(game)``
            by default on both sides) -- a caller that overrides one without
            the other gets a ``state_dict`` shape mismatch from
            ``load_state_dict``, not a silent misload.
        poll_interval: Seconds between retries while no checkpoint has been
            published yet. Ignored if ``wait`` is given.
        wait: The backoff strategy invoked while blocked on a first
            checkpoint. Defaults to a real sleep of ``poll_interval``; tests
            inject their own no-sleep fake.
        position_counter: Optional (issue #62). When given, every built
            evaluator is wrapped with ``core.observability.count_positions``
            before caching, so every leaf-inference call this actor makes
            through it adds ``1`` position to the counter (M3's batch-1
            bridge). ``None`` (the default) returns
            ``core.network.make_network_evaluator``'s evaluator unwrapped --
            byte-identical to every call site that predates this parameter.
            Wrapped once per newly-built version, never re-wrapped on a
            cache hit.

    Returns:
        A zero-argument ``RefreshFn`` returning ``(evaluator, model_version)``
        -- ``core.actor.RefreshResult``.
    """
    directory = Path(ckpt_dir)
    net_config = network_config if network_config is not None else NetworkConfig.from_game(game)
    wait_fn = wait if wait is not None else lambda: _default_wait(poll_interval)
    cache: dict[str, Any] = {"version": None, "evaluator": None}

    def refresh() -> RefreshResult:
        version = read_latest_pointer(directory)
        while version is None:
            wait_fn()
            version = read_latest_pointer(directory)
        if version != cache["version"]:
            bundle = load_checkpoint(published_checkpoint_path(directory, version), game)
            net = Network(net_config).to(device)
            net.load_state_dict(bundle.model_state_dict)
            evaluator = make_network_evaluator(net, game, device=device)
            if position_counter is not None:
                evaluator = count_positions(evaluator, position_counter)
            cache["evaluator"] = evaluator
            cache["version"] = version
        return cache["evaluator"], cache["version"]

    return refresh


def build_actor_pacing(run_dir: Path | str) -> PacingFn:
    """Build the real ``pacing`` seam: the learner's pacing file, live.

    A pure closure over ``run_dir`` -- same reusability argument as
    :func:`build_actor_refresh`. Mirrors
    ``core.learner.read_pacing_file``'s documented contract exactly: a
    missing file (fresh start, or the learner still in warm-up) reads as
    "go", never "hold" -- an actor must never be paused by a learner that
    has not yet made a floor decision.

    Args:
        run_dir: The run's root directory (``core.learner.pacing_file_path``).

    Returns:
        A zero-argument ``PacingFn``: ``True`` iff the learner's most recent
        decision is :data:`core.learner.PACING_HOLD`.
    """
    path = pacing_file_path(run_dir)

    def pacing() -> bool:
        payload = read_pacing_file(path)
        if payload is None:
            return False
        return payload["state"] == PACING_HOLD

    return pacing


# --- process entrypoints (module-level: picklable multiprocessing targets) --


def run_actor_process(
    *,
    game_factory: GameFactory,
    self_play: SelfPlayConfig,
    run_id: str,
    actor_id: int,
    shard_dir: Path | str,
    ckpt_dir: Path | str,
    run_dir: Path | str,
    run_seed: int,
    device: str = "cpu",
    network_config: NetworkConfig | None = None,
    refresh_poll_interval: float = 1.0,
    pacing_poll_interval: float = 1.0,
    max_games: int | None = None,
) -> None:
    """One actor's entire process body -- a ``multiprocessing.Process`` target.

    Builds the adapter, installs the signal-based :class:`ShutdownFlag`, wires
    :func:`build_actor_refresh` / :func:`build_actor_pacing` over the shared
    ``ckpt_dir`` / ``run_dir``, constructs ``core.actor.ActorDriver``, and
    runs it to exhaustion (production: forever, until signaled) or to
    ``max_games`` (test-facing). Every argument is picklable so this function
    is a valid ``spawn``-context ``Process`` target (module docstring).

    Every real actor process is observed unconditionally (issue #62: "captured
    live from step zero of every run") -- unlike ``core.actor.ActorDriver``'s
    own ``metrics_writer``/``position_counter`` constructor parameters, which
    stay optional/``None``-default so a driver built directly (most of
    ``tests/test_actor.py``) is unaffected. This process entrypoint opens this
    actor's ``run_dir/metrics/actor-<actor_id>-<epoch>.jsonl`` writer, a fresh
    :class:`~core.observability.PositionCounter`, and wires the counter into
    :func:`build_actor_refresh` so every leaf-inference call this actor makes
    is counted.

    Args:
        game_factory: Builds this process's adapter instance.
        self_play: The D6 validate-tier self-play scalars
            (``core.actor.validate_actor_self_play_config``).
        run_id: This run's identity (constant across every actor/the learner).
        actor_id: This actor's durable identity within the run.
        shard_dir: Directory this actor's shards and writer-state file
            publish into; also the directory the learner's replay window
            scans.
        ckpt_dir: The learner's checkpoint directory (read-only from here).
        run_dir: The run's root directory (the pacing file lives here).
        run_seed: The run's recorded root seed.
        device: Torch device for leaf inference.
        network_config: Network architecture for loaded checkpoints -- must
            match the learner's (see :func:`build_actor_refresh`).
        refresh_poll_interval: Seconds between retries while no checkpoint
            has been published yet.
        pacing_poll_interval: Seconds between pacing-hold retries.
        max_games: Optional cap on games played -- the test-facing stop
            condition; a production actor leaves this unset and relies on
            the installed :class:`ShutdownFlag`.
    """
    game = game_factory()
    stop = ShutdownFlag().install()
    metrics_writer = EpochMetricsWriter(run_dir, f"actor-{actor_id}")
    position_counter = PositionCounter()
    refresh = build_actor_refresh(
        game=game,
        ckpt_dir=ckpt_dir,
        device=device,
        network_config=network_config,
        poll_interval=refresh_poll_interval,
        position_counter=position_counter,
    )
    pacing = build_actor_pacing(run_dir)
    driver = ActorDriver(
        game=game,
        self_play=self_play,
        run_id=run_id,
        actor_id=actor_id,
        out_dir=shard_dir,
        run_seed=run_seed,
        refresh=refresh,
        pacing=pacing,
        wait=lambda: _default_wait(pacing_poll_interval),
        max_games=max_games,
        should_stop=stop,
        metrics_writer=metrics_writer,
        position_counter=position_counter,
    )
    driver.run()


def run_learner_process(
    *,
    game_factory: GameFactory,
    run_config: RunConfig,
    shard_dir: Path | str,
    ckpt_dir: Path | str,
    run_dir: Path | str,
    network_config: NetworkConfig | None = None,
    device: str = "cpu",
    ceiling_poll_interval: float = 1.0,
    max_steps: int | None = None,
) -> None:
    """The learner's entire process body -- a ``multiprocessing.Process`` target.

    Builds the adapter, installs the signal-based :class:`ShutdownFlag`,
    constructs ``core.learner.LearnerDriver`` over the shared ``shard_dir`` /
    ``ckpt_dir`` / ``run_dir``, and runs it to its pinned stop condition
    (``checkpoint_count * publish_interval``) or ``max_steps``
    (test-facing). Every argument is picklable (module docstring).

    Args:
        game_factory: Builds this process's adapter instance.
        run_config: The run's full protocol (``core.runconfig.RunConfig`` or
            an equivalent picklable duck-typed stand-in exposing ``training``/
            ``run_seed``/``to_dict()`` -- see ``core.learner.LearnerDriver``).
        shard_dir: Directory the replay window scans -- shared with every
            actor's ``ShardWriter``.
        ckpt_dir: Directory published/resume checkpoints are written to.
        run_dir: The run's root directory (pacing file, epoch metrics).
        network_config: Network architecture to build -- must match every
            actor's (see :func:`build_actor_refresh`).
        device: Torch device for training.
        ceiling_poll_interval: Seconds between D5 replay-ceiling retries.
        max_steps: Optional cap on steps trained -- the test-facing stop
            condition; a production learner leaves this unset and relies on
            the pinned stop condition plus the installed :class:`ShutdownFlag`.
    """
    game = game_factory()
    stop = ShutdownFlag().install()
    driver = LearnerDriver(
        game=game,
        run_config=run_config,
        shard_dir=shard_dir,
        ckpt_dir=ckpt_dir,
        run_dir=run_dir,
        network_config=network_config,
        device=device,
        wait=lambda: _default_wait(ceiling_poll_interval),
        should_stop=stop,
        max_steps=max_steps,
    )
    driver.run()


# --- the launcher: spawn-context process management ------------------------


def new_spawn_context() -> BaseContext:
    """Return the ``"spawn"`` multiprocessing context, explicitly.

    CUDA forbids ``fork``: a forked child inherits a CUDA context it cannot
    safely reuse (undefined behavior, not merely inefficiency), so every
    process this module starts is spawned explicitly rather than trusting
    the platform default (``fork`` on Linux).

    Returns:
        The ``"spawn"`` context.
    """
    return multiprocessing.get_context("spawn")


def start_actor_process(ctx: BaseContext, **kwargs: Any) -> multiprocessing.process.BaseProcess:
    """Start one actor process under ``ctx`` and return its live handle.

    A thin, reusable building block under :func:`launch_run` -- also the
    right call for restarting a single killed actor under the same
    ``(run_id, actor_id, shard_dir)`` identity, without relaunching the rest
    of the run.

    Args:
        ctx: A multiprocessing context (:func:`new_spawn_context`).
        **kwargs: Forwarded to :func:`run_actor_process`.

    Returns:
        The started process handle.
    """
    process = ctx.Process(
        target=run_actor_process, kwargs=kwargs, name=f"actor-{kwargs.get('actor_id')}"
    )
    process.start()
    return process


def start_learner_process(ctx: BaseContext, **kwargs: Any) -> multiprocessing.process.BaseProcess:
    """Start the learner process under ``ctx`` and return its live handle.

    Args:
        ctx: A multiprocessing context (:func:`new_spawn_context`).
        **kwargs: Forwarded to :func:`run_learner_process`.

    Returns:
        The started process handle.
    """
    process = ctx.Process(target=run_learner_process, kwargs=kwargs, name="learner")
    process.start()
    return process


@dataclass
class LaunchedRun:
    """Live process handles for one launched actor–learner run (:func:`launch_run`).

    Attributes:
        ctx: The spawn context every process was started under -- reused to
            start a replacement process for a restarted actor under the same
            identity (``start_actor_process(launched.ctx, ...)``).
        learner: The learner process handle.
        actors: Live actor process handles, keyed by ``actor_id``. A caller
            restarting a killed actor re-keys this dict with the new handle:
            ``launched.actors[actor_id] = start_actor_process(launched.ctx, **kwargs)``.
        metrics_writer: This run's ``orchestrator`` epoch-metrics writer
            (issue #62) -- the single, coordinator-owned source of GPU-hour
            segment records. :func:`launch_run` already appended this run's
            ``segment_start`` record before returning; :meth:`shutdown`
            appends the matching ``segment_end``.
    """

    ctx: BaseContext
    learner: multiprocessing.process.BaseProcess
    actors: dict[int, multiprocessing.process.BaseProcess]
    metrics_writer: EpochMetricsWriter

    def all_processes(self) -> tuple[multiprocessing.process.BaseProcess, ...]:
        """Return every process handle this run currently tracks."""
        return (self.learner, *self.actors.values())

    def shutdown(self, timeout: float = 30.0) -> None:
        """Send SIGTERM to every live process, join them, and close the GPU-hour segment.

        ``Process.terminate()`` sends ``SIGTERM`` (never ``SIGKILL``) --
        exactly the signal :class:`ShutdownFlag` catches, so every process
        runs its own clean-shutdown path (module docstring) rather than
        being killed mid-write. Every process is signaled before any join,
        so they shut down concurrently rather than one at a time.

        The ``segment_end`` record (issue #62) is appended last, after every
        process has been joined (or timed out) -- it marks the orchestrator's
        own single-counted GPU-hour segment as complete, so it should not be
        written while a process this run started might still be using the
        device. A run whose ``shutdown`` is never called (or that crashes
        first) simply never closes its segment -- ``core.observability
        .reduce_run``'s documented, conservative "an unterminated segment
        contributes zero" rule, not a special case here.

        Args:
            timeout: Seconds to wait for each process to exit on its own
                after being signaled. A process that outlives its timeout is
                left running, not force-killed -- a caller wanting a hard
                stop calls ``process.kill()`` itself.
        """
        for process in self.all_processes():
            if process.is_alive():
                process.terminate()
        for process in self.all_processes():
            process.join(timeout)
        self.metrics_writer.append(segment_end_record())


def launch_run(
    *,
    game_factory: GameFactory,
    run_config: RunConfig,
    self_play: SelfPlayConfig,
    run_id: str,
    num_actors: int,
    shard_dir: Path | str,
    ckpt_dir: Path | str,
    run_dir: Path | str,
    run_seed: int,
    device: str = "cpu",
    network_config: NetworkConfig | None = None,
    actor_ids: Sequence[int] | None = None,
    refresh_poll_interval: float = 1.0,
    pacing_poll_interval: float = 1.0,
    ceiling_poll_interval: float = 1.0,
    max_games_per_actor: int | None = None,
    max_learner_steps: int | None = None,
) -> LaunchedRun:
    """Start ``num_actors`` actor processes and one learner process.

    All processes share ``shard_dir`` / ``ckpt_dir`` / ``run_dir`` -- the
    filesystem artifacts that are this module's entire coordination surface
    (module docstring) -- and are started under one ``spawn`` context
    (:func:`new_spawn_context`). This is the ``#63`` run-entrypoint issue's
    launch primitive: the CLI it builds resolves a ``RunConfig`` and calls
    this; this module makes no config-file or CLI-argument decisions itself.

    This is also where the run's GPU-hour segment starts (issue #62): the
    orchestrator -- this calling process, never a spawned child -- is the one
    entity that knows about every process sharing the device, so it is the
    only sanctioned writer of GPU-hour records (module docstring's
    "single-counted regardless of process count"). The ``segment_start``
    record is appended before any child process starts, so the segment's
    span is a strict superset of every actor/learner process's lifetime;
    :meth:`LaunchedRun.shutdown` appends the matching ``segment_end``.

    Args:
        game_factory: Builds each process's own adapter instance (called
            once per process, not shared).
        run_config: The learner's full protocol.
        self_play: Every actor's D6 validate-tier self-play scalars.
        run_id: This run's identity.
        num_actors: Number of actor processes to start. Ignored if
            ``actor_ids`` is given.
        shard_dir: Shared replay-shard directory.
        ckpt_dir: Shared checkpoint directory.
        run_dir: Shared run root directory.
        run_seed: The run's recorded root seed.
        device: Torch device for every process (CPU in CI).
        network_config: Network architecture shared by every process. Must
            agree with whatever ``run_config``'s learner actually trains
            (see :func:`build_actor_refresh`).
        actor_ids: Explicit actor identities to start, e.g. when resuming a
            run with a specific surviving set. Defaults to
            ``range(num_actors)``.
        refresh_poll_interval: Forwarded to every actor.
        pacing_poll_interval: Forwarded to every actor.
        ceiling_poll_interval: Forwarded to the learner.
        max_games_per_actor: Forwarded to every actor (test-facing).
        max_learner_steps: Forwarded to the learner (test-facing).

    Returns:
        A :class:`LaunchedRun` with live handles for the learner and every
        started actor.
    """
    metrics_writer = EpochMetricsWriter(run_dir, "orchestrator")
    metrics_writer.append(segment_start_record(device=device))
    ctx = new_spawn_context()
    learner = start_learner_process(
        ctx,
        game_factory=game_factory,
        run_config=run_config,
        shard_dir=shard_dir,
        ckpt_dir=ckpt_dir,
        run_dir=run_dir,
        network_config=network_config,
        device=device,
        ceiling_poll_interval=ceiling_poll_interval,
        max_steps=max_learner_steps,
    )
    ids = tuple(actor_ids) if actor_ids is not None else tuple(range(num_actors))
    actors = {
        actor_id: start_actor_process(
            ctx,
            game_factory=game_factory,
            self_play=self_play,
            run_id=run_id,
            actor_id=actor_id,
            shard_dir=shard_dir,
            ckpt_dir=ckpt_dir,
            run_dir=run_dir,
            run_seed=run_seed,
            device=device,
            network_config=network_config,
            refresh_poll_interval=refresh_poll_interval,
            pacing_poll_interval=pacing_poll_interval,
            max_games=max_games_per_actor,
        )
        for actor_id in ids
    }
    return LaunchedRun(ctx=ctx, learner=learner, actors=actors, metrics_writer=metrics_writer)
