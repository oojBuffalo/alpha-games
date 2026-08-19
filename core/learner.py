"""The M3 learner driver (design doc §6.2/§7/§10, §12 M3, issue #60).

Draws seeded batches off ``core.replay_window.ReplayWindow``, augments each
sample through the existing D9 machinery, runs the existing D5 train step
(``core.train.train_step``), holds the D5 replay ratio inside ``[2, 4]``, and
periodically bumps the model version and publishes an immutable checkpoint
(``core.checkpoint``) plus an atomic ``latest`` pointer -- mirroring
``core.actor.ActorDriver``'s driver style: injectable seams, no real sleeps
in tests, one process's worth of durable state reloaded from disk rather
than kept in a second in-memory copy.

**Ratio arithmetic** (``core.replay_window``'s two exposed totals, issue #55).
``ratio = samples_drawn / positions_stored`` with ``samples_drawn =
learner_step * batch_size`` (:func:`core.replay_window.samples_drawn` --
derived, never a separate counter) and ``positions_stored`` the window's
monotone ingestion total. Enforcement runs every step, in two independent
directions:

* **Ceiling (4), actor-starved direction:** before training the next step,
  :meth:`LearnerDriver._enforce_ceiling` checks whether *taking* that step
  would push the ratio above 4; if so it blocks on the injectable ``wait``
  strategy, rescanning the window for newly-ingested shards between waits,
  until ingestion has caught the ratio back under the ceiling (or forever, in
  production, if actors truly stall -- exactly the blocking behavior the
  issue specifies).
* **Floor (2), learner-ahead direction:** the learner never blocks itself on
  this side -- storing further only digs the ratio down, so the learner
  keeps training regardless. Instead, after every step,
  :meth:`LearnerDriver._enforce_floor` writes/refreshes the pacing file
  (:func:`write_pacing_file`) to ``"hold"`` while the ratio sits below the
  floor, and back to ``"go"`` once it recovers -- an exact threshold, no
  hysteresis (nothing in the load-bearing test showed flapping that would
  justify one).

**Warm-up** (an implementation-level knob, not a design-doc-pinned scalar:
``training.replay_warmup_positions``). Both enforcements are skipped
entirely while ``positions_stored`` is below the configured minimum, so a
freshly started run -- window still filling, ratio arithmetic degenerate
against a near-empty denominator -- cannot signal "hold" to actors before
they have produced anything (the exact deadlock the design constraints call
out).

**Publication and the exactly-once marker.** :meth:`LearnerDriver._maybe_publish`
is called once at the end of ``__init__`` and once after every trained step;
it is a fast no-op unless ``self.step`` sits on a ``publish_interval``
boundary (which includes step 0 -- ``0 % anything == 0`` -- covering the
seeded init's mandatory version-0 publish at fresh startup for free, with no
separate special case). At a boundary it derives ``version = self.step //
publish_interval`` and makes two independent things durable if they are not
already: the immutable checkpoint file
(``core.checkpoint.write_published_checkpoint``, itself a no-op-if-exists
check) plus the atomic ``latest`` pointer, and a ``checkpoint_published``
marker record (``{kind, model_version, learner_step, timestamp}``) appended
to the learner's own per-process epoch metrics file
(``core.metrics.EpochMetricsWriter``, tasks/m3/011's naming convention).
Both checks are independently idempotent -- reordering them, calling the
method twice in a row, or crashing at any point between the checkpoint write
and the marker append and then resuming, can never drop or duplicate either
one. This is also why a resumed driver never needs a special "was this
fresh or a resume" branch anywhere in the publish path: the same
``_maybe_publish`` call, run once at construction against whatever
``self.step`` resume selection restored, transparently completes any publish
a crash left half-finished (the straddle case the load-bearing test proves)
and is a silent no-op otherwise.

**Resume transparency.** ``__init__`` always calls
``core.checkpoint.select_resume_bundle`` first; ``None`` means a genuinely
fresh start (net seeded from ``core.seeding.net_init_seed``, optimizer/scaler
freshly built, ``self.step = 0``), otherwise the net/optimizer/scaler state
dicts are loaded and the LR schedule is fast-forwarded by calling
``.step()`` ``learner_step`` times (the exact pattern
``tests/test_checkpoint.py``'s golden already proved bit-for-bit equivalent
to a continuously-stepped schedule -- see that module's docstring). Either
way, by the time ``__init__`` returns, the loop has no memory of which path
it took: the same per-step stream sequence
(``core.seeding.LearnerRNGs.for_step(run_seed, step)``) is reproduced solely
from ``(run_seed, self.step)``, never from anything held in-process across
the restart. The rolling resume snapshot (``resume.pt``) is written every
step (:meth:`LearnerDriver._write_resume_snapshot`, called from
:meth:`LearnerDriver.run` via :meth:`LearnerDriver._run_step`) -- the
simplest cadence that still lets a test simulate the marker-straddle case,
by calling :meth:`LearnerDriver._advance_one_step` (everything except the
snapshot write) directly instead of a full :meth:`LearnerDriver._run_step`.

**Augmentation and the empty-group decision (D9).** Per sampled record,
:meth:`LearnerDriver._train_one_step` draws one symmetry index from the
step's seeded ``LearnerRNGs.for_step(...).augmentation`` stream
(``core.seeding``) and applies it via ``core.augment.augment_sample`` --
*only* when ``len(game.symmetry_group) > 0``. A game declaring no symmetry
group (Tic-Tac-Toe, Connect 4) is treated as identity-only: the draw is
**skipped entirely**, not drawn-and-discarded. Skipping (rather than
drawing a value from an empty range and discarding it, which would raise
``ValueError`` from ``random.randrange``) keeps every game's augmentation
stream consumption a pure function of "how many symmetries does this game
actually have" -- an empty-group game's per-step stream is simply never
touched, so adding an aux draw elsewhere can never accidentally desync a
no-augmentation game's determinism against one with a real group.

**Stop condition.** ``total_steps = checkpoint_count * publish_interval``
exactly (§6.2's pinned arithmetic); :meth:`LearnerDriver.run` stops the
instant ``self.step`` reaches it, never rounding up to finish an
in-progress interval. ``max_steps``/``should_stop`` are additional,
test-facing stop conditions layered on top (mirroring
``core.actor.ActorDriver``'s exact seam), letting a test halt a driver
earlier than its production stop condition to simulate a kill.
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from core.augment import augment_sample
from core.checkpoint import (
    CheckpointBundle,
    build_bundle,
    list_published_versions,
    newest_published_version,
    select_resume_bundle,
    write_latest_pointer,
    write_published_checkpoint,
    write_resume_snapshot,
)
from core.game import Game
from core.losses import LossBreakdown
from core.metrics import EpochMetricsWriter, iter_epoch_records
from core.network import Network, NetworkConfig
from core.observability import (
    CHECKPOINT_PUBLISHED_KIND,
    gauge_record,
    total_record,
)
from core.replay_shard import SampleRecord, _atomic_write_json
from core.replay_window import ReplayWindow, samples_drawn
from core.runconfig import RunConfig
from core.seeding import LearnerRNGs, net_init_seed
from core.train import collate, make_lr_scheduler, make_optimizer, make_scaler, train_step

# D5 (§7, §10): "2-4 samples per stored position" -- the replay-ratio band
# the learner holds from both sides (module docstring).
REPLAY_RATIO_FLOOR = 2.0
REPLAY_RATIO_CEILING = 4.0

# This process's name in the tasks/m3/011 epoch-metrics naming convention
# (core.metrics) -- one learner per run, so no id suffix.
LEARNER_PROC = "learner"

# CHECKPOINT_PUBLISHED_KIND is defined in core.observability (issue #62 owns
# the full kind taxonomy) and re-exported here unchanged so existing
# ``from core.learner import CHECKPOINT_PUBLISHED_KIND`` call sites keep
# working verbatim.

# The learner-owned series this module emits every trained step (issue #62).
# ``learner_step`` is the exact ``total`` series; the rest are ``gauge``s,
# never summed by the reducer -- see core.observability's module docstring
# for the full kind taxonomy.
SERIES_LEARNER_STEP = "learner_step"
SERIES_LOSS_TOTAL = "loss_total"
SERIES_LOSS_VALUE = "loss_value"
SERIES_LOSS_POLICY = "loss_policy"
SERIES_LOSS_AUX = "loss_aux"
SERIES_REPLAY_RATIO = "replay_ratio"

PACING_FILENAME = "pacing.json"
PACING_HOLD = "hold"
PACING_GO = "go"

WaitFn = Callable[[], None]
StopFn = Callable[[], bool]


def _default_wait() -> None:
    """The production ceiling-block backoff.

    A real (short) sleep -- every test supplies its own ``wait`` instead, so
    this is never exercised off a driver's default construction path.
    """
    time.sleep(1.0)


# --- pacing file: learner-owned, atomic temp+rename JSON ---------------------


def pacing_file_path(run_dir: Path | str) -> Path:
    """Return the run-shared pacing file path.

    Args:
        run_dir: The run's root directory.

    Returns:
        ``run_dir / "pacing.json"``.
    """
    return Path(run_dir) / PACING_FILENAME


def write_pacing_file(
    path: Path | str,
    *,
    state: str,
    ratio: float,
    positions_stored: int,
    samples_drawn: int,
    learner_step: int,
) -> None:
    """Write the pacing file atomically (temp-name-then-``os.replace``).

    Args:
        path: The pacing file path (:func:`pacing_file_path`).
        state: :data:`PACING_HOLD` or :data:`PACING_GO`.
        ratio: The replay ratio this decision was made from.
        positions_stored: The window's ``positions_stored`` at decision time.
        samples_drawn: The learner's ``samples_drawn`` at decision time.
        learner_step: The learner step this decision was made at.
    """
    payload = {
        "state": state,
        "ratio": ratio,
        "positions_stored": positions_stored,
        "samples_drawn": samples_drawn,
        "learner_step": learner_step,
    }
    _atomic_write_json(Path(path), payload)


def read_pacing_file(path: Path | str) -> dict[str, Any] | None:
    """Read the learner's pacing file, or ``None`` if never written yet.

    Args:
        path: The pacing file path (:func:`pacing_file_path`).

    Returns:
        The parsed payload (``{"state", "ratio", "positions_stored",
        "samples_drawn", "learner_step"}``), or ``None`` before the
        learner's first post-warm-up floor decision (fresh start, or still
        in warm-up). Issue #61's ``core.actor.ActorDriver`` ``pacing`` adapter
        should treat a missing file the same as an explicit ``"go"``, never
        as a hold.
    """
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _checkpoint_published_record(version: int, learner_step: int) -> dict[str, Any]:
    """Build one ``checkpoint_published`` marker record.

    Args:
        version: The model version just published.
        learner_step: The learner step the publish happened at.

    Returns:
        ``{"kind", "model_version", "learner_step", "timestamp"}``.
        ``timestamp`` is informational only -- never part of any equality
        assertion (module docstring).
    """
    return {
        "kind": CHECKPOINT_PUBLISHED_KIND,
        "model_version": version,
        "learner_step": learner_step,
        "timestamp": time.time(),
    }


class LearnerDriver:
    """Drives the learner side of the actor-learner split (§6.2/§7, §12 M3).

    Args:
        game: The adapter to train against. Never imported by this module.
        run_config: The run's full protocol; this driver reads
            ``run_config.training`` (batch size, LR schedule, replay-window
            capacity, publish interval, checkpoint count, replay warm-up
            minimum) and ``run_config.run_seed``.
        shard_dir: Directory the replay window scans -- shared with the
            actors' ``core.replay_shard.ShardWriter``s.
        ckpt_dir: Directory published/resume checkpoints live in
            (``core.checkpoint``).
        run_dir: The run's root directory: the pacing file
            (:func:`pacing_file_path`) and this process's epoch metrics
            files (``run_dir/metrics/learner-<epoch>.jsonl``,
            ``core.metrics``) live here.
        network_config: Network architecture to build. Defaults to
            ``core.network.NetworkConfig.from_game(game)`` (the D5 8x128
            trunk); tests pass a tiny override.
        device: ``"cpu"`` or ``"cuda"`` -- forwarded to
            ``core.train.make_scaler`` and used to place the net/batches.
        wait: The backoff strategy invoked while the ceiling holds. Defaults
            to a short real sleep -- every test supplies its own no-sleep
            fake.
        should_stop: Optional callable polled once per step, alongside
            ``max_steps``; :meth:`run` stops as soon as either it, the
            production stop condition, or ``max_steps`` says to.
        max_steps: Optional cap on steps run by one :meth:`run` call -- the
            test-facing stop condition, mirroring
            ``core.actor.ActorDriver``'s ``max_games``. A production learner
            leaves this unset and relies on the pinned
            ``checkpoint_count * publish_interval`` stop condition.

    Raises:
        ValueError: If ``max_steps`` is not positive.
        core.artifact_fingerprint.FingerprintMismatchError: Propagated from
            ``core.checkpoint.select_resume_bundle`` if a resume candidate's
            stored fingerprint disagrees with ``game``'s live one.
    """

    def __init__(
        self,
        *,
        game: Game,
        run_config: RunConfig,
        shard_dir: Path | str,
        ckpt_dir: Path | str,
        run_dir: Path | str,
        network_config: NetworkConfig | None = None,
        device: str = "cpu",
        wait: WaitFn | None = None,
        should_stop: StopFn | None = None,
        max_steps: int | None = None,
    ) -> None:
        if max_steps is not None and max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")

        training = run_config.training
        self.game = game
        self.run_config = run_config
        self.run_seed = run_config.run_seed
        self.batch_size = training.batch_size
        self.publish_interval = training.publish_interval
        self.checkpoint_count = training.checkpoint_count
        self.total_steps = training.checkpoint_count * training.publish_interval
        self.replay_warmup_positions = training.replay_warmup_positions
        self.num_aux = len(game.value_targets.aux_names)
        self.group_size = len(game.symmetry_group)
        self.device = device
        self.wait = wait if wait is not None else _default_wait
        self.should_stop = should_stop
        self.max_steps = max_steps

        self.shard_dir = Path(shard_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.run_dir = Path(run_dir)
        self.pacing_path = pacing_file_path(self.run_dir)

        self.window = ReplayWindow(self.shard_dir, game, capacity=training.replay_window)
        self.epoch_writer = EpochMetricsWriter(self.run_dir, LEARNER_PROC)

        net_config = network_config if network_config is not None else NetworkConfig.from_game(game)
        resumed = select_resume_bundle(self.ckpt_dir, game)
        if resumed is None:
            torch.manual_seed(net_init_seed(self.run_seed))
            self.net = Network(net_config).to(device)
            self.optimizer = make_optimizer(self.net, lr=training.learning_rate)
            self.scaler = make_scaler(device)
            self.scheduler = make_lr_scheduler(
                self.optimizer, training.warmup_steps, training.cosine_total_steps
            )
            self.step = 0
        else:
            self.net = Network(net_config).to(device)
            self.net.load_state_dict(resumed.model_state_dict)
            self.optimizer = make_optimizer(self.net, lr=training.learning_rate)
            self.optimizer.load_state_dict(resumed.optimizer_state_dict)
            self.scaler = make_scaler(device)
            self.scaler.load_state_dict(resumed.scaler_state_dict)
            self.scheduler = make_lr_scheduler(
                self.optimizer, training.warmup_steps, training.cosine_total_steps
            )
            self.step = resumed.learner_step
            with warnings.catch_warnings():
                # torch warns when a scheduler is fast-forwarded without an
                # interleaved optimizer.step() -- benign here; the resulting
                # LR is bit-for-bit what a continuously-stepped schedule
                # would show at this step (tests/test_checkpoint.py's
                # golden proves it).
                warnings.simplefilter("ignore", UserWarning)
                for _ in range(self.step):
                    self.scheduler.step()

        # The learner-owned metrics high-water (issue #62): the opaque
        # ``metrics`` dict seam issue #56 built for exactly this. A fresh
        # start has nothing to restore (``{}``); a resume seeds it from the
        # winning bundle's own snapshot, so a query made before this
        # process's first new step still reflects the pre-crash state rather
        # than reading as freshly empty -- "restored totals continue, not
        # reset" (module docstring's resume-transparency section applies
        # here too).
        self._high_water: dict[str, float] = dict(resumed.metrics) if resumed is not None else {}

        newest = newest_published_version(self.ckpt_dir)
        self.version = newest if newest is not None else 0
        # Completes any publish a prior process left unfinished at this
        # exact step (including the mandatory version-0 publish on a
        # genuinely fresh start); a no-op otherwise. See the module
        # docstring's "Publication and the exactly-once marker" section.
        self._maybe_publish()

    # --- the main loop -----------------------------------------------------

    def run(self) -> list[LossBreakdown]:
        """Train steps until a stop condition triggers.

        Blocks (:meth:`_await_first_data`) before each step until the
        replay window holds at least one position -- unlike the D5 ratio
        band, warm-up never waives this: a freshly started learner racing a
        freshly started actor (issue #61's concurrent-process wiring) has no
        other guarantee that any shard has been ingested yet by the time
        this runs.

        Returns:
            The per-step ``LossBreakdown``s produced by this call, in step
            order.
        """
        results: list[LossBreakdown] = []
        while not self._stop_requested(len(results)):
            if not self._await_first_data():
                break  # should_stop fired before any data ever arrived
            results.append(self._run_step())
        return results

    def _await_first_data(self) -> bool:
        """Block (via ``wait``) until the window holds at least one position.

        Training's first step needs *some* data to sample a batch from --
        unlike the D5 ratio band, this is never waived by warm-up (the
        module docstring's warm-up section is only about ratio
        *enforcement*, not about whether training can start at all;
        ``core.replay_window.ReplayWindow.sample_batch`` raises against a
        genuinely empty window). A concurrently started learner has no other
        guarantee an actor has written its first shard yet (issue #61), so
        this is exactly where that startup race is resolved -- cheaply, a
        single no-op rescan per step once the first shard has landed, since
        ``positions_stored`` only grows.

        Returns:
            ``True`` once the window holds at least one position. ``False``
            if ``should_stop`` fires first -- the caller must not train a
            step against a window that might still be empty.
        """
        self.window.rescan()
        while self.window.positions_stored == 0:
            if self.should_stop is not None and self.should_stop():
                return False
            self.wait()
            self.window.rescan()
        return True

    def _stop_requested(self, steps_this_call: int) -> bool:
        """Return whether :meth:`run` should stop before training another step.

        Args:
            steps_this_call: Steps already trained by the current :meth:`run`
                call.

        Returns:
            ``True`` if the pinned ``total_steps`` has been reached, or
            ``max_steps``/``should_stop`` says to stop.
        """
        if self.step >= self.total_steps:
            return True
        if self.max_steps is not None and steps_this_call >= self.max_steps:
            return True
        return self.should_stop is not None and self.should_stop()

    def _run_step(self) -> LossBreakdown:
        """Train one step and write the rolling resume snapshot.

        Returns:
            The step's ``LossBreakdown``.
        """
        parts = self._advance_one_step()
        self._write_resume_snapshot()
        return parts

    def _advance_one_step(self) -> LossBreakdown:
        """Everything one step does except the rolling resume-snapshot write.

        Exposed separately from :meth:`_run_step` so a test can simulate a
        crash landing between a publish and its following resume snapshot
        (the marker-straddle case) by calling this directly and never
        calling :meth:`_write_resume_snapshot` for that step.

        Returns:
            The step's ``LossBreakdown``.
        """
        self._enforce_ceiling()
        parts = self._train_one_step()
        self.step += 1
        self.scheduler.step()
        self._enforce_floor()
        self._flush_step_metrics(parts)
        self._maybe_publish()
        return parts

    # --- D5 replay-ratio enforcement ----------------------------------------

    def _enforce_ceiling(self) -> None:
        """Block on ``wait`` while taking the next step would exceed the D5 ceiling.

        Rescans the window before checking (and again between waits), so a
        real actor publishing new shards is what clears the block -- the
        "waits for ingestion" half of the D5 band. A no-op during warm-up
        (module docstring).

        Also returns early if ``should_stop`` fires while blocked -- without
        this, a learner waiting here for actors that have genuinely stalled
        could never observe a shutdown signal (issue #61). Returning early
        lets the in-flight step it was called from finish (train, advance,
        publish-if-due, snapshot) exactly as the module docstring's
        "Stop condition" section already promises; the ceiling is briefly
        exceeded rather than the process hanging forever on exit.
        """
        self.window.rescan()
        if self.window.positions_stored < self.replay_warmup_positions:
            return
        while True:
            stored = self.window.positions_stored
            prospective = samples_drawn(self.step + 1, self.batch_size)
            if prospective / stored <= REPLAY_RATIO_CEILING:
                return
            if self.should_stop is not None and self.should_stop():
                return
            self.wait()
            self.window.rescan()

    def _enforce_floor(self) -> None:
        """Refresh the pacing file's hold/go state from the current D5 ratio.

        Called after training (``self.step`` already reflects the completed
        step). A no-op during warm-up -- no pacing file is written at all
        until the window has cleared the configured minimum.
        """
        stored = self.window.positions_stored
        if stored < self.replay_warmup_positions:
            return
        drawn = samples_drawn(self.step, self.batch_size)
        ratio = drawn / stored
        state = PACING_HOLD if ratio < REPLAY_RATIO_FLOOR else PACING_GO
        write_pacing_file(
            self.pacing_path,
            state=state,
            ratio=ratio,
            positions_stored=stored,
            samples_drawn=drawn,
            learner_step=self.step,
        )

    # --- observability: per-step total/gauge series (issue #62) ------------

    def _flush_step_metrics(self, parts: LossBreakdown) -> None:
        """Append this step's total/gauge series and refresh the in-memory high-water.

        Called once per trained step, after ``self.step`` already reflects
        the completed step -- unconditional, unlike ``ActorDriver``'s opt-in
        metrics wiring: this driver already owns ``self.epoch_writer``
        unconditionally (issue #60's ``checkpoint_published`` marker), so
        extending it with more series is "more of the same", not a new
        optional seam. ``learner_step`` is the exact ``total`` series; the
        loss components and the D5 replay ratio are ``gauge``s -- last-in-
        time-order, never summed by the reducer (``core.observability``).
        ``loss_aux`` and ``replay_ratio`` are appended only when there is
        something real to report (an aux head exists; the window is
        non-empty) -- never a fabricated placeholder value.

        ``self._high_water`` mirrors exactly what was just appended (minus
        ``learner_step``, which :meth:`_build_bundle` always derives fresh
        from ``self.step`` instead) so the *next* checkpoint bundle --
        published or the rolling resume snapshot -- carries it forward, and
        a resumed driver's ``_high_water`` is never emptier than what the
        winning bundle already captured.

        Args:
            parts: This step's ``LossBreakdown`` (``core.train.train_step``).
        """
        now = time.time()
        self.epoch_writer.append(total_record(SERIES_LEARNER_STEP, self.step, timestamp=now))
        for series, value in (
            (SERIES_LOSS_TOTAL, parts.total),
            (SERIES_LOSS_VALUE, parts.value),
            (SERIES_LOSS_POLICY, parts.policy),
        ):
            v = float(value)
            self._high_water[series] = v
            self.epoch_writer.append(gauge_record(series, v, timestamp=now))
        if parts.aux is not None:
            v = float(parts.aux)
            self._high_water[SERIES_LOSS_AUX] = v
            self.epoch_writer.append(gauge_record(SERIES_LOSS_AUX, v, timestamp=now))
        stored = self.window.positions_stored
        if stored > 0:
            ratio = samples_drawn(self.step, self.batch_size) / stored
            self._high_water[SERIES_REPLAY_RATIO] = ratio
            self.epoch_writer.append(gauge_record(SERIES_REPLAY_RATIO, ratio, timestamp=now))

    # --- one train step: sample, augment, collate, train_step ---------------

    def _training_row(self, record: SampleRecord, planes: Any, sparse_pi: Any) -> tuple[Any, ...]:
        """Return one record in ``core.train.collate``'s spec-driven row shape.

        Args:
            record: The (possibly still-original) sample record; only
                ``z``/``aux`` are read from it -- ``planes``/``sparse_pi`` are
                taken as separate arguments since augmentation may have
                already transformed them.
            planes: This sample's (possibly augmented) planes.
            sparse_pi: This sample's (possibly augmented) sparse policy pairs.

        Returns:
            ``(planes, sparse_pi, z, aux)`` when the game declares aux heads,
            ``(planes, sparse_pi, z)`` otherwise.
        """
        if self.num_aux:
            return (planes, sparse_pi, record.z, record.aux)
        return (planes, sparse_pi, record.z)

    def _train_one_step(self) -> LossBreakdown:
        """Draw this step's seeded batch, augment it, and run one train step.

        Returns:
            The step's ``LossBreakdown`` (``core.train.train_step``).
        """
        rngs = LearnerRNGs.for_step(self.run_seed, self.step)
        records = self.window.sample_batch(self.run_seed, self.step, self.batch_size)
        rows = []
        for record in records:
            planes, sparse_pi = record.planes, record.sparse_pi
            if self.group_size:
                g_index = rngs.augmentation.randrange(self.group_size)
                planes, sparse_pi = augment_sample(self.game, planes, sparse_pi, g_index)
                sparse_pi = tuple(sparse_pi)
            # An empty symmetry_group (e.g. Tic-Tac-Toe, Connect 4) is
            # identity-only: the augmentation draw above is skipped
            # entirely, never drawn-and-discarded (module docstring).
            rows.append(self._training_row(record, planes, sparse_pi))
        batch = collate(self.game, rows)
        if self.device != "cpu":
            batch = batch.to(self.device)
        return train_step(self.net, self.optimizer, self.scaler, batch)

    # --- publication + the checkpoint_published marker ----------------------

    def _build_bundle(self, version: int) -> CheckpointBundle:
        """Snapshot the live training objects into a bundle for ``version``.

        ``metrics`` carries the learner's own high-water snapshot (issue
        #62): ``self._high_water`` (the loss gauges / replay ratio as of the
        most recent flushed step) plus ``learner_step`` computed fresh from
        ``self.step`` every call -- never read from ``self._high_water``
        itself, so it is always current even at a fresh step-0 build, before
        :meth:`_flush_step_metrics` has ever run.

        Args:
            version: The model-version ordinal to stamp on the bundle.

        Returns:
            The assembled bundle (``core.checkpoint.build_bundle``).
        """
        metrics = dict(self._high_water)
        metrics[SERIES_LEARNER_STEP] = float(self.step)
        return build_bundle(
            version=version,
            learner_step=self.step,
            game=self.game,
            run_config=self.run_config.to_dict(),
            net=self.net,
            optimizer=self.optimizer,
            scaler=self.scaler,
            metrics=metrics,
        )

    def _marker_published(self, version: int) -> bool:
        """Return whether a ``checkpoint_published`` marker for ``version`` already exists.

        Scans every one of this learner's epoch files (every restart, not
        just the current process's), so a marker written before a crash is
        never re-appended after a resume.

        Args:
            version: The model version to check for.

        Returns:
            ``True`` iff a matching marker has already been durably appended.
        """
        return any(
            rec.get("kind") == CHECKPOINT_PUBLISHED_KIND and rec.get("model_version") == version
            for rec in iter_epoch_records(self.run_dir, LEARNER_PROC)
        )

    def _maybe_publish(self) -> None:
        """Publish ``self.step``'s version if due, and ensure its marker exists.

        A fast no-op unless ``self.step`` sits on a ``publish_interval``
        boundary (module docstring: this includes step 0, so it is also
        what performs the seeded init's mandatory version-0 publish). Both
        the checkpoint/latest-pointer write and the marker append are
        independently idempotent, so calling this at construction time
        (covering a resume that straddled a publish) and again after every
        trained step can never double-publish a version or double-append its
        marker.
        """
        if self.step % self.publish_interval != 0:
            return
        version = self.step // self.publish_interval
        if version not in list_published_versions(self.ckpt_dir):
            bundle = self._build_bundle(version)
            write_published_checkpoint(self.ckpt_dir, bundle)
            write_latest_pointer(self.ckpt_dir, version)
        self.version = version
        if not self._marker_published(version):
            self.epoch_writer.append(_checkpoint_published_record(version, self.step))

    def _write_resume_snapshot(self) -> None:
        """Overwrite the rolling resume snapshot with the live training state."""
        write_resume_snapshot(self.ckpt_dir, self._build_bundle(self.version))
