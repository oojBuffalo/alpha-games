"""Run the pinned M2.5 micro-Blokus loop end to end (§12 M2.5, task 6.4).

The learner half of the minimal loop, and its driver. ``core/selfplay.py`` owns
the game-generic, stdlib-pure self-play side (D7/D10/D12 records, replay window,
run record); the torch-dependent assembly lives here because the pyproject pin
confines torch to ``core/network.py`` / ``core/losses.py`` / ``core/train.py``
inside the installed package — dev-only ``scripts/`` may import it.

The loop is a fixed games:steps alternation from config: play one self-play game
into the replay window, then run ``training.steps_per_game`` learner steps —
each drawing a uniform batch from the window, applying one D9 symmetry element
per sample (uniform over the adapter's *declared* group, so the micro instance's
Klein-4 arrives from ``games/`` and not from a constant here), collating, and
taking one D5 AMP train step (AMP is a no-op off CUDA, so this runs on CPU).

**Both halves run at the same precision.** :class:`AmpMode` resolves AMP once
per run and drives *both* the learner step's autocast (via ``core.train``'s
device-derived setting) and the self-play leaf inference (via
:func:`amp_evaluator`). Leaving leaf inference in FP32 while the learner trains
under autocast would make the loop's measured cost — the thing
``scripts/bench_micro_throughput.py`` times through this very module — describe
a configuration nothing actually runs.

Two artifacts land in the run dir:

* ``run_record.json`` — the persisted evidence task 7's exit gate reads. Per
  learner step: policy / value / aux / total loss, LR, window occupancy, games
  played. Plus seeds, config identity, the game's orientation-table hash, and
  the checkpoint identities.
* ``checkpoint_*.pt`` — ``torch.save`` of weights + the run config dict + the
  micro orientation hash + the run seed (Invariant 4; M3 adds validate-on-load,
  optimizer state, and schema versioning).

Usage::

    python3 scripts/run_micro.py                       # the pinned config
    python3 scripts/run_micro.py --run-dir runs/try1 --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.augment import augment_sample  # noqa: E402
from core.game import Game  # noqa: E402
from core.mcts import Evaluator  # noqa: E402
from core.network import Network, NetworkConfig, make_network_evaluator  # noqa: E402
from core.runconfig import MICRO_RUN_CONFIG_PATH, RunConfig, load_run_config  # noqa: E402
from core.seeding import GameRNGs, LearnerRNGs, net_init_seed  # noqa: E402
from core.selfplay import ReplayWindow, RunRecord, play_game  # noqa: E402
from core.train import (  # noqa: E402
    collate,
    make_lr_scheduler,
    make_optimizer,
    make_scaler,
    train_step,
)
from games.blokus_duo import BlokusDuo  # noqa: E402
from games.blokus_duo.pieces import orientation_table_hash  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Checkpoint schema tag. M2.5's checkpoint is deliberately minimal (weights +
# identity); M3 supersedes it with optimizer state, versioning, and
# validate-on-load — hence a tag from the start.
CHECKPOINT_SCHEMA = "alpha-games/micro-checkpoint/v1"

RUN_RECORD_NAME = "run_record.json"


def build_game(cfg: RunConfig) -> Game:
    """Build the adapter this run config names.

    The registry in ``core.runconfig`` resolves the *instance config*; turning
    that into an adapter needs the game package, which only a driver may import
    (``core/`` stays game-agnostic).

    Args:
        cfg: The run config.

    Returns:
        The constructed adapter (Blokus: the bitboard engine for that instance).

    Raises:
        ValueError: If the config names a game this driver cannot construct.
    """
    if cfg.game != "blokus_duo":
        raise ValueError(f"scripts/run_micro.py drives blokus_duo only; config names {cfg.game!r}")
    return BlokusDuo(config=cfg.resolve_game_config())


def game_identity(cfg: RunConfig) -> dict[str, str]:
    """Return the identity block stamped into the record and every checkpoint.

    Invariant 4: the orientation-table hash version-binds data to the instance
    that produced it — the micro instance re-derives its orientation ids within
    its piece subset, so its digest differs from the full game's by
    construction.

    Args:
        cfg: The run config.

    Returns:
        ``{"game", "game_config", "orientation_hash"}``.

    Raises:
        ValueError: If the config names a game this driver cannot construct.
    """
    if cfg.game != "blokus_duo":
        raise ValueError(f"scripts/run_micro.py drives blokus_duo only; config names {cfg.game!r}")
    return {
        "game": cfg.game,
        "game_config": cfg.game_config,
        "orientation_hash": orientation_table_hash(cfg.resolve_game_config()),
    }


def resolve_device(device: str) -> torch.device:
    """Resolve the ``--device`` argument to a torch device.

    Args:
        device: ``"auto"`` (CUDA when present, else CPU), or any explicit torch
            device string. Nothing in the loop is gated on CUDA: AMP degrades to
            an exact no-op off it.

    Returns:
        The resolved device.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass(frozen=True)
class AmpMode:
    """The run's single resolved mixed-precision setting (D5 AMP; §12 M2.5).

    One instance drives every autocast context in a run **and** whatever
    "AMP on/off" the caller records — §12 M2.5 lists AMP on/off among the
    throughput evidence, and a flag computed separately from the context that
    was actually entered can drift from it silently (FP32 inference recorded as
    AMP is a wrong number, not a crash). :meth:`observed` closes that loop by
    reading torch's *live* autocast state from inside the context, so a
    reported flag can be a measurement rather than a claim.

    Attributes:
        device_type: Torch device type the context is entered for.
        enabled: Whether autocast is live. CUDA only — the same rule as
            ``core.train.make_scaler`` / ``core.train.train_step``, so AMP is an
            exact no-op off CUDA and the CPU battery computes plain float32.
    """

    device_type: str
    enabled: bool

    @classmethod
    def resolve(cls, device: torch.device) -> AmpMode:
        """Resolve AMP for a device — the one place the on/off rule is decided.

        Args:
            device: The device the run's forwards execute on.

        Returns:
            The mode: enabled on CUDA, disabled everywhere else.
        """
        return cls(device_type=device.type, enabled=device.type == "cuda")

    def autocast(self) -> torch.autocast:
        """Open the run's autocast context.

        No explicit dtype: ``core.train.train_step`` passes none either, so
        this is torch's default for the device type (float16 on CUDA). Self-play
        inference must not run at a different precision than the trained-under
        one by accident.

        Returns:
            The context manager to wrap forwards in; an exact no-op when
            ``enabled`` is False.
        """
        return torch.autocast(device_type=self.device_type, enabled=self.enabled)

    def observed(self) -> bool:
        """Whether autocast is live *right now* for this device type.

        Called from inside :meth:`autocast` this reports what actually
        happened, which is what a report may state; called outside it reports
        False whatever ``enabled`` says.

        Returns:
            torch's live autocast state for ``device_type``.
        """
        try:
            return bool(torch.is_autocast_enabled(self.device_type))
        except TypeError:  # torch < 2.4: the query is per-device-type.
            if self.device_type == "cpu":
                return bool(torch.is_autocast_cpu_enabled())
            return bool(torch.is_autocast_enabled())


def amp_evaluator(evaluator: Evaluator, amp: AmpMode) -> Evaluator:
    """Wrap a leaf evaluator so its forward runs under the run's AMP setting.

    Applied at the call site rather than inside
    ``core.network.make_network_evaluator``: precision is a *run* policy (this
    driver's), not a property of the net→MCTS bridge, and ``core/`` has other
    callers whose behaviour must not move. Autocast is thread-local dispatch
    state, so the inner evaluator's ``torch.inference_mode`` forward picks it up
    unchanged — inference-only, no ``GradScaler`` anywhere on this path.

    Args:
        evaluator: The evaluator to wrap (``make_network_evaluator``'s result).
        amp: The run's resolved AMP mode.

    Returns:
        An ``Evaluator`` with identical semantics, evaluated under ``amp``.
    """

    def evaluate(game: Game, state: object) -> tuple[float, dict[int, float] | None]:
        """Evaluate one leaf inside the run's autocast context.

        Args:
            game: The adapter, passed through untouched.
            state: The leaf state.

        Returns:
            The wrapped evaluator's ``(value, priors)`` result.
        """
        with amp.autocast():
            return evaluator(game, state)

    return evaluate


def write_checkpoint(
    path: Path,
    net: Network,
    cfg: RunConfig,
    identity: dict[str, str],
    step: int,
) -> Path:
    """Write the minimal M2.5 checkpoint.

    Args:
        path: Destination file.
        net: The trained network (state dict is saved on CPU, so a checkpoint
            written on GPU reloads anywhere).
        cfg: The run config, persisted as a plain dict.
        identity: The :func:`game_identity` block — the orientation hash rides
            along so M3's validate-on-load has something to check.
        step: Learner steps completed when it was written.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "step": step,
            "run_seed": cfg.run_seed,
            "run_name": cfg.name,
            "config": cfg.to_dict(),
            "network_config": asdict(net.config),
            "model_state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            **identity,
        },
        path,
    )
    return path


def _learner_step(
    game: Game,
    net: Network,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    window: ReplayWindow,
    cfg: RunConfig,
    step: int,
    device: torch.device,
    record: RunRecord,
    games_played: int,
) -> None:
    """Run one learner step and append its component losses to the record.

    Per-sample D9 augmentation draws ``g`` uniformly over the adapter's declared
    ``symmetry_group`` — the identity element included, so the un-augmented
    sample is one of the four outcomes, not a special case.

    Args:
        game: The adapter (supplies the declared symmetry group and aux spec).
        net: The network, in train mode on ``device``.
        optimizer: The D5 optimizer.
        scheduler: The D5 warmup+cosine schedule; stepped once here.
        scaler: The AMP scaler (disabled off CUDA).
        window: The replay window to draw from.
        cfg: The run config.
        step: Zero-based learner-step index; keys this step's RNG streams.
        device: Device the batch is moved to.
        record: The run record to append to.
        games_played: Self-play games completed before this step.
    """
    rngs = LearnerRNGs.for_step(cfg.run_seed, step)
    num_aux = len(game.value_targets.aux_names)
    group_size = len(game.symmetry_group)
    batch = window.sample_batch(cfg.training.batch_size, rngs.window_sampling)

    rows = []
    for sample in batch:
        if group_size:
            g_index = rngs.augmentation.randrange(group_size)
            planes, sparse_pi = augment_sample(game, sample.planes, sample.sparse_pi, g_index)
            sample = replace(sample, planes=planes, sparse_pi=tuple(sparse_pi))
        # The sample owns collate's spec-driven row arity, so augmented and
        # un-augmented samples can never disagree about it.
        rows.append(sample.training_row(num_aux))

    collated = collate(game, rows).to(device)
    learning_rate = float(optimizer.param_groups[0]["lr"])
    losses = train_step(net, optimizer, scaler, collated)
    scheduler.step()
    record.record_step(
        step,
        policy_loss=float(losses.policy.item()),
        value_loss=float(losses.value.item()),
        aux_loss=None if losses.aux is None else float(losses.aux.item()),
        total_loss=float(losses.total.item()),
        learning_rate=learning_rate,
        window_size=len(window),
        games_played=games_played,
    )


def run_loop(
    cfg: RunConfig,
    *,
    run_dir: Path | str | None = None,
    device: str = "auto",
    checkpoint_every: int = 0,
    verbose: bool = False,
) -> RunRecord:
    """Run the configured self-play/train loop and write its artifacts.

    Pacing is the config's fixed alternation: ``training.games`` iterations of
    "one self-play game, then ``training.steps_per_game`` learner steps", which
    is exactly ``training.learner_steps`` steps (the config asserts the
    identity). Self-play runs the net in ``eval`` mode and training in ``train``
    mode, explicitly — the network-evaluator bridge switches to ``eval`` when it
    is built, and a batch-norm trunk left in the wrong mode is a silent
    corruption, not a crash. Leaf inference runs under the run's
    :class:`AmpMode`, the same setting the learner step trains under (an exact
    no-op off CUDA, so CPU runs are bit-for-bit what they were).

    Args:
        cfg: The validated run config (``core.runconfig.load_run_config``).
        run_dir: Output directory; defaults to ``cfg.run_dir`` under the repo
            root. Created if missing.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, … (see :func:`resolve_device`).
        checkpoint_every: Write a periodic checkpoint every N learner steps;
            ``0`` (default) writes only the final one, which is the checkpoint
            the gate evaluates (``training.checkpoint_selection == "final"``).
        verbose: Print a one-line progress report per game to stdout.

    Returns:
        The completed :class:`~core.selfplay.RunRecord` — already written to
        ``run_dir/run_record.json``.

    Raises:
        ValueError: If the config's ``aux_loss_weight`` disagrees with the
            adapter's declared weight (the two are pinned together by
            ``tests/test_micro_config_file.py``; a divergence here would train
            against a different λ than the protocol records), or if the config
            names a game this driver cannot construct.
    """
    out_dir = Path(run_dir) if run_dir is not None else ROOT / cfg.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = game_identity(cfg)
    dev = resolve_device(device)

    game = build_game(cfg)
    spec = game.value_targets
    declared = tuple(spec.aux_loss_weights)
    if declared and any(w != cfg.training.aux_loss_weight for w in declared):
        raise ValueError(
            f"config aux_loss_weight {cfg.training.aux_loss_weight} disagrees with "
            f"{type(game).__name__}'s declared weights {declared}"
        )

    # Net init is its own named stream, so changing the self-play or learner
    # streams never reshuffles the initial weights.
    torch.manual_seed(net_init_seed(cfg.run_seed))
    net = Network(NetworkConfig.from_game(game)).to(dev)
    optimizer = make_optimizer(net, lr=cfg.training.learning_rate)
    scheduler = make_lr_scheduler(
        optimizer, cfg.training.warmup_steps, cfg.training.cosine_total_steps
    )
    scaler = make_scaler(dev.type)
    # One resolved AMP setting for the whole run: the learner step reads it
    # from the batch's device inside core.train, self-play through
    # amp_evaluator below. Both halves therefore run at the same precision.
    amp = AmpMode.resolve(dev)

    window = ReplayWindow(cfg.training.replay_window)
    record = RunRecord(
        run_name=cfg.name,
        run_seed=cfg.run_seed,
        config=cfg.to_dict(),
        game_identity=identity,
        device=str(dev),
    )

    self_play_seconds = 0.0
    train_seconds = 0.0
    started = time.perf_counter()
    step = 0
    for game_index in range(cfg.training.games):
        net.eval()
        evaluator = amp_evaluator(make_network_evaluator(net, game, device=str(dev)), amp)
        rngs = GameRNGs.for_game(cfg.run_seed, game_index)
        mark = time.perf_counter()
        result = play_game(game, evaluator, cfg.self_play, rngs)
        self_play_seconds += time.perf_counter() - mark
        window.extend(result.samples)
        record.record_game(game_index, result)

        net.train()
        mark = time.perf_counter()
        for _ in range(cfg.training.steps_per_game):
            _learner_step(
                game,
                net,
                optimizer,
                scheduler,
                scaler,
                window,
                cfg,
                step,
                dev,
                record,
                games_played=game_index + 1,
            )
            step += 1
            if checkpoint_every and step % checkpoint_every == 0:
                path = write_checkpoint(
                    out_dir / f"checkpoint_step{step:06d}.pt", net, cfg, identity, step
                )
                record.record_checkpoint(step, "periodic", str(path))
        train_seconds += time.perf_counter() - mark

        if verbose:
            last = record.steps[-1] if record.steps else {}
            print(
                f"game {game_index + 1}/{cfg.training.games} "
                f"plies={result.plies} window={len(window)} step={step} "
                f"total_loss={last.get('total_loss', float('nan')):.4f}",
                flush=True,
            )

    final_path = write_checkpoint(out_dir / "checkpoint_final.pt", net, cfg, identity, step)
    record.record_checkpoint(step, "final", str(final_path))
    record.timing = {
        "self_play_seconds": self_play_seconds,
        "train_seconds": train_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    record.write(out_dir / RUN_RECORD_NAME)
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the driver's command line.

    Args:
        argv: Argument list; ``None`` reads ``sys.argv``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Run the M2.5 micro-Blokus loop end to end.")
    parser.add_argument(
        "--config",
        type=Path,
        default=MICRO_RUN_CONFIG_PATH,
        help="run config JSON (default: the pinned configs/blokus_micro.json)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="output directory (default: the config's run_dir, under the repo root)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device: auto (default), cpu, cuda, ...",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="also write a periodic checkpoint every N learner steps (0: final only)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-game progress lines")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load the config, run the loop, and report where the artifacts landed.

    Args:
        argv: Argument list; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code (0).
    """
    args = parse_args(argv)
    cfg = load_run_config(args.config)
    record = run_loop(
        cfg,
        run_dir=args.run_dir,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
        verbose=not args.quiet,
    )
    out_dir = Path(args.run_dir) if args.run_dir is not None else ROOT / cfg.run_dir
    print(f"run record: {out_dir / RUN_RECORD_NAME}")
    for entry in record.checkpoints:
        print(f"checkpoint ({entry['kind']}, step {entry['step']}): {entry['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
