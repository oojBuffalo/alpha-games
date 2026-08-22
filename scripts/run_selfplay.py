"""The M3 self-play run entrypoint: launch, resume, fork, verify (§12 M3, issue #63).

One config-driven CLI that launches -- or resumes, or forks -- the fixed-128-
sim self-play baseline (actors + learner + metrics, ``core.ipc.launch_run``)
from a single recorded ``core.run_identity.LaunchConfig``, and separately
checks a completed run's durable artifacts against the milestone's GPU-
acceptance criteria (``core.acceptance``).

Usage::

    # Fresh launch, blocking until every process exits on its own
    # (production: never, until Ctrl-C; a bounded test run: at its own
    # --max-games-per-actor / --max-learner-steps caps).
    python3 scripts/run_selfplay.py configs/blokus_duo.json

    # Resume an existing run directory. Refuses (no override flag) if
    # configs/blokus_duo.json now differs from the run's stored config in
    # any material field; a non-material difference (paths, poll cadences)
    # proceeds, logged.
    python3 scripts/run_selfplay.py configs/blokus_duo.json --resume runs/blokus_duo/<run-id>

    # Fork: a brand-new run identity with recorded lineage, never mutating
    # the parent. The passed config may freely differ from the parent's,
    # including in material fields -- that is a fork's whole purpose.
    python3 scripts/run_selfplay.py configs/blokus_duo.json --fork-from runs/blokus_duo/<run-id>

    # After a GPU acceptance run completes:
    python3 scripts/run_selfplay.py --verify-acceptance runs/blokus_duo/<run-id>

``core/`` and ``games/`` are both imported here (the script layer, never
``core/`` itself -- ``games.registry`` resolves the config's game name to a
picklable ``core.ipc.GameFactory``); ``core/run_identity.py``'s own docstring
documents the material/non-material config classification this CLI's resume
refusal is built on.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.acceptance import verify_acceptance  # noqa: E402
from core.ipc import LaunchedRun, launch_run  # noqa: E402
from core.network import NetworkConfig  # noqa: E402
from core.run_identity import (  # noqa: E402
    ENTRY_CONDITION,
    LaunchConfig,
    MaterialConfigDiffError,
    RunRecord,
    generate_run_id,
    iso_now,
    load_launch_config,
    resolve_fork,
    resolve_resume,
    run_root,
    write_provenance,
)
from games.registry import build_game_factory  # noqa: E402

DEFAULT_WAIT_POLL_INTERVAL = 1.0


def _launch(
    launch_config: LaunchConfig,
    run_id: str,
    root: Path,
    *,
    max_games_per_actor: int | None,
    max_learner_steps: int | None,
    network_config: NetworkConfig | None,
) -> LaunchedRun:
    """Start every process for one run identity (the shared tail of launch/resume/fork).

    Args:
        launch_config: The config to run with (already validated).
        run_id: The run identity every process is launched under.
        root: The concrete run directory (``core.run_identity.run_root``).
        max_games_per_actor: Test-facing cap forwarded to every actor;
            ``None`` in production.
        max_learner_steps: Test-facing cap forwarded to the learner;
            ``None`` in production.
        network_config: Network architecture override. ``None`` (the
            production default) builds the pinned D5 8x128 trunk
            (``core.network.NetworkConfig.from_game``); tests pass a tiny
            override, mirroring every other driver/IPC test in this repo.
            Not exposed as a CLI flag -- the architecture is design-doc
            pinned, not a runtime choice.

    Returns:
        The live process handles.
    """
    root.mkdir(parents=True, exist_ok=True)
    game_factory = build_game_factory(launch_config.run)
    return launch_run(
        game_factory=game_factory,
        run_config=launch_config.run,
        self_play=launch_config.run.self_play,
        run_id=run_id,
        num_actors=launch_config.num_actors,
        shard_dir=root / "shards",
        ckpt_dir=root / "checkpoints",
        run_dir=root,
        run_seed=launch_config.run.run_seed,
        device=launch_config.device,
        network_config=network_config,
        refresh_poll_interval=launch_config.runtime.refresh_poll_interval,
        pacing_poll_interval=launch_config.runtime.pacing_poll_interval,
        ceiling_poll_interval=launch_config.runtime.ceiling_poll_interval,
        max_games_per_actor=max_games_per_actor,
        max_learner_steps=max_learner_steps,
    )


def wait_for_completion(
    launched: LaunchedRun, poll_interval: float = DEFAULT_WAIT_POLL_INTERVAL
) -> None:
    """Block until every process exits on its own, or until interrupted, then shut down.

    A production run (no test-facing step/game caps) never exits on its own,
    so this blocks until Ctrl-C (``SIGINT``); a bounded test run's processes
    exit on their own once they hit their caps, and this returns promptly.
    Either way, :meth:`~core.ipc.LaunchedRun.shutdown` always runs -- signaling
    every process (a no-op for one already dead) and closing the GPU-hour
    segment.

    Args:
        launched: The live run to wait on.
        poll_interval: Seconds between liveness polls.
    """
    try:
        while any(p.is_alive() for p in launched.all_processes()):
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        pass
    finally:
        launched.shutdown()


def cmd_launch(
    config_path: Path | str,
    *,
    max_games_per_actor: int | None = None,
    max_learner_steps: int | None = None,
    block: bool = True,
    now: float | None = None,
    network_config: NetworkConfig | None = None,
) -> LaunchedRun:
    """Launch a brand-new run from a config file.

    Generates a fresh run id, records the config verbatim plus the entry
    condition into the new run directory (:func:`~core.run_identity.write_provenance`),
    and starts every process.

    Args:
        config_path: Path to a ``core.run_identity.LaunchConfig`` JSON file.
        max_games_per_actor: Test-facing cap; ``None`` in production.
        max_learner_steps: Test-facing cap; ``None`` in production.
        block: Whether to wait for completion before returning
            (:func:`wait_for_completion`). ``False`` is test-facing: the
            caller owns shutdown.
        now: Launch timestamp override, for deterministic run ids in tests.
        network_config: See :func:`_launch`. Not exposed as a CLI flag.

    Returns:
        The live run.
    """
    launch_config = load_launch_config(config_path)
    run_id = generate_run_id(launch_config, now=now)
    root = run_root(launch_config, run_id)
    write_provenance(
        root,
        launch_config,
        RunRecord(run_id=run_id, created_at=iso_now(now), entry_condition=ENTRY_CONDITION),
    )
    launched = _launch(
        launch_config,
        run_id,
        root,
        max_games_per_actor=max_games_per_actor,
        max_learner_steps=max_learner_steps,
        network_config=network_config,
    )
    if block:
        wait_for_completion(launched)
    return launched


def cmd_resume(
    config_path: Path | str,
    run_dir: Path | str,
    *,
    max_games_per_actor: int | None = None,
    max_learner_steps: int | None = None,
    block: bool = True,
    network_config: NetworkConfig | None = None,
) -> LaunchedRun:
    """Resume an existing run directory.

    Refuses (:class:`~core.run_identity.MaterialConfigDiffError`, no override
    flag) if ``config_path`` differs from the run's stored ``config.json`` in
    any material field. A non-material difference proceeds, logged to
    stdout.

    Args:
        config_path: Path to the config to resume with.
        run_dir: The existing run directory (as returned by a prior launch's
            :func:`~core.run_identity.run_root`).
        max_games_per_actor: Test-facing cap; ``None`` in production.
        max_learner_steps: Test-facing cap; ``None`` in production.
        block: See :func:`cmd_launch`.
        network_config: See :func:`_launch`. Not exposed as a CLI flag.

    Returns:
        The live run.

    Raises:
        core.run_identity.MaterialConfigDiffError: On any material config
            difference.
    """
    new_launch_config = load_launch_config(config_path)
    resolution = resolve_resume(run_dir, new_launch_config)
    for path, (old, new) in sorted(resolution.non_material_diff.items()):
        print(f"resume: non-material field {path!r} changed {old!r} -> {new!r}; proceeding")
    launched = _launch(
        resolution.effective_config,
        resolution.run_id,
        resolution.run_root,
        max_games_per_actor=max_games_per_actor,
        max_learner_steps=max_learner_steps,
        network_config=network_config,
    )
    if block:
        wait_for_completion(launched)
    return launched


def cmd_fork(
    config_path: Path | str,
    parent_run_dir: Path | str,
    *,
    max_games_per_actor: int | None = None,
    max_learner_steps: int | None = None,
    block: bool = True,
    now: float | None = None,
    network_config: NetworkConfig | None = None,
) -> LaunchedRun:
    """Fork a brand-new run identity from an existing run directory.

    The parent run directory is never written to. The new run's
    ``run_record.json`` carries the parent's identity and config hash as
    lineage (:class:`~core.run_identity.Lineage`) -- weights are never
    imported in this milestone (documented seam, ``core.run_identity.Lineage``'s
    docstring).

    Args:
        config_path: Path to the fork's own config (may differ from the
            parent's in any field, material or not).
        parent_run_dir: The existing run directory to fork from.
        max_games_per_actor: Test-facing cap; ``None`` in production.
        max_learner_steps: Test-facing cap; ``None`` in production.
        block: See :func:`cmd_launch`.
        now: Fork timestamp override, for deterministic run ids in tests.
        network_config: See :func:`_launch`. Not exposed as a CLI flag.

    Returns:
        The live run.
    """
    new_launch_config = load_launch_config(config_path)
    resolution = resolve_fork(parent_run_dir, new_launch_config, now=now)
    write_provenance(
        resolution.run_root,
        new_launch_config,
        RunRecord(
            run_id=resolution.run_id,
            created_at=iso_now(now),
            entry_condition=ENTRY_CONDITION,
            lineage=resolution.lineage,
        ),
    )
    launched = _launch(
        new_launch_config,
        resolution.run_id,
        resolution.run_root,
        max_games_per_actor=max_games_per_actor,
        max_learner_steps=max_learner_steps,
        network_config=network_config,
    )
    if block:
        wait_for_completion(launched)
    return launched


def cmd_verify_acceptance(run_dir: Path | str) -> int:
    """Check a run directory against the M3 GPU-acceptance criteria and print the result.

    Args:
        run_dir: The run directory to verify.

    Returns:
        ``0`` if every check passed, ``1`` otherwise.
    """
    root = Path(run_dir)
    launch_config = load_launch_config(root / "config.json")
    game = build_game_factory(launch_config.run)()
    report = verify_acceptance(root, game, launch_config.run)
    print(report.render())
    return 0 if report.passed else 1


def build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="run_selfplay.py",
        description="Launch, resume, fork, or verify the M3 self-play baseline.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="path to a launch-config JSON file (required unless --verify-acceptance)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", metavar="RUN_DIR", default=None, help="resume an existing run")
    mode.add_argument(
        "--fork-from", metavar="RUN_DIR", default=None, help="fork a new run from an existing one"
    )
    mode.add_argument(
        "--verify-acceptance",
        metavar="RUN_DIR",
        default=None,
        help="check a run directory against the M3 GPU-acceptance criteria and exit",
    )
    parser.add_argument(
        "--max-games-per-actor",
        type=int,
        default=None,
        help="test-facing cap on games played per actor; unset in production",
    )
    parser.add_argument(
        "--max-learner-steps",
        type=int,
        default=None,
        help="test-facing cap on learner steps trained; unset in production",
    )
    parser.add_argument(
        "--no-block",
        action="store_true",
        help="return immediately after starting processes, without waiting (test-facing)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """The CLI entrypoint.

    Args:
        argv: Argument vector, excluding the program name. Defaults to
            ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.verify_acceptance is not None:
        return cmd_verify_acceptance(args.verify_acceptance)

    if args.config is None:
        parser.error("config is required unless --verify-acceptance is given")

    block = not args.no_block
    try:
        if args.resume is not None:
            cmd_resume(
                args.config,
                args.resume,
                max_games_per_actor=args.max_games_per_actor,
                max_learner_steps=args.max_learner_steps,
                block=block,
            )
        elif args.fork_from is not None:
            cmd_fork(
                args.config,
                args.fork_from,
                max_games_per_actor=args.max_games_per_actor,
                max_learner_steps=args.max_learner_steps,
                block=block,
            )
        else:
            cmd_launch(
                args.config,
                max_games_per_actor=args.max_games_per_actor,
                max_learner_steps=args.max_learner_steps,
                block=block,
            )
    except MaterialConfigDiffError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
