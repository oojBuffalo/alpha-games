"""The M4 eval harness's CLI entrypoint: launch, report, plateau, bench (§9; tasks/m4/009.3).

A thin CLI over ``core.eval_run``/``core.eval_stats`` -- the ``scripts/run_selfplay.py``
pattern, applied to the eval namespace instead of the training run: this script owns
argument parsing, resolving the watched run's game/eval profile through
``games.registry`` (the one place this module imports ``games.*`` -- ``core/`` never
does), and human-readable rendering; every actual decision (scheduling, playing,
fitting, gating, benching) lives in ``core/``.

Usage::

    # Default mode: launch (or resume) the concurrent watch/report loop. Blocks
    # until every member 1..K is complete (production: effectively forever, since
    # K grows no further once the watched run finishes; a bounded test run passes
    # --max-idle-polls or --single-pass instead). After every checkpoint the loop
    # itself completes, regenerates elo_curve.json/verdict.json from a fresh
    # snapshot and prints the refreshed eval lag -- the loop hook, zero manual
    # steps.
    python3 scripts/run_eval.py configs/m4_eval.json

    # Load one task-5 snapshot, regenerate elo_curve.json/verdict.json from it,
    # and print the eval lag alongside -- no gate and delta: null before the
    # K-set completes (task 7); a partial cell is never read.
    python3 scripts/run_eval.py configs/m4_eval.json --report

    # A read-only design doc §12 M4 profiled-plateau reading -- triggers nothing,
    # writes nothing; the tri-state outcome is rendered as one of
    # "plateau"/"no_plateau"/"insufficient_data", never coerced to a bool.
    python3 scripts/run_eval.py configs/m4_eval.json --plateau

    # Play a small, fixed pair count against the newest published member at the
    # configured production S and device, and report seconds/game, games/hour,
    # and the projected hours per (checkpoint x full required cell set) against
    # the watched run's own publish cadence -- the task 1 pin 11 feasibility
    # number, measured rather than asserted. Writes nothing to the eval store.
    python3 scripts/run_eval.py configs/m4_eval.json --bench

``core/`` never imports ``games.*`` (design doc §Repo layout); this script is the
one call site that resolves ``EvalConfig.run_dir``'s stored game name
(``core.run_identity.read_stored_config``) to a live ``Game`` instance
(``games.registry.build_game_factory``) and the game's declared
``core.eval_profile.EvalProfile`` (``games.registry.build_eval_profile``), then
hands both to ``core.eval_run``'s game-generic functions -- mirroring exactly how
``scripts/run_selfplay.py`` resolves a training run's own game factory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.eval_profile import EvalProfile  # noqa: E402
from core.eval_protocol import PLATEAU_CONFIRMATION_COUNT  # noqa: E402
from core.eval_run import (  # noqa: E402
    BenchResult,
    EvalConfig,
    WatchLoopResult,
    bench_candidate,
    eval_lag,
    load_eval_config,
    run_watch_loop,
    watched_k_total,
)
from core.eval_stats import PlateauResult, build_verdict, detect_plateau  # noqa: E402
from core.game import Game  # noqa: E402
from core.run_identity import read_stored_config  # noqa: E402
from games.registry import build_eval_profile, build_game_factory  # noqa: E402

#: Production watch-loop poll cadence -- generous, since a completed member's
#: worth of games (24 pairs x up to 5 opponents x 3 forms, at S = 512 sims for
#: two of those forms) dwarfs it; tests pass a far smaller value.
DEFAULT_POLL_INTERVAL = 5.0

#: Default mirrored pairs sampled per required cell during ``--bench`` -- small
#: on purpose (the *production* ``pairs_per_cell`` is used only for the
#: projection, never actually played by the bench itself).
DEFAULT_BENCH_PAIRS = 1


def _resolve_watched_game(run_dir: Path | str) -> tuple[Game, EvalProfile]:
    """Resolve the watched run's live game adapter and declared eval profile.

    Args:
        run_dir: The watched run's root directory (must already carry
            ``core.run_identity``'s recorded ``config.json``).

    Returns:
        ``(game, profile)`` -- exactly the pair ``core.eval_run.run_watch_loop``
        and ``core.eval_run.bench_candidate`` need. This function, and the
        module-level imports above, are the only place this script reaches
        into ``games.*``.

    Raises:
        FileNotFoundError: If ``run_dir`` has no recorded ``config.json`` yet.
        ValueError: If the stored game name has no registry entry, or (for
            ``build_eval_profile``) no declared M4 eval profile.
    """
    stored = read_stored_config(run_dir)
    game = build_game_factory(stored.run)()
    profile = build_eval_profile(stored.run.game)
    return game, profile


def _build_report(config: EvalConfig) -> dict:
    """Regenerate the §1 report artifacts from one fresh snapshot, plus the eval lag.

    Shared by :func:`cmd_report` and :func:`cmd_launch`'s own loop hook, so
    there is exactly one report-regeneration call site (``core.eval_stats.
    build_verdict`` already refreshes ``elo_curve.json`` from the identical
    snapshot object it fits -- see that function's docstring).

    Args:
        config: The harness's resolved :class:`~core.eval_run.EvalConfig`.

    Returns:
        ``{"verdict": <verdict.json payload>, "eval_lag": int}``.
    """
    verdict = build_verdict(config.run_dir, B=config.bootstrap_b)
    lag = eval_lag(config.run_dir, watched_k_total(config.run_dir))
    return {"verdict": verdict, "eval_lag": lag}


def cmd_report(config_path: Path | str) -> dict:
    """``--report``: load one task-5 snapshot and regenerate the §1 report artifacts.

    Args:
        config_path: Path to an :class:`~core.eval_run.EvalConfig` JSON file.

    Returns:
        :func:`_build_report`'s payload.
    """
    return _build_report(load_eval_config(config_path))


def cmd_plateau(config_path: Path | str) -> PlateauResult:
    """``--plateau``: a read-only ``detect_plateau`` reading -- triggers nothing.

    Args:
        config_path: Path to an :class:`~core.eval_run.EvalConfig` JSON file.

    Returns:
        The tri-state :class:`~core.eval_stats.PlateauResult`.
    """
    config = load_eval_config(config_path)
    return detect_plateau(config.run_dir, B=config.bootstrap_b)


def cmd_bench(config_path: Path | str, *, n_pairs: int = DEFAULT_BENCH_PAIRS) -> BenchResult:
    """``--bench``: measure real throughput against the newest published member.

    Args:
        config_path: Path to an :class:`~core.eval_run.EvalConfig` JSON file.
        n_pairs: Mirrored pairs sampled per required cell (small; see
            ``core.eval_run.bench_candidate``).

    Returns:
        The measured :class:`~core.eval_run.BenchResult`.
    """
    config = load_eval_config(config_path)
    game, profile = _resolve_watched_game(config.run_dir)
    return bench_candidate(config.run_dir, config, profile, game, n_pairs=n_pairs)


def cmd_launch(
    config_path: Path | str,
    *,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    max_idle_polls: int | None = None,
    single_pass: bool = False,
) -> WatchLoopResult:
    """Default mode: launch (or resume) the concurrent watch/report loop.

    Wires ``core.eval_run.run_watch_loop``'s ``on_member_complete`` hook to
    :func:`_build_report`, so every checkpoint the loop itself finishes
    scoring is immediately followed by a fresh §1 report regeneration and a
    printed eval-lag line -- the report is never more than one poll behind
    the store, with zero manual steps (design doc §9's coupling requirement,
    subtask 9.3).

    Args:
        config_path: Path to an :class:`~core.eval_run.EvalConfig` JSON file.
        poll_interval: Seconds between polls (production default
            :data:`DEFAULT_POLL_INTERVAL`; tests pass a far smaller value).
        max_idle_polls: Test-facing stop condition; ``None`` (production)
            runs until every member ``1..K`` is complete.
        single_pass: Test-facing: run exactly one poll cycle and return,
            regardless of completeness.

    Returns:
        The loop's summary (``core.eval_run.WatchLoopResult``).
    """
    config = load_eval_config(config_path)
    game, profile = _resolve_watched_game(config.run_dir)

    def _on_member_complete(member_version: int) -> None:
        report = _build_report(config)
        print(
            f"[run_eval] checkpoint {member_version} complete -- eval_lag="
            f"{report['eval_lag']} authoritative={report['verdict']['authoritative']}"
        )

    return run_watch_loop(
        config.run_dir,
        config,
        profile,
        game,
        poll_interval=poll_interval,
        max_idle_polls=max_idle_polls,
        single_pass=single_pass,
        on_member_complete=_on_member_complete,
    )


def _render_report(payload: dict) -> str:
    """Render :func:`cmd_report`'s payload as human-readable lines."""
    verdict = payload["verdict"]
    lines = [
        f"eval_lag: {payload['eval_lag']}",
        f"checkpoints_evaluated: {verdict['checkpoints_evaluated']}/{verdict['k_target']}",
        f"authoritative: {verdict['authoritative']}",
    ]
    delta = verdict["delta"]
    if delta is None:
        lines.append(f"delta: null ({verdict['reason']})")
    else:
        lines.append(
            f"delta_hat: {delta['delta_hat']:.2f} "
            f"ci=[{delta['ci'][0]:.2f}, {delta['ci'][1]:.2f}] gate={delta['gate']}"
        )
    mk = verdict["mann_kendall"]
    lines.append(f"mann_kendall: insufficient_data={mk['insufficient_data']} p={mk['p']}")
    return "\n".join(lines)


def _render_plateau(result: PlateauResult) -> str:
    """Render a :class:`~core.eval_stats.PlateauResult` as human-readable lines.

    ``outcome`` is always printed verbatim as one of ``"plateau"``,
    ``"no_plateau"``, or ``"insufficient_data"`` -- never coerced to a
    ``True``/``False``/count-like rendering (the tri-state contract).
    """
    lines = [
        f"outcome: {result.outcome}",
        f"window_m: {result.window_m}",
        f"confirmation_count: {result.confirmation_count}/{PLATEAU_CONFIRMATION_COUNT}",
    ]
    for label, window in (("current", result.current), ("previous", result.previous)):
        if window is not None:
            lines.append(
                f"{label} window (ending at member {window.newest_version}, "
                f"versions {list(window.versions)}): "
                f"mk_non_significant={window.mk_non_significant} "
                f"ci_narrow={window.ci_narrow} "
                f"gpu_span_sufficient={window.gpu_span_sufficient} "
                f"satisfied={window.satisfied}"
            )
    if result.reason is not None:
        lines.append(f"reason: {result.reason}")
    return "\n".join(lines)


def _render_bench(result: BenchResult) -> str:
    """Render a :class:`~core.eval_run.BenchResult` as human-readable lines."""
    return "\n".join(
        [
            f"candidate_version: {result.candidate_version}",
            f"device: {result.device}  eval_sims: {result.eval_sims}",
            f"cells_benched: {result.cells_benched}  "
            f"games_played: {result.games_played} "
            f"({result.bench_pairs_per_cell} pair(s)/cell sampled)",
            f"seconds_per_game: {result.seconds_per_game:.3f}",
            f"games_per_hour: {result.games_per_hour:.1f}",
            f"projected per checkpoint (production {result.pairs_per_cell} pairs/cell): "
            f"{result.games_per_checkpoint} games, "
            f"{result.projected_hours_per_checkpoint:.2f}h",
            f"publish_cadence_hours: {result.publish_cadence_hours}",
            f"feasible: {result.feasible}",
        ]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description=(
            "Launch (or resume) the M4 concurrent eval harness, or read its "
            "report/plateau/bench modes."
        ),
    )
    parser.add_argument(
        "config", help="path to an EvalConfig JSON file (e.g. configs/m4_eval.json)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--report",
        action="store_true",
        help=(
            "regenerate elo_curve.json/verdict.json from the latest snapshot, print "
            "the eval lag alongside, and exit"
        ),
    )
    mode.add_argument(
        "--plateau",
        action="store_true",
        help="print a read-only detect_plateau reading and exit (triggers nothing)",
    )
    mode.add_argument(
        "--bench",
        action="store_true",
        help=(
            "play a small fixed pair count at the configured production S and report "
            "throughput/feasibility, then exit"
        ),
    )
    parser.add_argument(
        "--bench-pairs",
        type=int,
        default=DEFAULT_BENCH_PAIRS,
        help=(
            "mirrored pairs sampled per required cell during --bench "
            f"(default: {DEFAULT_BENCH_PAIRS})"
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="seconds between watch-loop polls (default launch mode only)",
    )
    parser.add_argument(
        "--max-idle-polls",
        type=int,
        default=None,
        help=(
            "test-facing: stop the watch loop after this many idle polls in a row; "
            "unset in production"
        ),
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="test-facing: run exactly one watch-loop poll cycle and return",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """The CLI entrypoint.

    Args:
        argv: Argument vector, excluding the program name. Defaults to
            ``sys.argv[1:]``.

    Returns:
        The process exit code (``0`` on success).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.report:
        print(_render_report(cmd_report(args.config)))
    elif args.plateau:
        print(_render_plateau(cmd_plateau(args.config)))
    elif args.bench:
        print(_render_bench(cmd_bench(args.config, n_pairs=args.bench_pairs)))
    else:
        result = cmd_launch(
            args.config,
            poll_interval=args.poll_interval,
            max_idle_polls=args.max_idle_polls,
            single_pass=args.single_pass,
        )
        print(
            f"[run_eval] stopped: {result.stopped_reason} (polls={result.polls}, "
            f"pairs_played={result.pairs_played}, "
            f"completed_members={list(result.completed_members)})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
