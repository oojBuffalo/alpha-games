"""M3 GPU-acceptance-run verification (§12 M3's actual milestone evidence, issue #63).

The tiny CPU tests elsewhere in this repo prove the actor/learner/IPC
*mechanics*; the milestone's real evidence is a recorded run of the full
Blokus config on the human's GPU (``configs/blokus_duo.json``). This module
is the checker the human runs afterward (``scripts/run_selfplay.py
--verify-acceptance``): it reads a run directory's durable artifacts --
shards, checkpoints, and the reduced metrics report -- and validates every
property the milestone's acceptance criteria list, printing a PASS/FAIL
checklist rather than asserting blindly, so a failing run says exactly what
is missing.

Takes an already-constructed :class:`~core.game.Game` rather than importing
one -- the same pattern every other ``core/`` module reading shards or
checkpoints follows (``core.replay_shard.read_shard``,
``core.checkpoint.load_checkpoint``) -- so this module never imports
``games/``; the caller (``scripts/run_selfplay.py``, via
``games.registry``) builds the adapter the run's stored config names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.checkpoint import list_published_versions
from core.game import Game
from core.observability import (
    SERIES_GAMES_COMPLETED,
    SERIES_LEARNER_STEP,
    SERIES_POSITIONS_EVALUATED,
    SERIES_SIMS_RUN,
    reduce_run,
)
from core.replay_shard import GameId, SampleRecord, read_shard
from core.replay_window import samples_drawn
from core.runconfig import RunConfig

#: The milestone's floor for "at least 20 full games" (design constraints).
MIN_COMPLETED_GAMES = 20

#: Every series design doc §1/§12 M3 documents; :func:`verify_acceptance`
#: checks each is present with a positive cumulative value.
_REQUIRED_POSITIVE_SERIES = (
    SERIES_GAMES_COMPLETED,
    SERIES_POSITIONS_EVALUATED,
    SERIES_SIMS_RUN,
    SERIES_LEARNER_STEP,
)


@dataclass(frozen=True)
class CheckResult:
    """One named acceptance check's outcome.

    Attributes:
        name: Short, stable check name.
        passed: Whether it passed.
        detail: Human-readable evidence or failure reason.
    """

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class AcceptanceReport:
    """The full PASS/FAIL checklist :func:`verify_acceptance` produces.

    Attributes:
        checks: Every check, in evaluation order.
    """

    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        """Whether every check passed."""
        return all(c.passed for c in self.checks)

    def render(self) -> str:
        """Render the checklist as human-readable text.

        Returns:
            One ``[PASS]``/``[FAIL]`` line per check, plus a final verdict
            line.
        """
        lines = [f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}" for c in self.checks]
        lines.append("")
        lines.append(f"verdict: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def _scan_shards(shard_dir: Path, game: Game) -> dict[GameId, list[SampleRecord]]:
    """Read every shard under ``shard_dir`` and group its records by game id.

    Reads shard files directly (``core.replay_shard.read_shard``, the checked
    reader) rather than through ``core.replay_window.ReplayWindow`` -- this
    checker wants every record from every shard ever written, live or
    already evicted from a learner's in-memory window, since eviction never
    deletes a shard's *evidence* of games having been played, only the
    learner's willingness to keep sampling it.

    Args:
        shard_dir: The run's shard directory.
        game: The adapter to validate every shard's fingerprint against.

    Returns:
        Every observed game id mapped to its records, each list sorted by
        ``ply``.
    """
    by_game: dict[GameId, list[SampleRecord]] = {}
    if not shard_dir.exists():
        return by_game
    for path in sorted(shard_dir.glob("shard-*.npz")):
        data = read_shard(path, game)
        for record in data.records:
            by_game.setdefault(record.game_id, []).append(record)
    for records in by_game.values():
        records.sort(key=lambda r: r.ply)
    return by_game


def _consecutive_mover_game(by_game: dict[GameId, list[SampleRecord]]) -> GameId | None:
    """Return a game id whose stored samples include two consecutive same-mover plies.

    At M3's fixed-128-sims tier every ply is stored (D12's drop policy is not
    exercisable here -- design doc §12 M3), so two adjacent *stored* records
    sharing a mover is exactly a consecutive-mover (blocked-skip) sequence in
    the real game, not an artifact of dropped fast-search positions.

    Args:
        by_game: Every observed game's ply-sorted records
            (:func:`_scan_shards`).

    Returns:
        The first matching game id in iteration order, or ``None``.
    """
    for game_id, records in by_game.items():
        if any(a.mover == b.mover for a, b in zip(records, records[1:], strict=False)):
            return game_id
    return None


def verify_acceptance(run_dir: Path | str, game: Game, run_config: RunConfig) -> AcceptanceReport:
    """Validate one run directory against the M3 GPU-acceptance criteria.

    Args:
        run_dir: The run's root directory (``core.run_identity.run_root``'s
            return value for the acceptance run).
        game: The adapter the run's stored config names (built via
            ``games.registry`` by the caller).
        run_config: The run's protocol (``core.run_identity.read_stored_config
            (run_dir).run``), for ``training.batch_size``/``publish_interval``.

    Returns:
        The full checklist. Never raises for a failing run -- a check that
        cannot even be evaluated (e.g. no shards at all) reports failed with
        an explanatory detail, exactly like any other failing check.
    """
    run_dir = Path(run_dir)
    shard_dir = run_dir / "shards"
    ckpt_dir = run_dir / "checkpoints"
    checks: list[CheckResult] = []

    by_game = _scan_shards(shard_dir, game)
    num_games = len(by_game)
    checks.append(
        CheckResult(
            "shards readable",
            True,
            f"{sum(len(r) for r in by_game.values())} sample(s) across {num_games} game(s) decoded",
        )
    )
    checks.append(
        CheckResult(
            f"at least {MIN_COMPLETED_GAMES} completed games",
            num_games >= MIN_COMPLETED_GAMES,
            f"{num_games} distinct game id(s) observed across all shards",
        )
    )

    consecutive_game = _consecutive_mover_game(by_game)
    checks.append(
        CheckResult(
            "at least one consecutive-mover (blocked-skip) sequence",
            consecutive_game is not None,
            f"game {consecutive_game}"
            if consecutive_game is not None
            else "no game had two consecutive stored samples sharing a mover",
        )
    )

    versions_played = {r.model_version for records in by_game.values() for r in records}
    checks.append(
        CheckResult(
            "actor reload observed (some game played at model_version >= 1)",
            any(v >= 1 for v in versions_played),
            f"model_version(s) played: {sorted(versions_played)}",
        )
    )

    reduced = reduce_run(run_dir)
    learner_step = int(reduced.totals.get(SERIES_LEARNER_STEP, 0.0))
    drawn = samples_drawn(learner_step, run_config.training.batch_size)
    checks.append(
        CheckResult(
            "shards ingested and sampled",
            num_games > 0 and drawn > 0,
            f"{num_games} game(s) ingested from shards; samples_drawn={drawn} "
            f"(learner_step={learner_step}, batch_size={run_config.training.batch_size})",
        )
    )

    published = list_published_versions(ckpt_dir)
    checks.append(
        CheckResult(
            "at least one published checkpoint beyond the seeded v0",
            any(v >= 1 for v in published),
            f"published versions: {list(published)}",
        )
    )
    checks.append(
        CheckResult(
            "at least one full publish interval of learner steps",
            learner_step >= run_config.training.publish_interval,
            f"learner_step={learner_step}, publish_interval={run_config.training.publish_interval}",
        )
    )

    missing_series = [
        name for name in _REQUIRED_POSITIVE_SERIES if reduced.totals.get(name, 0.0) <= 0
    ]
    checks.append(
        CheckResult(
            "every documented series present with a positive total",
            not missing_series,
            "all present and positive"
            if not missing_series
            else f"missing or zero: {missing_series}",
        )
    )
    checks.append(
        CheckResult(
            "checkpoints x-axis coordinates recorded",
            bool(reduced.checkpoints),
            f"{len(reduced.checkpoints)} checkpoint marker(s) in the reduced report",
        )
    )

    return AcceptanceReport(checks=tuple(checks))
