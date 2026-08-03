"""Benchmark the D5 train step at batch 128/256/512 on the 4060 Ti (§7, §12 M2).

D5 pins "batch 256 (benchmark 128/256/512)" — the benchmark is part of the
pinned decision. This script produces the measurement artifact behind it:
seeded synthetic Blokus-shaped batches (binary 46×14×14 planes, legal-set
sizes drawn from real seeded random playouts, 512-trial multinomial visit
counts, D1-consistent z/aux targets) driven through the real
``core.train.train_step`` — AMP autocast + GradScaler, live on CUDA — on a
fresh full-size D5 net per batch size. One fixed batch per size is reused
every step so the timing isolates the train step, not data loading.

Timing follows the two care points named by the task: per-step times come
from CUDA events and positions/sec from a wall clock bracketed by
``torch.cuda.synchronize``, and peak VRAM is read from
``torch.cuda.max_memory_allocated`` after a post-warmup
``reset_peak_memory_stats`` (params, momentum buffers, and the resident
batch stay counted; warmup-only allocator churn does not).

**Manual GPU task:** CI is CPU-only and never runs this. Without CUDA the
script exits loudly — it never degrades to a silent CPU benchmark.

The one-line verdict in the report is mechanical so the committed artifact
is reproducible from its own table: batch 256 "fits comfortably" iff its
peak allocation stays within ``COMFORT_VRAM_FRACTION`` of device memory,
and is "throughput-reasonable" iff its positions/sec reaches
``THROUGHPUT_FRACTION`` of the best measured batch size. Either way the raw
numbers are what M3 argues from.

Usage (on the 4060 Ti box; writes the acceptance artifact):
    python3 scripts/bench_train_step.py --out docs/bench/m2-train-step.md
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.network import Network, NetworkConfig  # noqa: E402
from core.train import Batch, collate, make_optimizer, make_scaler, train_step  # noqa: E402
from games.blokus_duo.actions import IN_BOUNDS_ACTIONS  # noqa: E402
from games.blokus_duo.game import BlokusDuo  # noqa: E402
from games.blokus_duo.targets import MAX_SCORE_DIFF  # noqa: E402

# D6 full-search budget: stored positions carry root visit counts from 512-sim
# (plies 0–1, boosted for the 828-wide root) or 256-sim searches; the synthetic
# counts use the boosted budget.
SIMULATIONS = 512

# Operational reading of the task's verdict ("fits comfortably",
# "throughput-reasonable") — mechanical so the artifact's one-line verdict is
# reproducible from its own table. M3 argues from the raw numbers either way.
COMFORT_VRAM_FRACTION = 0.8
THROUGHPUT_FRACTION = 0.9


@dataclass(frozen=True)
class BenchResult:
    """Measurements for one batch size.

    Attributes:
        batch_size: Positions per train step.
        mean_ms: Mean step time in ms (synchronized wall clock / steps).
        median_ms: Median per-step time in ms (CUDA events).
        min_ms: Fastest per-step time in ms (CUDA events).
        max_ms: Slowest per-step time in ms (CUDA events).
        positions_per_s: Training throughput, ``batch_size * steps / wall``.
        peak_alloc_bytes: ``torch.cuda.max_memory_allocated`` over the timed
            steps (post-warmup reset; resident params/batch stay counted).
        peak_reserved_bytes: ``torch.cuda.max_memory_reserved`` ditto — the
            allocator's real footprint, reported alongside the pinned metric.
    """

    batch_size: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    positions_per_s: float
    peak_alloc_bytes: int
    peak_reserved_bytes: int


@dataclass(frozen=True)
class RunMeta:
    """Everything the report header states about a run.

    Attributes:
        gpu_name: ``torch.cuda.get_device_name`` of the benchmarked device.
        total_memory_bytes: Device memory, the denominator of the VRAM verdict.
        torch_version: ``torch.__version__``.
        cuda_version: ``torch.version.cuda``.
        seed: Root seed for playouts and batch synthesis.
        warmup: Untimed steps per batch size.
        steps: Timed steps per batch size.
        playouts: Seeded random playouts feeding the legal-set-size pool.
        pool: The per-ply legal-set sizes those playouts produced.
        date: Run date, ``YYYY-MM-DD``.
    """

    gpu_name: str
    total_memory_bytes: int
    torch_version: str
    cuda_version: str
    seed: int
    warmup: int
    steps: int
    playouts: int
    pool: tuple[int, ...]
    date: str


def require_cuda() -> None:
    """Exit loudly when CUDA is absent — never a silent CPU benchmark.

    Raises:
        SystemExit: Always, when ``torch.cuda.is_available()`` is false; the
            message names the manual-GPU-task contract and lands on stderr
            with a nonzero exit status.
    """
    if torch.cuda.is_available():
        return
    sys.exit(
        "ERROR: CUDA is unavailable — this is the manual D5 GPU benchmark "
        "(§7: batch 256, benchmark 128/256/512, on the RTX 4060 Ti 16 GB). "
        "A CPU timing would be meaningless for the decision, so there is no "
        "CPU fallback; run this on the GPU box."
    )


def legal_size_pool(game: BlokusDuo, seed: int, playouts: int) -> list[int]:
    """Collect realistic legal-set sizes from seeded uniform-random playouts.

    Plays through the adapter surface only (``legal_moves``/``apply``/
    ``is_terminal``), recording ``len(legal_moves)`` at every ply — the
    empirical distribution the synthetic batches draw their raggedness from,
    opening 828s and endgame near-singletons alike.

    Args:
        game: The Blokus adapter to play.
        seed: RNG seed; same seed, same pool.
        playouts: Number of complete random games.

    Returns:
        Per-ply legal-set sizes, all ``>= 1`` (pass invariant), in play order.
    """
    rng = np.random.default_rng(seed)
    sizes: list[int] = []
    for _ in range(playouts):
        state = game.initial_state()
        while not game.is_terminal(state):
            legal = game.legal_moves(state)
            sizes.append(len(legal))
            state = game.apply(state, legal[int(rng.integers(len(legal)))])
    return sizes


def synthetic_samples(
    game: BlokusDuo, batch_size: int, pool: list[int], rng: np.random.Generator
) -> list[tuple]:
    """Build seeded Blokus-shaped collate inputs (§6.1 sample arity).

    Per sample: binary float32 ``(46, 14, 14)`` planes (occupancy, inventory,
    and flag planes are all 0/1); a sparse π over ``n`` distinct in-bounds
    action ids — ``n`` drawn from ``pool`` — with a ``SIMULATIONS``-trial
    uniform multinomial as visit counts (ΣN = 512 > 0, D10); and
    D1-consistent targets from a uniform score diff:
    ``z = sign(diff)`` (incl. ``sign(0) = 0``), ``aux = diff / 109``.

    Args:
        game: The Blokus adapter; declares the plane geometry collate
            validates against.
        batch_size: Number of samples.
        pool: Legal-set sizes to draw from (``legal_size_pool``).
        rng: Seeded generator; same generator state, same samples.

    Returns:
        ``(planes, sparse_pi, z, aux)`` tuples ready for ``core.train.collate``.
    """
    h, w = game.input_shape
    samples = []
    for _ in range(batch_size):
        planes = rng.integers(0, 2, size=(game.input_planes, h, w)).astype(np.float32)
        n_legal = int(pool[int(rng.integers(len(pool)))])
        ids = rng.choice(len(IN_BOUNDS_ACTIONS), size=n_legal, replace=False)
        counts = rng.multinomial(SIMULATIONS, np.full(n_legal, 1.0 / n_legal))
        pairs = [(int(IN_BOUNDS_ACTIONS[i]), int(c)) for i, c in zip(ids, counts, strict=True)]
        diff = int(rng.integers(-MAX_SCORE_DIFF, MAX_SCORE_DIFF + 1))
        samples.append((planes, pairs, float(np.sign(diff)), (diff / MAX_SCORE_DIFF,)))
    return samples


def build_batch(game: BlokusDuo, batch_size: int, pool: list[int], seed: int) -> Batch:
    """Collate one seeded synthetic batch through the real collate boundary.

    Args:
        game: The Blokus adapter.
        batch_size: Number of samples.
        pool: Legal-set sizes to draw from.
        seed: Root seed; combined with ``batch_size`` so every measured size
            gets an independent, order-free stream.

    Returns:
        A CPU ``Batch``; the caller moves it to the device (``Batch.to``).
    """
    rng = np.random.default_rng([seed, batch_size])
    return collate(game, synthetic_samples(game, batch_size, pool, rng))


def bench_one(
    game: BlokusDuo,
    batch_size: int,
    warmup: int,
    steps: int,
    seed: int,
    pool: list[int],
    device: torch.device,
) -> BenchResult:
    """Time the D5 train step at one batch size on a fresh net/optimizer.

    Args:
        game: The Blokus adapter (dims + collate validation).
        batch_size: Positions per step.
        warmup: Untimed steps before measurement (cudnn autotune, allocator
            steady state).
        steps: Timed steps.
        seed: Root seed for the synthetic batch.
        pool: Legal-set sizes to draw from.
        device: The CUDA device to run on.

    Returns:
        The ``BenchResult`` for this batch size.
    """
    net = Network(NetworkConfig.from_game(game)).to(device).train()
    optimizer = make_optimizer(net)
    scaler = make_scaler(device.type)
    batch = build_batch(game, batch_size, pool, seed).to(device)
    for _ in range(warmup):
        train_step(net, optimizer, scaler, batch)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    wall_start = time.perf_counter()
    for start, end in zip(starts, ends, strict=True):
        start.record()
        train_step(net, optimizer, scaler, batch)
        end.record()
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - wall_start
    step_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends, strict=True))
    return BenchResult(
        batch_size=batch_size,
        mean_ms=wall * 1000.0 / steps,
        median_ms=statistics.median(step_ms),
        min_ms=step_ms[0],
        max_ms=step_ms[-1],
        positions_per_s=batch_size * steps / wall,
        peak_alloc_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )


def _gib(n_bytes: int) -> float:
    """Convert bytes to GiB.

    Args:
        n_bytes: Byte count.

    Returns:
        The count in GiB (base 2).
    """
    return n_bytes / 2**30


def d5_verdict(results: list[BenchResult], total_memory_bytes: int) -> str:
    """Compose the one-line batch-256 verdict from the measured table.

    Args:
        results: One ``BenchResult`` per measured batch size.
        total_memory_bytes: Device memory (VRAM-fraction denominator).

    Returns:
        One line: whether batch 256 fits comfortably (peak allocation within
        ``COMFORT_VRAM_FRACTION`` of device memory) and is
        throughput-reasonable (within ``THROUGHPUT_FRACTION`` of the best
        measured positions/sec) — or a no-verdict note when 256 was not among
        the measured sizes.
    """
    by_batch = {r.batch_size: r for r in results}
    r256 = by_batch.get(256)
    if r256 is None:
        return "No D5 verdict: batch 256 was not among the measured sizes."
    best = max(results, key=lambda r: r.positions_per_s)
    vram_frac = r256.peak_alloc_bytes / total_memory_bytes
    thr_frac = r256.positions_per_s / best.positions_per_s
    fits = vram_frac <= COMFORT_VRAM_FRACTION
    reasonable = thr_frac >= THROUGHPUT_FRACTION
    fits_clause = (
        f"{'fits comfortably' if fits else 'does NOT fit comfortably'} "
        f"({_gib(r256.peak_alloc_bytes):.2f} of {_gib(total_memory_bytes):.1f} GiB peak-allocated"
        f" = {vram_frac:.0%}; comfortable <= {COMFORT_VRAM_FRACTION:.0%})"
    )
    thr_clause = (
        f"{'is' if reasonable else 'is NOT'} the throughput-reasonable choice "
        f"({r256.positions_per_s:,.0f} positions/s = {thr_frac:.0%} of the best, "
        f"{best.positions_per_s:,.0f} at batch {best.batch_size}; "
        f"reasonable >= {THROUGHPUT_FRACTION:.0%})"
    )
    outcome = (
        "D5 batch 256 confirmed"
        if fits and reasonable
        else "D5 batch 256 NOT confirmed — take these numbers to M3"
    )
    return f"Batch 256 {fits_clause} and {thr_clause}: {outcome}."


def build_report(meta: RunMeta, results: list[BenchResult]) -> str:
    """Render the markdown artifact (committed as ``docs/bench/m2-train-step.md``).

    Args:
        meta: Run header facts.
        results: One ``BenchResult`` per measured batch size, in run order.

    Returns:
        The complete markdown report, verdict line included.
    """
    pool = sorted(meta.pool)
    lines = [
        "# M2 train-step benchmark — D5 batches 128/256/512",
        "",
        'D5 pins "batch 256 (benchmark 128/256/512)" (§7) on the RTX 4060 Ti 16 GB; this',
        "artifact is the measurement behind that pin. Produced by",
        "`python3 scripts/bench_train_step.py --out docs/bench/m2-train-step.md`.",
        "",
        f"- **GPU:** {meta.gpu_name}, {_gib(meta.total_memory_bytes):.1f} GiB",
        f"- **torch:** {meta.torch_version} (CUDA {meta.cuda_version})",
        f"- **Date:** {meta.date}",
        f"- **Config:** seed {meta.seed}; {meta.warmup} warmup + {meta.steps} timed steps per"
        " batch size; one fixed synthetic batch per size reused every step; legal-set sizes"
        f" from {len(meta.pool)} plies of {meta.playouts} seeded random playouts"
        f" (min/median/max {pool[0]}/{int(statistics.median(pool))}/{pool[-1]});"
        f" visit counts {SIMULATIONS}-trial multinomial.",
        "",
        "| batch | mean ms/step | median | min | max | positions/s"
        " | peak alloc (GiB) | peak reserved (GiB) |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    lines += [
        f"| {r.batch_size} | {r.mean_ms:.1f} | {r.median_ms:.1f} | {r.min_ms:.1f}"
        f" | {r.max_ms:.1f} | {r.positions_per_s:,.0f} | {_gib(r.peak_alloc_bytes):.2f}"
        f" | {_gib(r.peak_reserved_bytes):.2f} |"
        for r in results
    ]
    lines += [
        "",
        "**Methodology.** Real `core.train.train_step` (AMP autocast + GradScaler on CUDA)"
        " on a fresh full-size D5 net and optimizer per batch size; per-step times from CUDA"
        " events; positions/s from a wall clock bracketed by `torch.cuda.synchronize`; peak"
        " VRAM from `torch.cuda.max_memory_allocated` after a post-warmup"
        " `reset_peak_memory_stats` (resident params/batch stay counted).",
        "",
        f"**Verdict:** {d5_verdict(results, meta.total_memory_bytes)}",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate CLI arguments.

    Args:
        argv: Argument list, or ``None`` for ``sys.argv``.

    Returns:
        The parsed namespace.

    Raises:
        SystemExit: On invalid arguments (nonpositive sizes/steps/playouts,
            negative warmup), via ``argparse`` error handling.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[128, 256, 512],
        help="batch sizes to sweep (D5: 128 256 512)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="untimed steps per batch size")
    parser.add_argument("--steps", type=int, default=50, help="timed steps per batch size")
    parser.add_argument("--seed", type=int, default=0, help="root seed (playouts + batches)")
    parser.add_argument(
        "--playouts", type=int, default=8, help="random playouts feeding the legal-set-size pool"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="also write the report here (docs/bench/m2-train-step.md for the committed artifact)",
    )
    args = parser.parse_args(argv)
    if any(b < 1 for b in args.batch_sizes) or args.steps < 1 or args.playouts < 1:
        parser.error("batch sizes, steps, and playouts must all be >= 1")
    if args.warmup < 0:
        parser.error("warmup must be >= 0")
    return args


def main() -> None:
    """Run the sweep and emit the markdown report (stdout, and ``--out`` if given)."""
    args = parse_args()
    require_cuda()
    device = torch.device("cuda")
    game = BlokusDuo()
    pool = legal_size_pool(game, args.seed, args.playouts)
    print(
        f"[bench] legal-set pool: {len(pool)} plies from {args.playouts} playouts",
        file=sys.stderr,
    )
    results = []
    for batch_size in args.batch_sizes:
        result = bench_one(game, batch_size, args.warmup, args.steps, args.seed, pool, device)
        results.append(result)
        print(
            f"[bench] batch {batch_size}: {result.mean_ms:.1f} ms/step,"
            f" {result.positions_per_s:,.0f} positions/s,"
            f" peak alloc {_gib(result.peak_alloc_bytes):.2f} GiB",
            file=sys.stderr,
        )
        torch.cuda.empty_cache()
    meta = RunMeta(
        gpu_name=torch.cuda.get_device_name(device),
        total_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "unknown",
        seed=args.seed,
        warmup=args.warmup,
        steps=args.steps,
        playouts=args.playouts,
        pool=tuple(pool),
        date=time.strftime("%Y-%m-%d"),
    )
    report = build_report(meta, results)
    print(report, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"[bench] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
