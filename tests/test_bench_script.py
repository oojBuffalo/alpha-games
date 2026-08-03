"""Bench-script battery (§12 M2): CUDA-absence contract + synthetic batch shape.

The benchmark proper is a manual GPU task — the acceptance artifact
``docs/bench/m2-train-step.md`` is produced on the 4060 Ti, and CI (CPU-only)
never runs it. What is testable here: the loud-failure contract (no silent
CPU benchmark — nonzero exit, error on stderr, no report on stdout), the
seeded synthetic batch generator feeding the real collate boundary
(Blokus-shaped planes, in-bounds sparse π with ΣN = 512, D1-consistent
z/aux, seed determinism), and the mechanical batch-256 verdict rule the
committed artifact's one-line verdict follows.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from games.blokus_duo import BlokusDuo
from games.blokus_duo.actions import IN_BOUNDS_ACTIONS
from games.blokus_duo.targets import MAX_SCORE_DIFF

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_train_step.py"

BLOKUS = BlokusDuo()


def load_bench():
    """Import ``scripts/bench_train_step.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("bench_train_step", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass resolution needs the sys.modules entry
    spec.loader.exec_module(module)
    return module


BENCH = load_bench()


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="CUDA present: the loud-failure path cannot fire"
)
def test_exits_loudly_without_cuda():
    """No CUDA → nonzero exit, the error names CUDA on stderr, and no report."""
    proc = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode != 0
    assert "CUDA" in proc.stderr
    assert "GPU" in proc.stderr
    assert proc.stdout == ""  # not a silent CPU benchmark — nothing report-like emitted


def test_legal_size_pool_is_realistic_and_seeded():
    """Playout pool: per-ply sizes ≥ 1 (pass invariant), opening 828 first, seeded."""
    pool = BENCH.legal_size_pool(BLOKUS, seed=3, playouts=1)
    assert pool, "a playout must record at least the opening ply"
    assert all(size >= 1 for size in pool)
    assert pool[0] == 828  # the golden opening width
    assert pool == BENCH.legal_size_pool(BLOKUS, seed=3, playouts=1)


def test_synthetic_batch_is_blokus_shaped_and_seeded():
    """Collated batch: declared plane geometry, in-bounds ΣN=512 π, D1 targets, seeded."""
    pool = [1, 63, 828]
    batch = BENCH.build_batch(BLOKUS, 4, pool, seed=7)
    assert tuple(batch.planes.shape) == (4, 46, 14, 14)
    assert set(batch.planes.unique().tolist()) <= {0.0, 1.0}
    in_bounds = set(IN_BOUNDS_ACTIONS)
    for ids, counts in zip(batch.legal_ids, batch.visit_counts, strict=True):
        real = ids[ids >= 0]
        assert len(real) in set(pool)
        assert len(set(real.tolist())) == len(real)  # distinct action ids
        assert set(real.tolist()) <= in_bounds
        assert counts.sum().item() == BENCH.SIMULATIONS  # ΣN = 512 (D10)
        assert (counts[ids < 0] == 0).all()  # zero in pad slots
    # D1 consistency: z = sign(diff), aux = diff/109 → z recoverable from aux.
    assert batch.aux is not None and tuple(batch.aux.shape) == (4, 1)
    assert torch.equal(batch.z, torch.sign(batch.aux.squeeze(1) * MAX_SCORE_DIFF))
    # Same seed, same batch — the benchmark input is reproducible.
    again = BENCH.build_batch(BLOKUS, 4, pool, seed=7)
    assert torch.equal(batch.planes, again.planes)
    assert torch.equal(batch.legal_ids, again.legal_ids)
    assert torch.equal(batch.visit_counts, again.visit_counts)
    assert torch.equal(batch.z, again.z)
    assert torch.equal(batch.aux, again.aux)


def result(batch_size, positions_per_s, peak_alloc_gib):
    """A fabricated BenchResult for verdict-rule tests (timing fields inert)."""
    return BENCH.BenchResult(
        batch_size=batch_size,
        mean_ms=1.0,
        median_ms=1.0,
        min_ms=1.0,
        max_ms=1.0,
        positions_per_s=positions_per_s,
        peak_alloc_bytes=int(peak_alloc_gib * 2**30),
        peak_reserved_bytes=int(peak_alloc_gib * 2**30),
    )


def test_verdict_rule():
    """The mechanical verdict: confirmed iff 256 fits (≤80%) and keeps ≥90% throughput."""
    total = 16 * 2**30
    confirmed = BENCH.d5_verdict(
        [result(128, 5000, 2.0), result(256, 6000, 4.0), result(512, 6200, 8.0)], total
    )
    assert "confirmed" in confirmed and "NOT" not in confirmed
    too_slow = BENCH.d5_verdict(
        [result(128, 5000, 2.0), result(256, 5000, 4.0), result(512, 9000, 8.0)], total
    )
    assert "NOT confirmed" in too_slow
    too_big = BENCH.d5_verdict([result(256, 6000, 14.0), result(128, 5000, 2.0)], total)
    assert "NOT confirmed" in too_big
    assert "No D5 verdict" in BENCH.d5_verdict([result(128, 5000, 2.0)], total)


def test_report_contains_table_and_verdict():
    """The rendered artifact carries the header facts, all rows, and the verdict line."""
    meta = BENCH.RunMeta(
        gpu_name="NVIDIA GeForce RTX 4060 Ti",
        total_memory_bytes=16 * 2**30,
        torch_version="2.9.0",
        cuda_version="12.8",
        seed=0,
        warmup=10,
        steps=50,
        playouts=8,
        pool=(828, 828, 400, 63, 2),
        date="2026-01-01",
    )
    rows = [result(128, 5000, 2.0), result(256, 6000, 4.0), result(512, 6200, 8.0)]
    report = BENCH.build_report(meta, rows)
    assert "RTX 4060 Ti" in report
    assert "torch:** 2.9.0 (CUDA 12.8)" in report
    for row in rows:
        assert f"\n| {row.batch_size} | " in report
    assert "**Verdict:**" in report
    assert "confirmed" in report
