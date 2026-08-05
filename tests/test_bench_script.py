"""Bench-script battery (§12 M2): CUDA-absence contract + synthetic batch shape.

The benchmark proper is a manual GPU task — the acceptance artifact
``docs/bench/m2-train-step.md`` is produced on the 4060 Ti, and CI (CPU-only)
never runs it. What is testable here: the loud-failure contract (no silent
CPU benchmark — nonzero exit, error on stderr, no report on stdout), the
canonical-artifact gates (``--out`` demands the exact 128/256/512 sweep and
the 4060 Ti 16 GB; exploratory reports label themselves), the seeded
synthetic batch generator feeding the real collate boundary (Blokus-shaped
planes, in-bounds sparse π with ΣN = 512, D1-consistent z/aux, seed
determinism), and the observational batch-256 summary line — measurements
only, no pass/fail verdict (D5 is pinned; re-pins are doc-first).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
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
    """A fabricated BenchResult for summary/report tests (timing fields inert)."""
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


def test_summary_is_observational():
    """The summary restates measurements — fractions, no pass/fail verdict words."""
    total = 16 * 2**30
    summary = BENCH.d5_summary(
        [result(128, 5000, 2.0), result(256, 6000, 4.0), result(512, 6200, 8.0)], total
    )
    assert "25%" in summary  # 4 of 16 GiB
    assert "97%" in summary  # 6000 of 6200 positions/s
    assert "batch 512" in summary  # the best measured size is named
    # No mechanical acceptance verdict: the pin is D5's, re-pins are doc-first.
    assert "confirmed" not in summary
    assert "doc-first" in summary
    no_256 = BENCH.d5_summary([result(128, 5000, 2.0)], total)
    assert "not among the measured sizes" in no_256


def make_meta(**overrides):
    """A plausible RunMeta for report tests, with keyword overrides."""
    fields = dict(
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
    fields.update(overrides)
    return BENCH.RunMeta(**fields)


def test_report_contains_table_and_summary():
    """The canonical artifact carries the header facts, all rows, and the summary."""
    rows = [result(128, 5000, 2.0), result(256, 6000, 4.0), result(512, 6200, 8.0)]
    report = BENCH.build_report(make_meta(), rows)
    assert "D5 batches 128/256/512" in report
    assert "RTX 4060 Ti" in report
    assert "torch:** 2.9.0 (CUDA 12.8)" in report
    for row in rows:
        assert f"\n| {row.batch_size} | " in report
    assert "**Summary:**" in report
    assert "Exploratory" not in report  # the exact D5 sweep is the canonical artifact


def test_non_canonical_report_labels_itself_exploratory():
    """A partial sweep or non-canonical hardware is headed as exploratory."""
    report = BENCH.build_report(make_meta(), [result(256, 6000, 4.0)])
    assert "D5 batches 256" in report
    assert "Exploratory sweep" in report
    assert "measurement behind that pin" not in report
    # The full sweep on the wrong GPU must not claim canonical provenance either.
    rows = [result(128, 5000, 2.0), result(256, 6000, 4.0), result(512, 6200, 8.0)]
    wrong_gpu = BENCH.build_report(
        make_meta(gpu_name="NVIDIA GeForce RTX 4090", total_memory_bytes=24 * 2**30), rows
    )
    assert "Exploratory sweep" in wrong_gpu
    assert "--out docs/bench" not in wrong_gpu  # no false produced-by line


def test_out_requires_the_canonical_sweep():
    """--out demands exactly 128/256/512 in order; stdout runs take any sweep."""
    for sweep in (["256"], ["512", "256", "128"], ["128", "256", "256", "512"]):
        with pytest.raises(SystemExit):
            BENCH.parse_args(["--batch-sizes", *sweep, "--out", "docs/bench/m2-train-step.md"])
    args = BENCH.parse_args(["--batch-sizes", "256"])  # exploratory: any sizes
    assert args.batch_sizes == [256] and args.out is None
    args = BENCH.parse_args(["--out", "docs/bench/m2-train-step.md"])  # default sweep is canonical
    assert tuple(args.batch_sizes) == BENCH.CANONICAL_BATCH_SIZES


def test_canonical_hardware_gate():
    """--out accepts only the 4060 Ti 16 GB — not other GPUs, not the 8 GB variant."""
    BENCH.require_canonical_hardware("NVIDIA GeForce RTX 4060 Ti", 16 * 2**30)
    with pytest.raises(SystemExit, match="4090"):
        BENCH.require_canonical_hardware("NVIDIA GeForce RTX 4090", 24 * 10**9)
    with pytest.raises(SystemExit, match="8.0 GiB"):
        BENCH.require_canonical_hardware("NVIDIA GeForce RTX 4060 Ti", 8 * 2**30)


def test_main_gates_out_runs_on_hardware(monkeypatch, tmp_path):
    """main() itself fires the hardware gate on --out, before any benching.

    Wiring test for the gate's one production call site: with CUDA faked
    present on a faked 4090, a --out run must exit on the hardware check —
    if a refactor drops the ``main()`` call, this fails.
    """
    monkeypatch.setattr(BENCH.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        BENCH.torch.cuda, "get_device_name", lambda device: "NVIDIA GeForce RTX 4090"
    )
    monkeypatch.setattr(
        BENCH.torch.cuda,
        "get_device_properties",
        lambda device: types.SimpleNamespace(total_memory=24 * 2**30),
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--out", str(tmp_path / "report.md")])
    with pytest.raises(SystemExit, match="4090"):
        BENCH.main()
    assert not (tmp_path / "report.md").exists()  # exited before any report
