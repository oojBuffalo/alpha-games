"""Throughput go/no-go battery (§12 M2.5, task 8): arithmetic + the gates.

The spike itself is a manual run — the signed verdict is produced once on the
RTX 4060 Ti 16 GB, and CI (CPU-only) never runs it. **Nothing here runs the
gate's 50/200 spike** (one single-game invocation exercises the leaf-evaluation
wiring, which is cheap and not a measurement); what is testable without it is
exactly the mechanical part of the gate:

* the projection arithmetic ``3600 E / (r S P)`` on fixed inputs, including its
  loud rejection of nonpositive factors;
* the GO predicate above, below, and *exactly at* the floor — the ``>=`` vs.
  ``>`` boundary §12 M2.5 pins ("GO iff ``games_per_hour_full >= 100``");
* the exit-code mapping, including that an unofficial run never returns a
  verdict code whatever the arithmetic says;
* the hardware gate — no official report without CUDA on the canonical device,
  and the ``--allow-unverified-hardware`` path labelling itself in the title,
  the intro, the measurement heading and the verdict, with the 4060 Ti slot
  left PENDING;
* the pinned measurement interval being un-overridable on the official path;
* the AMP flag being **observed** on both timed paths rather than re-derived
  from the device — CPU autocast (bfloat16) is real, so a forced-on mode on
  CPU is a genuine falsifier of a flag wired to ``device.type == "cuda"``;
* the persistence rule: an official run always leaves gate evidence behind
  (defaulting to the canonical artifact), an unofficial one never writes
  implicitly.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_micro_throughput.py"


def load_bench():
    """Import ``scripts/bench_micro_throughput.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("bench_micro_throughput", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass resolution needs the sys.modules entry
    spec.loader.exec_module(module)
    return module


BENCH = load_bench()


def projection(games_per_hour, floor=100.0, go=None):
    """A fabricated Projection with the given rate (other factors plausible)."""
    return BENCH.Projection(
        net_evals_per_second=1000.0,
        forward_ratio=2.0,
        sims=128,
        plies_per_game=35,
        games_per_hour=games_per_hour,
        floor=floor,
        go=BENCH.is_go(games_per_hour, floor) if go is None else go,
    )


def spike_result(**overrides):
    """A fabricated SpikeResult; every field plausible, all timings inert."""
    fields = dict(
        games=200,
        plies=1240,
        sims=79360,
        net_evals=33000,
        learner_steps=200,
        legal_at_eval=270000,
        amp_evals=0,
        wall_seconds=60.0,
        self_play_seconds=45.0,
        train_seconds=15.0,
    )
    fields.update(overrides)
    return BENCH.SpikeResult(**fields)


def forward_ratio(micro_ms=1.5, full_ms=3.0, trials=50, amp=False):
    """A fabricated ForwardRatio."""
    return BENCH.ForwardRatio(micro_ms=micro_ms, full_ms=full_ms, trials=trials, amp=amp)


def make_meta(**overrides):
    """A plausible RunMeta for report tests, with keyword overrides."""
    fields = dict(
        official=True,
        device="cuda",
        device_name="NVIDIA GeForce RTX 4060 Ti",
        device_memory_gib=16.0,
        amp=True,
        torch_version="2.9.0",
        cuda_version="12.8",
        config_path="configs/blokus_micro.json",
        run_seed=2500,
        net_init_seed=13521923826590675288,
        identity={
            "game": "blokus_duo",
            "game_config": "MICRO_CONFIG",
            "orientation_hash": "78ea621a",
        },
        warmup_games=50,
        measure_games=200,
        sims=64,
        date="2026-01-01",
    )
    fields.update(overrides)
    return BENCH.RunMeta(**fields)


def report(meta, proj):
    """Render a report for ``meta``/``proj`` with inert measurement inputs."""
    ratio = forward_ratio()
    rows = [("board cells", "25", "196", "7.84×")]
    return BENCH.build_report(meta, spike_result(), ratio, proj, rows)


# --- the projection arithmetic -------------------------------------------------


def test_projection_is_the_pinned_formula():
    """Fixed inputs → the §12 M2.5 number: 3600·E/(r·S·P), to the digit."""
    # 3600 * 1000 / (2 * 128 * 35) = 3_600_000 / 8960 = 401.785714...
    assert BENCH.project_games_per_hour(1000.0, 2.0, 128, 35) == pytest.approx(401.7857142857)
    # Doubling E doubles the rate; doubling r, S or P halves it.
    assert BENCH.project_games_per_hour(2000.0, 2.0, 128, 35) == pytest.approx(803.5714285714)
    assert BENCH.project_games_per_hour(1000.0, 4.0, 128, 35) == pytest.approx(200.8928571429)
    assert BENCH.project_games_per_hour(1000.0, 2.0, 256, 35) == pytest.approx(200.8928571429)
    assert BENCH.project_games_per_hour(1000.0, 2.0, 128, 70) == pytest.approx(200.8928571429)
    # r = 1 is the degenerate "same net" case: 3600·E/(S·P), not a special case.
    assert BENCH.project_games_per_hour(4480.0, 1.0, 128, 35) == pytest.approx(3600.0)


def test_projection_rejects_nonpositive_factors():
    """A zero or negative factor is a measurement bug, not a number to divide by."""
    for args in (
        (0.0, 2.0, 128, 35),
        (-1.0, 2.0, 128, 35),
        (1000.0, 0.0, 128, 35),
        (1000.0, -2.0, 128, 35),
        (1000.0, 2.0, 0, 35),
        (1000.0, 2.0, 128, 0),
    ):
        with pytest.raises(ValueError, match="positive"):
            BENCH.project_games_per_hour(*args)


def test_make_projection_reads_the_pinned_scalars_from_config():
    """S, P and the floor come from the config's throughput block, not from code."""
    from core.runconfig import MICRO_RUN_CONFIG_PATH, load_run_config

    cfg = load_run_config(MICRO_RUN_CONFIG_PATH)
    spike = spike_result(net_evals=33000, wall_seconds=33.0)  # E = 1000/s exactly
    proj = BENCH.make_projection(spike, forward_ratio(1.5, 3.0), cfg)
    assert proj.sims == cfg.throughput.projection_sims
    assert proj.plies_per_game == cfg.throughput.projection_plies_per_game
    assert proj.floor == cfg.throughput.min_projected_games_per_hour
    assert proj.net_evals_per_second == pytest.approx(1000.0)
    assert proj.forward_ratio == pytest.approx(2.0)
    assert proj.games_per_hour == pytest.approx(
        BENCH.project_games_per_hour(1000.0, 2.0, proj.sims, proj.plies_per_game)
    )
    assert proj.go is BENCH.is_go(proj.games_per_hour, proj.floor)


# --- the GO predicate, incl. the boundary --------------------------------------


def test_go_predicate_at_above_and_below_the_floor():
    """GO iff projected >= floor — the boundary is inclusive (§12 M2.5's 'iff >= 100')."""
    assert BENCH.is_go(100.0001, 100.0) is True  # above
    assert BENCH.is_go(100.0, 100.0) is True  # exactly at: '>=', not '>'
    assert BENCH.is_go(99.9999, 100.0) is False  # below


def test_exit_codes_separate_verdicts_from_provisional_runs():
    """0/2 are the signed verdicts; an unofficial run is always 3, never a verdict."""
    go, no_go = projection(150.0), projection(50.0)
    assert go.go is True and no_go.go is False
    assert BENCH.verdict_exit_code(go, official=True) == BENCH.EXIT_GO == 0
    assert BENCH.verdict_exit_code(no_go, official=True) == BENCH.EXIT_NO_GO == 2
    # A provisional GO must not be indistinguishable from the signed one.
    assert BENCH.verdict_exit_code(go, official=False) == BENCH.EXIT_UNOFFICIAL == 3
    assert BENCH.verdict_exit_code(no_go, official=False) == BENCH.EXIT_UNOFFICIAL


def test_verdict_line_labels_provisional_runs():
    """Official runs say GO/NO-GO; unofficial ones say NO VERDICT and why."""
    assert BENCH.verdict_line(projection(150.0), official=True).startswith("GO:")
    assert BENCH.verdict_line(projection(50.0), official=True).startswith("NO-GO:")
    provisional = BENCH.verdict_line(projection(150.0), official=False)
    assert provisional.startswith("NO VERDICT")
    assert BENCH.UNOFFICIAL_TAG in provisional
    assert "4060 Ti" in provisional


# --- the hardware gate ----------------------------------------------------------


def test_canonical_hardware_recognition():
    """Only the 4060 Ti 16 GB is canonical — not other GPUs, not the 8 GB variant."""
    assert BENCH.is_canonical_hardware("NVIDIA GeForce RTX 4060 Ti", 16 * 2**30)
    assert not BENCH.is_canonical_hardware("NVIDIA GeForce RTX 4090", 24 * 2**30)
    assert not BENCH.is_canonical_hardware("NVIDIA GeForce RTX 4060 Ti", 8 * 2**30)


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="CUDA present: the no-CUDA refusal cannot fire"
)
def test_refuses_an_official_run_without_cuda():
    """No CUDA and no opt-in → SystemExit naming the pinned device and the flag."""
    with pytest.raises(SystemExit, match="4060 Ti"):
        BENCH.require_official_hardware(allow_unverified=False)
    official, device = BENCH.require_official_hardware(allow_unverified=True)
    assert official is False
    assert device.type == "cpu"  # never silently promoted to a verdict-capable device


def test_refuses_an_official_run_on_the_wrong_gpu(monkeypatch):
    """CUDA on a 4090 is still not the pinned device: refused without the opt-in."""
    monkeypatch.setattr(BENCH.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        BENCH.torch.cuda, "get_device_name", lambda device: "NVIDIA GeForce RTX 4090"
    )
    monkeypatch.setattr(
        BENCH.torch.cuda,
        "get_device_properties",
        lambda device: type("Props", (), {"total_memory": 24 * 2**30})(),
    )
    with pytest.raises(SystemExit, match="4090"):
        BENCH.require_official_hardware(allow_unverified=False)
    official, _ = BENCH.require_official_hardware(allow_unverified=True)
    assert official is False


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="CUDA present: the no-CUDA refusal cannot fire"
)
def test_script_exits_loudly_without_cuda_or_opt_in(tmp_path):
    """End to end: no CUDA, no flag → nonzero exit, error on stderr, no report written."""
    out = tmp_path / "report.md"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode != 0
    assert "4060 Ti" in proc.stderr
    assert "--allow-unverified-hardware" in proc.stderr
    assert proc.stdout == ""  # nothing report-like emitted
    assert not out.exists()  # exited before any spike, and before any write


def test_measurement_interval_is_pinned_on_the_official_path():
    """--warmup-games/--measure-games need the opt-in; the official run uses the config."""
    for override in (["--warmup-games", "1"], ["--measure-games", "5"]):
        with pytest.raises(SystemExit):
            BENCH.parse_args(override)
    args = BENCH.parse_args(["--allow-unverified-hardware", "--warmup-games", "1"])
    assert args.warmup_games == 1
    default = BENCH.parse_args([])
    assert default.warmup_games is None and default.measure_games is None  # → config's 50/200
    assert default.allow_unverified_hardware is False


def test_parse_args_rejects_degenerate_measurement_knobs():
    """Trials/playouts must be >= 1 and forward warm-up >= 0."""
    for argv in (
        ["--forward-trials", "0"],
        ["--playouts", "0"],
        ["--forward-warmup", "-1"],
        ["--allow-unverified-hardware", "--measure-games", "0"],
        ["--allow-unverified-hardware", "--warmup-games", "-1"],
    ):
        with pytest.raises(SystemExit):
            BENCH.parse_args(argv)


# --- the report -----------------------------------------------------------------


def test_official_report_carries_the_predicate_method_and_verdict():
    """The signed artifact states the pinned predicate, the model, and GO/NO-GO."""
    text = report(make_meta(), projection(150.0))
    assert text.startswith("# M2.5 throughput go/no-go")
    assert BENCH.UNOFFICIAL_TAG not in text
    assert "PENDING" not in text
    assert "games_per_hour_full = 3600 * E / (r * S * P)" in text
    assert "GO iff games_per_hour_full >= 100" in text
    assert "## Scaling model, and where it is weak" in text
    assert "weakest assumption in the gate" in text
    assert "**GO: projected 150.0 full-game games/hour" in text
    assert "signed gate evidence for starting M3" in text
    no_go = report(make_meta(), projection(50.0))
    assert "**NO-GO: projected 50.0" in no_go
    assert "routes back to the design doc" in no_go


def test_provisional_report_is_labelled_everywhere_and_leaves_the_slot_pending():
    """An unofficial run can never be mistaken for the signed verdict."""
    text = report(
        make_meta(official=False, device="cpu", device_name="cpu host", amp=False),
        projection(150.0),
    )
    head, _, tail = text.partition("## Method")
    # Title, intro and the measurement heading are all stamped.
    assert text.splitlines()[0].endswith(f"[{BENCH.UNOFFICIAL_TAG}]")
    assert f"**{BENCH.UNOFFICIAL_TAG} — this is not the gate verdict.**" in head
    assert f"## Measured — micro loop ({BENCH.UNOFFICIAL_TAG})" in tail
    # The verdict slot stays open, with the exact command that fills it.
    assert "### RTX 4060 Ti verdict — PENDING" in tail
    assert "| verdict | _pending — GO / NO-GO_ |" in tail
    assert f"--out {BENCH.CANONICAL_OUT}" in tail
    assert "NO VERDICT" in tail
    assert "nothing above is gate evidence" in tail
    # ...and it never claims the signed provenance.
    assert "signed gate evidence for starting M3" not in text


def test_report_states_the_measurements_a_no_go_would_be_diagnosed_from():
    """A NO-GO must be diagnosable: per-phase split and the loop counters are present."""
    text = report(make_meta(), projection(150.0))
    for label in (
        "end-to-end games/hour",
        "sims/sec (end-to-end)",
        "net-evals/sec (end-to-end) — **E**",
        "net-evals/sec (self-play time only)",
        "learner steps/sec",
        "net-evals per sim",
        "— self-play share",
        "— learner share",
    ):
        assert f"| {label} |" in text
    assert "| 75.0% |" in text  # 45 s of 60 s self-play
    assert "| 25.0% |" in text  # 15 s of 60 s learner


def test_ratio_rows_are_read_off_the_adapters():
    """The micro:full table is derived from the two instances, never hardcoded."""
    from games.blokus_duo import BlokusDuo
    from games.blokus_duo.config import FULL_CONFIG, MICRO_CONFIG

    micro, full = BlokusDuo(config=MICRO_CONFIG), BlokusDuo(config=FULL_CONFIG)
    stats = BENCH.PlayoutStats(playouts=1, mean_plies=6.2, mean_legal=12.0)
    rows = BENCH.ratio_rows(micro, full, stats, stats, forward_ratio())
    table = {name: (m, f, r) for name, m, f, r in rows}
    assert table["board cells"][:2] == ("25", "196")  # §5.3 vs. the 14×14 golden
    assert table["input planes"][:2] == ("12", "46")  # D3 vs. the micro plane count
    assert table["raw actions"][:2] == ("225", "17,836")  # (5,5,9) vs. 14×14×91
    assert table["raw actions"][2] == "79.27×"
    assert table["batch-1 forward (ms, measured)"] == ("1.500", "3.000", "2.00×")


def test_playout_stats_are_seeded_and_shaped():
    """Playout statistics are reproducible and match the micro instance's golden shape."""
    from games.blokus_duo import BlokusDuo
    from games.blokus_duo.config import MICRO_CONFIG

    micro = BlokusDuo(config=MICRO_CONFIG)
    stats = BENCH.playout_stats(micro, seed=2500, playouts=3)
    assert stats == BENCH.playout_stats(micro, seed=2500, playouts=3)
    assert stats.playouts == 3
    assert stats.mean_plies >= 2  # both openings are always playable
    assert stats.mean_legal > 0
    with pytest.raises(ValueError, match="positive"):
        BENCH.playout_stats(micro, seed=0, playouts=0)


def test_counting_evaluator_counts_without_touching_the_result():
    """The wrapper is transparent: same (value, priors) out, counters incremented."""
    calls = []
    off = BENCH.AmpMode(device_type="cpu", enabled=False)

    def fake(game, state):
        """A stand-in evaluator returning a fixed value and three priors."""
        calls.append(state)
        return 0.25, {1: 0.1, 2: 0.2, 3: 0.3}

    counted, counters = BENCH.counting_evaluator(fake, off)
    assert counted(None, "s0") == (0.25, {1: 0.1, 2: 0.2, 3: 0.3})
    assert counted(None, "s1") == (0.25, {1: 0.1, 2: 0.2, 3: 0.3})
    assert counters == [2, 6, 0]  # two evaluations, six legal actions, no AMP
    assert calls == ["s0", "s1"]
    # A uniform-prior evaluator (priors None) still counts the evaluation.
    none_counted, none_counters = BENCH.counting_evaluator(lambda game, state: (0.0, None), off)
    none_counted(None, "s")
    assert none_counters == [1, 0, 0]


def test_display_path_keeps_the_committed_report_machine_independent():
    """In-repo paths render repo-relative; outside ones stay absolute."""
    assert BENCH.display_path(ROOT / "configs" / "blokus_micro.json") == "configs/blokus_micro.json"
    outside = ROOT.parent / "not-in-this-repo" / "config.json"
    assert BENCH.display_path(outside) == str(outside)


def test_measure_forward_ratio_rejects_degenerate_budgets():
    """Guarded before any net is built — a zero-trial 'measurement' is not one."""
    off = BENCH.AmpMode(device_type="cpu", enabled=False)
    with pytest.raises(ValueError, match="trials must be positive"):
        BENCH.measure_forward_ratio(None, None, torch.device("cpu"), trials=0, warmup=1, amp=off)
    with pytest.raises(ValueError, match="warmup must be non-negative"):
        BENCH.measure_forward_ratio(None, None, torch.device("cpu"), trials=1, warmup=-1, amp=off)


# --- AMP is measured, not asserted ----------------------------------------------
#
# The falsifier these tests lean on: CPU autocast is *real* (bfloat16), so an
# AmpMode forced on while running on CPU separates "what the run actually did"
# from "what device it ran on". Any flag re-derived as ``device.type == "cuda"``
# reports False there and fails these tests — which is precisely the FP32-while-
# recording-AMP defect they exist to prevent.


def micro_game():
    """The micro-Blokus adapter (built per call; the instance is cheap)."""
    from games.blokus_duo import BlokusDuo
    from games.blokus_duo.config import MICRO_CONFIG

    return BlokusDuo(config=MICRO_CONFIG)


def test_amp_mode_resolves_once_and_reports_torchs_live_state():
    """One resolved setting opens the context; ``observed`` reads torch, not the field."""
    cpu = BENCH.AmpMode.resolve(torch.device("cpu"))
    assert (cpu.device_type, cpu.enabled) == ("cpu", False)
    assert BENCH.AmpMode.resolve(torch.device("cuda")).enabled is True  # no GPU needed to resolve
    # Outside any context nothing is live, whatever the mode claims.
    forced = BENCH.AmpMode(device_type="cpu", enabled=True)
    assert forced.observed() is False
    with cpu.autocast():
        assert cpu.observed() is False  # a disabled context is an exact no-op
    with forced.autocast():
        assert forced.observed() is True
        # ...and it is a real computation change, not just a flag: CPU autocast
        # runs autocast-eligible ops in bfloat16.
        assert (torch.zeros(2, 2) @ torch.zeros(2, 2)).dtype is torch.bfloat16
    assert (torch.zeros(2, 2) @ torch.zeros(2, 2)).dtype is torch.float32


def test_leaf_evaluations_are_counted_as_amp_only_when_they_ran_under_it():
    """The self-play leaf path: wrapped evaluations count as AMP, bare ones do not."""
    forced = BENCH.AmpMode(device_type="cpu", enabled=True)
    dtypes = []

    def fake(game, state):
        """A stand-in evaluator recording the precision its 'forward' ran at."""
        dtypes.append((torch.zeros(2, 2) @ torch.zeros(2, 2)).dtype)
        return 0.0, {1: 0.5}

    counted, counters = BENCH.counting_evaluator(fake, forced)
    wrapped = BENCH.amp_evaluator(counted, forced)
    assert wrapped(None, "s0") == (0.0, {1: 0.5})  # transparent
    assert counters == [1, 1, 1]
    # The pre-fix wiring, reproduced: call the evaluator *without* the context.
    counted(None, "s1")
    assert counters == [2, 2, 1]  # the bare evaluation is not counted as AMP
    assert dtypes == [torch.bfloat16, torch.float32]


def test_spike_amp_flag_is_all_or_nothing_over_the_interval():
    """Partial autocast coverage is a wiring bug, not a flag value."""
    assert spike_result(net_evals=100, amp_evals=100).amp is True
    assert spike_result(net_evals=100, amp_evals=0).amp is False
    with pytest.raises(ValueError, match="whole interval"):
        _ = spike_result(net_evals=100, amp_evals=40).amp


def test_observed_amp_refuses_paths_that_ran_at_different_precisions():
    """Wiring the two timed paths separately is exactly the drift this catches."""
    live = spike_result(net_evals=10, amp_evals=10)
    dead = spike_result(net_evals=10, amp_evals=0)
    assert BENCH.observed_amp(forward_ratio(amp=True), live) is True
    assert BENCH.observed_amp(forward_ratio(amp=False), dead) is False
    with pytest.raises(ValueError, match="disagree"):
        BENCH.observed_amp(forward_ratio(amp=True), dead)
    with pytest.raises(ValueError, match="disagree"):
        BENCH.observed_amp(forward_ratio(amp=False), live)


def test_measure_forward_ratio_reports_the_amp_its_forwards_ran_under():
    """``r`` is timed inside the run's context, and the flag is read back out of it."""
    micro, cpu = micro_game(), torch.device("cpu")
    off = BENCH.measure_forward_ratio(
        micro, micro, cpu, trials=1, warmup=0, amp=BENCH.AmpMode("cpu", False)
    )
    assert off.amp is False
    on = BENCH.measure_forward_ratio(
        micro, micro, cpu, trials=1, warmup=0, amp=BENCH.AmpMode("cpu", True)
    )
    # Same device, same nets: only the resolved setting differs, so the flag
    # cannot be a restatement of the device.
    assert on.amp is True
    assert off.trials == on.trials == 1


def test_run_spike_evaluates_leaves_under_the_runs_amp_setting():
    """The production composition, on a one-game spike (not the 50/200 gate run).

    ``run_spike`` nests the counting wrapper *inside* ``amp_evaluator``; get that
    order wrong (or drop the wrapper, which is the defect this replaces) and the
    forced-on run below counts zero AMP evaluations while still evaluating
    leaves. Forced on/off on the same CPU device, so nothing here needs a GPU.
    """
    from core.runconfig import MICRO_RUN_CONFIG_PATH, load_run_config

    cfg = load_run_config(MICRO_RUN_CONFIG_PATH)
    game, cpu = BENCH.build_game(cfg), torch.device("cpu")
    on = BENCH.run_spike(cfg, game, cpu, 0, 1, BENCH.AmpMode("cpu", True), verbose=False)
    assert on.net_evals > 0
    assert on.amp_evals == on.net_evals  # every leaf ran inside the context
    assert on.amp is True
    off = BENCH.run_spike(cfg, game, cpu, 0, 1, BENCH.AmpMode("cpu", False), verbose=False)
    assert off.net_evals > 0 and off.amp_evals == 0
    assert off.amp is False  # ...and a disabled mode really is a no-op


# --- main's assembly: metadata provenance and persisted evidence ----------------


def drive_main(monkeypatch, tmp_path, *, official, argv, ratio_amp=False, spike_amp=False):
    """Run ``BENCH.main`` with the hardware gate and the expensive spike stubbed out.

    Nothing here runs the spike, builds the 14×14 adapter or touches CUDA: the
    hardware verdict, both measurements and the random playouts are replaced,
    leaving exactly main's assembly — metadata provenance, report, persistence,
    exit code — under test. ``BENCH.ROOT`` is redirected at ``tmp_path`` so the
    canonical artifact's *relative* path is exercised without writing to the
    committed one.

    Args:
        monkeypatch: The pytest fixture.
        tmp_path: Stand-in repo root.
        official: What the stubbed hardware gate reports.
        argv: The CLI arguments under test.
        ratio_amp: The AMP flag the stubbed forward measurement observed.
        spike_amp: Whether the stubbed spike observed AMP at every evaluation.

    Returns:
        ``main``'s exit code.
    """
    micro = micro_game()
    evals = 33000
    monkeypatch.setattr(BENCH, "ROOT", tmp_path)
    monkeypatch.setattr(
        BENCH, "require_official_hardware", lambda allow: (official, torch.device("cpu"))
    )
    monkeypatch.setattr(BENCH, "BlokusDuo", lambda config: micro)
    monkeypatch.setattr(
        BENCH, "playout_stats", lambda game, seed, playouts: BENCH.PlayoutStats(1, 6.2, 12.0)
    )
    monkeypatch.setattr(
        BENCH, "ratio_rows", lambda *args, **kwargs: [("board cells", "25", "196", "7.84×")]
    )
    monkeypatch.setattr(
        BENCH, "measure_forward_ratio", lambda *args, **kwargs: forward_ratio(amp=ratio_amp)
    )
    monkeypatch.setattr(
        BENCH,
        "run_spike",
        lambda *args, **kwargs: spike_result(net_evals=evals, amp_evals=evals if spike_amp else 0),
    )
    return BENCH.main(argv)


def test_reported_amp_is_the_observed_one_not_the_devices(monkeypatch, tmp_path, capsys):
    """The header states what the forwards ran under — here, on CPU, AMP **on**.

    The stubbed device is CPU throughout, so the pre-fix wiring
    (``amp = device.type == "cuda"``) reports "off" in both halves of this test
    and fails the first assertion: the flag can only follow the measurements.
    """
    drive_main(monkeypatch, tmp_path, official=True, argv=[], ratio_amp=True, spike_amp=True)
    assert "AMP **on**" in capsys.readouterr().out
    drive_main(monkeypatch, tmp_path, official=True, argv=[], ratio_amp=False, spike_amp=False)
    assert "AMP **off**" in capsys.readouterr().out
    # And a report can never be written at all if the two paths disagreed.
    with pytest.raises(ValueError, match="disagree"):
        drive_main(monkeypatch, tmp_path, official=True, argv=[], ratio_amp=True, spike_amp=False)


def test_official_run_without_out_persists_to_the_canonical_artifact(monkeypatch, tmp_path, capsys):
    """§12 M2.5 gates on persisted evidence: a signed verdict always leaves the file."""
    code = drive_main(monkeypatch, tmp_path, official=True, argv=[])
    out = capsys.readouterr().out
    artifact = tmp_path / BENCH.CANONICAL_OUT
    assert artifact.exists()
    assert artifact.read_text().startswith("# M2.5 throughput go/no-go")
    assert artifact.read_text() in out  # what was persisted is what was reported
    assert code == BENCH.EXIT_GO  # a signed verdict, and it left evidence behind
    # ...and the run says so, loudly, on stdout.
    assert "no --out given" in out
    assert BENCH.CANONICAL_OUT in out


def test_unofficial_run_without_out_persists_nothing(monkeypatch, tmp_path, capsys):
    """A provisional number is not evidence: it must not overwrite the artifact."""
    code = drive_main(monkeypatch, tmp_path, official=False, argv=["--allow-unverified-hardware"])
    out = capsys.readouterr().out
    assert code == BENCH.EXIT_UNOFFICIAL
    assert BENCH.UNOFFICIAL_TAG in out  # the numbers were emitted...
    assert list(tmp_path.rglob("*")) == []  # ...and nothing at all was written


def test_explicit_out_still_writes_on_the_unofficial_path(monkeypatch, tmp_path):
    """``--out`` is how the provisional report is regenerated; that path is unchanged."""
    dest = tmp_path / "elsewhere" / "report.md"
    code = drive_main(
        monkeypatch,
        tmp_path,
        official=False,
        argv=["--allow-unverified-hardware", "--out", str(dest)],
    )
    assert code == BENCH.EXIT_UNOFFICIAL
    assert BENCH.UNOFFICIAL_TAG in dest.read_text()
    assert not (tmp_path / BENCH.CANONICAL_OUT).exists()  # never the canonical one


def test_resolve_out_path_defaults_only_official_runs(monkeypatch, tmp_path):
    """The rule in isolation: official → the committed artifact, unofficial → nothing."""
    monkeypatch.setattr(BENCH, "ROOT", tmp_path)
    assert BENCH.resolve_out_path(None, official=True) == tmp_path / BENCH.CANONICAL_OUT
    assert BENCH.resolve_out_path(None, official=False) is None
    explicit = tmp_path / "elsewhere.md"
    assert BENCH.resolve_out_path(explicit, official=True) == explicit
    assert BENCH.resolve_out_path(explicit, official=False) == explicit
