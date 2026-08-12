"""The M2.5 early throughput go/no-go spike (§12 M2.5, task 8).

Throughput is §3's binding constraint, and §13 names the "throughput wall" as a
project risk that must **not** be first verified at M5. This script is the early
feasibility gate: it runs a dedicated micro-Blokus self-play spike, measures the
loop, projects the measurement onto the full 14×14 game, and emits the
pre-registered mechanical verdict.

**The pinned predicate (§12 M2.5, verbatim).** From the measured micro
net-evals/sec ``E`` and the measured batch-1 forward-time ratio
``r = t_full / t_micro`` of the two nets on the same device::

    games_per_hour_full = 3600 * E / (r * S * P)      S = 128, P = 35

and **GO iff ``games_per_hour_full >= min_projected_games_per_hour`` (100)**.
``S``, ``P`` and the floor are read from ``configs/blokus_micro.json``'s
``throughput`` block (``projection_sims`` / ``projection_plies_per_game`` /
``min_projected_games_per_hour``), which is golden-tested against the doc. The
predicate is pinned *in advance*: nothing here chooses it, and a NO-GO routes
back to the design doc (sims budget, config size, or pulling M5 levers forward)
before M3 starts — it is never a reason to soften the floor.

**The scaling model, stated plainly** (the projection is the load-bearing step;
a projection whose reasoning is hidden is not a gate):

* ``E`` is measured **end-to-end** over the measurement interval — self-play
  *and* the learner steps interleaved with it — so the loop's non-network cost
  is already inside it. The self-play-only rate is reported alongside as a
  sensitivity line.
* ``r`` rescales **one** thing: the per-simulation network cost, from the micro
  net's ``12×5×5 → (5,5,9)`` shape to the full net's ``46×14×14 → (14,14,91)``
  shape. The D5 8×128 trunk is identical in both (§5.3 keeps it unchanged
  precisely so this number transfers).
* ``S`` and ``P`` replace the micro loop's own sims/ply and plies/game with
  M3's fixed 128 sims and the assumed ≈35 plies of a full game.
* One net evaluation per simulation (batch-1 leaf inference — the known
  M2.5/M3 configuration, recorded as the M5 lever, not optimized here). The
  measured net-evals-per-sim ratio is reported so the assumption is checkable.

**Which assumptions are weak** (the first four lean *optimistic*, i.e. the true
full-game rate is likely below the projection):

1. **Dividing the whole loop's cost by ``r``.** ``E`` contains tree descent,
   move generation, ``apply``, state encoding and the learner step — none of
   which scale like the network. The full game's non-network cost grows much
   faster than ``r``: 17,836 raw actions vs. 225, 828 legal openings vs. 42, a
   196-cell board vs. 25, 46 planes encoded per leaf vs. 12. **This is the
   weakest assumption in the gate.**
2. **``r`` measured at batch 1 on a GPU is latency-bound.** Both forwards are
   dominated by kernel-launch overhead there, so the measured ``r`` can sit
   near 1 while the true compute ratio is several-fold — again optimistic.
3. **The learner step is not rescaled at all.** The micro step is batch 32 on a
   5×5 board; M3's is batch 256 on 14×14. Folding the micro learner into ``E``
   and then dividing by ``r`` understates the real learner share.
4. **``P = 35`` is an assumption, not a measurement.** The full game's measured
   random-playout mean plies is recorded in the ratio table as a check; the
   projection scales inversely with ``P``.
5. **Net-evals-per-sim differs between the two boards** (direction ambiguous,
   unlike 1–4). The micro tree is small enough that many simulations end in an
   already-expanded or terminal node and evaluate nothing, so the measured
   ratio sits well below 1; the full game's tree is nowhere near exhausted at
   128 sims, so its ratio is ≈1. ``E`` therefore carries more non-network work
   per evaluation than the full game will, while the full game's per-sim
   non-network work is itself far larger. Both are reported so the reader can
   see the gap rather than infer it.

**Hardware discipline** (mirrors ``scripts/bench_train_step.py``). The
**official** verdict requires **CUDA on an RTX 4060 Ti 16 GB** — the device
§12 M2.5 names — and the script exits loudly on anything else. A CPU/MPS run is
possible only behind the explicit ``--allow-unverified-hardware`` opt-in, is
labelled **UNOFFICIAL / PROVISIONAL** in every line it emits, and returns a
distinct exit code that is not a verdict at all. CI never runs either path (the
test battery exercises the arithmetic and the gates, never the spike).

**AMP is measured, not asserted.** §12 M2.5 pins "CUDA + AMP" and lists AMP
on/off among the recorded evidence, so both timed paths — the batch-1 forwards
behind ``r`` (both nets identically, or the ratio is not apples-to-apples) and
every self-play leaf evaluation — run inside the run's single
``run_micro.AmpMode`` context, inference-only (``torch.inference_mode`` +
autocast, no ``GradScaler``). The reported flag is then read back out of
:attr:`ForwardRatio.amp` and :attr:`SpikeResult.amp`, both *observed* from
inside those contexts, and the two must agree — nothing recomputes
``device.type == "cuda"`` for the header, so the report cannot claim a
precision the measurement did not use.

**Persistence.** §12 M2.5 gates on *persisted* evidence, so an official run
writes the report whether or not ``--out`` is given: without it the report goes
to the canonical artifact (:data:`CANONICAL_OUT`) and the path is announced on
stdout. An unofficial run without ``--out`` writes nothing — a provisional
number is not evidence, and must not overwrite the committed artifact by
default.

Exit codes::

    0  official run, GO
    2  official run, NO-GO          (routes back to the doc, per §12)
    3  unofficial run — provisional numbers only, no verdict either way

Usage::

    # The signed verdict, on the 4060 Ti box (persists to CANONICAL_OUT):
    python3 scripts/bench_micro_throughput.py

    # A provisional smoke anywhere else (clearly labelled, never a verdict);
    # --out is required for it to persist anything:
    python3 scripts/bench_micro_throughput.py --allow-unverified-hardware \\
        --out docs/bench/m2_5-throughput-gate.md
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ``run_micro`` (sibling script, on sys.path above): the spike times *the*
# learner step and *the* AMP policy, not copies of them — any drift between a
# re-implementation here and the real loop would silently change the measured
# number, which is the one thing this script exists to get right.
from run_micro import AmpMode, _learner_step, amp_evaluator, build_game, game_identity  # noqa: E402

from core.game import Game, State  # noqa: E402
from core.mcts import Evaluator  # noqa: E402
from core.network import Network, NetworkConfig, make_network_evaluator  # noqa: E402
from core.runconfig import MICRO_RUN_CONFIG_PATH, RunConfig, load_run_config  # noqa: E402
from core.seeding import GameRNGs, net_init_seed  # noqa: E402
from core.selfplay import ReplayWindow, RunRecord, play_game  # noqa: E402
from core.train import make_lr_scheduler, make_optimizer, make_scaler  # noqa: E402
from games.blokus_duo import BlokusDuo  # noqa: E402
from games.blokus_duo.config import FULL_CONFIG  # noqa: E402

# The canonical-hardware contract, identical to the M2 benchmark's: §12 M2.5
# pins "measured on the RTX 4060 Ti with CUDA + AMP". The VRAM floor tells the
# 16 GB card apart from the 8 GB variant, which reports the same device name.
CANONICAL_GPU_SUBSTRING = "4060 Ti"
CANONICAL_MIN_VRAM_GIB = 15.0

# The repo root, and the committed artifact's path relative to it (the M2
# report's docs/bench/ convention). An official run with no ``--out`` writes
# there: §12 M2.5 gates on persisted evidence.
ROOT = Path(__file__).resolve().parents[1]
CANONICAL_OUT = "docs/bench/m2_5-throughput-gate.md"

EXIT_GO = 0
EXIT_NO_GO = 2
EXIT_UNOFFICIAL = 3

UNOFFICIAL_TAG = "UNOFFICIAL / PROVISIONAL"


@dataclass(frozen=True)
class SpikeResult:
    """Raw counters from the measurement interval (warm-up already excluded).

    Attributes:
        games: Games completed in the measurement interval.
        plies: Plies played across those games.
        sims: Simulations run across those games (``plies * sims_per_move``).
        net_evals: Leaf network evaluations counted by the wrapping evaluator.
        learner_steps: Learner steps taken inside the interval.
        legal_at_eval: Legal-set sizes seen at evaluated leaves, summed.
        amp_evals: Of those evaluations, how many ran with autocast **observed**
            live — the measured half of the report's AMP flag.
        wall_seconds: End-to-end wall clock of the interval (self-play + learner).
        self_play_seconds: Self-play share of ``wall_seconds``.
        train_seconds: Learner share of ``wall_seconds``.
    """

    games: int
    plies: int
    sims: int
    net_evals: int
    learner_steps: int
    legal_at_eval: int
    amp_evals: int
    wall_seconds: float
    self_play_seconds: float
    train_seconds: float

    @property
    def games_per_hour(self) -> float:
        """End-to-end games/hour of the complete micro loop."""
        return 3600.0 * self.games / self.wall_seconds

    @property
    def plies_per_game(self) -> float:
        """Mean plies per micro game."""
        return self.plies / self.games

    @property
    def sims_per_second(self) -> float:
        """End-to-end simulations/sec."""
        return self.sims / self.wall_seconds

    @property
    def net_evals_per_second(self) -> float:
        """End-to-end net-evals/sec — the ``E`` of the pinned predicate."""
        return self.net_evals / self.wall_seconds

    @property
    def self_play_net_evals_per_second(self) -> float:
        """Net-evals/sec counting self-play time only (the sensitivity line)."""
        return self.net_evals / self.self_play_seconds

    @property
    def learner_steps_per_second(self) -> float:
        """End-to-end learner steps/sec."""
        return self.learner_steps / self.wall_seconds

    @property
    def net_evals_per_sim(self) -> float:
        """Evaluated leaves per simulation — the "one eval per sim" check."""
        return self.net_evals / self.sims

    @property
    def mean_legal_at_eval(self) -> float:
        """Mean legal-set size at evaluated leaves."""
        return self.legal_at_eval / self.net_evals

    @property
    def amp(self) -> bool:
        """Whether AMP was live at **every** leaf evaluation, as observed.

        Returns:
            True iff autocast was observed live at all ``net_evals``
            evaluations (and there was at least one).

        Raises:
            ValueError: If autocast covered only *part* of the interval. That
                is a wiring bug — half the measurement would then be FP32 while
                the report states a single precision — and it must not be
                reported as either "on" or "off".
        """
        if self.amp_evals not in (0, self.net_evals):
            raise ValueError(
                f"autocast was live for {self.amp_evals} of {self.net_evals} leaf "
                "evaluations; the reported AMP flag must describe the whole interval"
            )
        return self.net_evals > 0 and self.amp_evals == self.net_evals


@dataclass(frozen=True)
class ForwardRatio:
    """The measured batch-1 forward-time ratio of the two nets.

    Attributes:
        micro_ms: Median batch-1 forward time of the micro net, ms.
        full_ms: Median batch-1 forward time of the full 14×14 net, ms.
        trials: Timed forwards per net (after warm-up).
        amp: Whether autocast was **observed** live inside the timed forwards —
            identically for both nets, or the ratio would not be
            apples-to-apples (:func:`measure_forward_ratio` refuses otherwise).
    """

    micro_ms: float
    full_ms: float
    trials: int
    amp: bool

    @property
    def ratio(self) -> float:
        """``r = t_full / t_micro`` — the predicate's network-cost scale factor."""
        return self.full_ms / self.micro_ms


@dataclass(frozen=True)
class PlayoutStats:
    """Uniform-random-playout shape statistics for one game instance.

    Recorded for the micro:full ratio table so the extrapolation argument is
    inspectable. These feed the *table*, never the predicate — the projection's
    plies/game is the pinned ``projection_plies_per_game``.

    Attributes:
        playouts: Complete random games played.
        mean_plies: Mean plies per playout.
        mean_legal: Mean legal-set size over all plies.
    """

    playouts: int
    mean_plies: float
    mean_legal: float


@dataclass(frozen=True)
class Projection:
    """The full-game projection and its mechanical verdict.

    Attributes:
        net_evals_per_second: ``E``, the measured micro rate.
        forward_ratio: ``r``, the measured batch-1 forward-time ratio.
        sims: ``S``, ``throughput.projection_sims``.
        plies_per_game: ``P``, ``throughput.projection_plies_per_game``.
        games_per_hour: The projected full-game self-play rate.
        floor: ``throughput.min_projected_games_per_hour``.
        go: Whether the projection reaches the floor.
    """

    net_evals_per_second: float
    forward_ratio: float
    sims: int
    plies_per_game: int
    games_per_hour: float
    floor: float
    go: bool


@dataclass(frozen=True)
class RunMeta:
    """Everything the report header states about a run.

    Attributes:
        official: True only for a CUDA run on the canonical 4060 Ti 16 GB; the
            report is stamped provisional otherwise.
        device: The torch device the spike ran on.
        device_name: Human-readable device name (GPU model, or the CPU/MPS tag).
        device_memory_gib: Device memory in GiB, or ``None`` off CUDA.
        amp: Whether AMP was live — :func:`observed_amp`'s reconciliation of
            what the timed forwards and the leaf evaluations actually ran
            under, never a re-derivation from the device.
        torch_version: ``torch.__version__``.
        cuda_version: ``torch.version.cuda``, or ``"n/a"``.
        config_path: The run config the scalars came from.
        run_seed: The run seed every stream derives from.
        net_init_seed: The fresh-weights seed (self-play throughput is
            weight-independent — §12 M2.5 pins freshly initialized weights).
        identity: The game-identity block (game, instance config, orientation
            hash — Invariant 4).
        warmup_games: Games discarded before measurement.
        measure_games: Games in the measurement interval.
        sims: Micro sims/move during the spike.
        date: Run date, ``YYYY-MM-DD``.
    """

    official: bool
    device: str
    device_name: str
    device_memory_gib: float | None
    amp: bool
    torch_version: str
    cuda_version: str
    config_path: str
    run_seed: int
    net_init_seed: int
    identity: dict[str, str]
    warmup_games: int
    measure_games: int
    sims: int
    date: str


def _gib(n_bytes: int) -> float:
    """Convert bytes to GiB.

    Args:
        n_bytes: Byte count.

    Returns:
        The count in GiB (base 2).
    """
    return n_bytes / 2**30


def display_path(path: Path) -> str:
    """Render a path repo-relative when it is inside the repo, else absolute.

    The report is a committed artifact: an absolute path baked into it would be
    a machine-specific detail that changes on every box the spike runs on.

    Args:
        path: The path to render.

    Returns:
        The repo-relative path (POSIX separators) if ``path`` lies under the
        repo root, otherwise its absolute form.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(ROOT).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_out_path(out: Path | None, official: bool) -> Path | None:
    """Decide where — if anywhere — the report is persisted.

    §12 M2.5's gates are "pass/fail on *persisted* evidence", so an **official**
    verdict may never be emitted with nothing left behind: without ``--out`` it
    defaults to the committed artifact :data:`CANONICAL_OUT`. An **unofficial**
    run defaults to writing nothing — a provisional number is not evidence, and
    silently overwriting the committed artifact with one would be worse than
    printing it to stdout alone (``--out`` remains available to do so
    deliberately, which is how the provisional report is regenerated).

    Args:
        out: The ``--out`` path, or ``None``.
        official: Whether this run may sign a verdict.

    Returns:
        The path to write, or ``None`` for an unofficial run without ``--out``.
    """
    if out is not None:
        return out
    return Path(ROOT) / CANONICAL_OUT if official else None


def is_canonical_hardware(gpu_name: str, total_memory_bytes: int) -> bool:
    """Whether the device is the documented RTX 4060 Ti 16 GB.

    Args:
        gpu_name: ``torch.cuda.get_device_name`` of the device.
        total_memory_bytes: Device memory; the floor distinguishes the 8 GB
            4060 Ti variant, which reports the same name.

    Returns:
        True iff the name and memory match the §12 M2.5 contract.
    """
    return (
        CANONICAL_GPU_SUBSTRING in gpu_name and _gib(total_memory_bytes) >= CANONICAL_MIN_VRAM_GIB
    )


def require_official_hardware(allow_unverified: bool) -> tuple[bool, torch.device]:
    """Resolve the run device, refusing to fake an official verdict.

    The official verdict is defined on one machine (§12 M2.5: "measured on the
    RTX 4060 Ti with CUDA + AMP"). Without that machine the only thing this
    script may produce is an explicitly-opted-into provisional number — never a
    silent CPU verdict, and never a relabelled one.

    Args:
        allow_unverified: The ``--allow-unverified-hardware`` opt-in.

    Returns:
        ``(official, device)`` — ``official`` is True only for CUDA on the
        canonical 4060 Ti 16 GB.

    Raises:
        SystemExit: When the hardware is not canonical and the opt-in is
            absent; the message names the contract and the flag.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(device)
        total = torch.cuda.get_device_properties(device).total_memory
        if is_canonical_hardware(name, total):
            return True, device
        detail = f"CUDA device is {name} ({_gib(total):.1f} GiB)"
    else:
        device = torch.device("cpu")
        detail = "CUDA is unavailable"
    if allow_unverified:
        return False, device
    raise SystemExit(
        f"ERROR: {detail}; the M2.5 throughput go/no-go is pinned to the RTX "
        f"{CANONICAL_GPU_SUBSTRING} 16 GB with CUDA + AMP (§12 M2.5), so no official "
        "verdict can be produced here. Rerun with --allow-unverified-hardware for a "
        f"{UNOFFICIAL_TAG.lower()} smoke that is labelled as such everywhere and "
        f"exits {EXIT_UNOFFICIAL} (no verdict)."
    )


def counting_evaluator(evaluator: Evaluator, amp: AmpMode) -> tuple[Evaluator, list[int]]:
    """Wrap an evaluator so leaf evaluations, legal sets and AMP are counted.

    The spike's whole instrumentation trick: counters live in this wrapper, so
    ``core/mcts.py`` needs no change. M3's observability task formalizes the
    same counters in ``core/metrics.py``; these are deliberately script-local
    and do not satisfy it.

    The third counter is the AMP measurement: it asks torch, per evaluation,
    whether autocast is live *at that moment*. For it to see anything, this
    wrapper must sit **inside** ``run_micro.amp_evaluator``'s context — which
    is exactly how :func:`run_spike` composes them, and why the report's AMP
    flag describes the forwards that ran rather than the device they ran on.

    Args:
        evaluator: The evaluator to wrap (``core.network.make_network_evaluator``).
        amp: The run's resolved AMP mode; queried, never used to open a context
            here (opening one here would make the counter self-fulfilling).

    Returns:
        ``(wrapped, counters)`` where ``counters`` is the mutable triple
        ``[evaluations, summed_legal_set_size, evaluations_under_live_autocast]``,
        read after the run.
    """
    counters = [0, 0, 0]

    def counted(game: Game, state: State) -> tuple[float, dict[int, float] | None]:
        """Evaluate one leaf, incrementing the spike's counters.

        Args:
            game: The adapter, passed through untouched.
            state: The leaf state.

        Returns:
            The wrapped evaluator's ``(value, priors)`` result, unmodified.
        """
        live = amp.observed()
        value, priors = evaluator(game, state)
        counters[0] += 1
        counters[1] += 0 if priors is None else len(priors)
        counters[2] += 1 if live else 0
        return value, priors

    return counted, counters


def observed_amp(ratio: ForwardRatio, spike: SpikeResult) -> bool:
    """Reconcile the two measured AMP observations into the reported flag.

    The single value the report states, and the only one it may state: both
    timed paths must have run at the same precision, or "AMP on/off" describes
    neither of them.

    Args:
        ratio: The forward-ratio measurement, carrying its observed flag.
        spike: The self-play measurement, carrying its observed flag.

    Returns:
        The AMP setting both paths actually ran under.

    Raises:
        ValueError: If the two disagree (or, via :attr:`SpikeResult.amp`, if
            autocast covered only part of the measurement interval).
    """
    if ratio.amp != spike.amp:
        raise ValueError(
            f"AMP observations disagree: timed forwards ran with autocast "
            f"{'on' if ratio.amp else 'off'} but leaf evaluations with autocast "
            f"{'on' if spike.amp else 'off'}; the projection combines both, so one "
            "reported flag cannot describe them"
        )
    return ratio.amp


def playout_stats(game: Game, seed: int, playouts: int) -> PlayoutStats:
    """Measure mean plies and mean legal-set size from uniform-random playouts.

    Network-free and search-free: this is the shape of the *game*, measured the
    same way on both instances so the micro:full ratio table compares like with
    like.

    Args:
        game: The adapter to play.
        seed: RNG seed; same seed, same statistics.
        playouts: Complete random games to play.

    Returns:
        The collected :class:`PlayoutStats`.

    Raises:
        ValueError: If ``playouts`` is not positive.
    """
    if playouts < 1:
        raise ValueError(f"playouts must be positive, got {playouts}")
    rng = random.Random(seed)
    sizes: list[int] = []
    plies: list[int] = []
    for _ in range(playouts):
        state = game.initial_state()
        count = 0
        while not game.is_terminal(state):
            legal = game.legal_moves(state)
            sizes.append(len(legal))
            state = game.apply(state, rng.choice(list(legal)))
            count += 1
        plies.append(count)
    return PlayoutStats(
        playouts=playouts,
        mean_plies=statistics.fmean(plies),
        mean_legal=statistics.fmean(sizes),
    )


def measure_forward_ratio(
    micro: Game, full: Game, device: torch.device, trials: int, warmup: int, amp: AmpMode
) -> ForwardRatio:
    """Time batch-1 forwards of the micro and full nets on the same device.

    Both nets carry the identical D5 8×128 trunk (§5.3 keeps it unchanged so
    exactly this ratio transfers); only the input planes, board area and policy
    head differ. Weights are irrelevant to timing, so both are freshly
    initialized. Median of ``trials`` timed forwards under
    ``torch.inference_mode`` **and** the run's autocast context — the same
    precision self-play runs at, and identically for both nets, since ``r`` is
    only meaningful as an apples-to-apples ratio — after ``warmup`` untimed
    ones, with a CUDA synchronize bracketing each timed forward so the number
    is not an async launch time. Whether autocast was live is read back out of
    the context rather than assumed.

    Args:
        micro: The micro-Blokus adapter.
        full: The full 14×14 adapter.
        device: The device both nets are timed on.
        trials: Timed forwards per net.
        warmup: Untimed forwards per net.
        amp: The run's resolved AMP mode; opens the context and is observed
            inside it.

    Returns:
        The measured :class:`ForwardRatio`, its ``amp`` field observed.

    Raises:
        ValueError: If ``trials`` is not positive, ``warmup`` is negative, or
            the two nets somehow did not run at the same precision.
    """
    if trials < 1:
        raise ValueError(f"trials must be positive, got {trials}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")

    def median_ms(game: Game) -> tuple[float, bool]:
        """Median batch-1 forward time of ``game``'s net in ms, and the live AMP state."""
        net = Network(NetworkConfig.from_game(game)).to(device).eval()
        planes = torch.zeros(1, game.input_planes, *game.input_shape, device=device)
        times: list[float] = []
        with torch.inference_mode(), amp.autocast():
            for _ in range(warmup):
                net(planes)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            for _ in range(trials):
                mark = time.perf_counter()
                net(planes)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                times.append((time.perf_counter() - mark) * 1000.0)
            # Read outside the timed brackets, inside the context that ran them.
            live = amp.observed()
        return statistics.median(times), live

    micro_ms, micro_amp = median_ms(micro)
    full_ms, full_amp = median_ms(full)
    if micro_amp != full_amp:
        raise ValueError(
            f"r must be apples-to-apples: micro net timed with autocast "
            f"{'on' if micro_amp else 'off'}, full net with autocast "
            f"{'on' if full_amp else 'off'}"
        )
    return ForwardRatio(micro_ms=micro_ms, full_ms=full_ms, trials=trials, amp=micro_amp)


def run_spike(
    cfg: RunConfig,
    game: Game,
    device: torch.device,
    warmup_games: int,
    measure_games: int,
    amp: AmpMode,
    verbose: bool = True,
) -> SpikeResult:
    """Run the dedicated spike and return the measurement interval's counters.

    The loop is exactly ``scripts/run_micro.py``'s pacing — one self-play game,
    then ``training.steps_per_game`` learner steps, drawing from the same replay
    window — and the learner step is literally that module's, imported rather
    than re-implemented. Weights are freshly initialized (§12 M2.5: self-play
    throughput is weight-independent) and never checkpointed; nothing is
    persisted, because the artifact of this run is the report, not a run record.

    The first ``warmup_games`` games are played and **discarded** from the
    counters (allocator/cudnn steady state, replay-window fill); the counters
    are reset and the next ``measure_games`` games are the measurement.

    Args:
        cfg: The validated run config supplying every scalar.
        game: The micro adapter to play.
        device: Device the net and learner run on.
        warmup_games: Leading games excluded from the measurement.
        measure_games: Games in the measurement interval.
        amp: The run's resolved AMP mode. Leaf inference runs inside it (the
            learner step already reads the same setting off the batch's device
            inside ``core.train``), and the counting wrapper sits inside that
            context so the result carries an *observed* AMP flag.
        verbose: Print progress lines to stderr.

    Returns:
        The :class:`SpikeResult` for the measurement interval only.
    """
    torch.manual_seed(net_init_seed(cfg.run_seed))
    net = Network(NetworkConfig.from_game(game)).to(device)
    optimizer = make_optimizer(net, lr=cfg.training.learning_rate)
    scheduler = make_lr_scheduler(
        optimizer, cfg.training.warmup_steps, cfg.training.cosine_total_steps
    )
    scaler = make_scaler(device.type)
    window = ReplayWindow(cfg.training.replay_window)
    # The record is a throwaway sink: _learner_step appends to it, and nothing
    # here reads it back. The spike's artifact is the report.
    sink = RunRecord(
        run_name=cfg.name,
        run_seed=cfg.run_seed,
        config=cfg.to_dict(),
        game_identity=game_identity(cfg),
        device=str(device),
    )

    plies = net_evals = legal = amp_evals = measured_steps = 0
    self_play_seconds = train_seconds = 0.0
    started = 0.0
    # The learner-step index is monotonic across warm-up and measurement — it
    # keys the LR schedule and the per-step RNG streams, so it must never
    # restart; only the *measured* counters are reset at the boundary.
    step_index = 0
    for index in range(warmup_games + measure_games):
        measuring = index >= warmup_games
        if index == warmup_games:
            # Discard everything the warm-up accumulated, then start the clock.
            plies = net_evals = legal = amp_evals = measured_steps = 0
            self_play_seconds = train_seconds = 0.0
            started = time.perf_counter()

        net.eval()
        # Composition order is load-bearing: amp_evaluator opens the context,
        # counting_evaluator observes it from inside, the network evaluator
        # runs the forward under it.
        counted, counters = counting_evaluator(
            make_network_evaluator(net, game, device=str(device)), amp
        )
        evaluator = amp_evaluator(counted, amp)
        mark = time.perf_counter()
        result = play_game(game, evaluator, cfg.self_play, GameRNGs.for_game(cfg.run_seed, index))
        elapsed = time.perf_counter() - mark
        window.extend(result.samples)

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
                step_index,
                device,
                sink,
                games_played=index + 1,
            )
            step_index += 1
        train_elapsed = time.perf_counter() - mark

        if measuring:
            plies += result.plies
            net_evals += counters[0]
            legal += counters[1]
            amp_evals += counters[2]
            measured_steps += cfg.training.steps_per_game
            self_play_seconds += elapsed
            train_seconds += train_elapsed
        if verbose and (index + 1) % 25 == 0:
            phase = "measure" if measuring else "warmup"
            print(
                f"[spike] {phase} game {index + 1}/{warmup_games + measure_games}"
                f" plies={result.plies} window={len(window)}",
                file=sys.stderr,
                flush=True,
            )

    return SpikeResult(
        games=measure_games,
        plies=plies,
        sims=plies * cfg.self_play.sims,
        net_evals=net_evals,
        learner_steps=measured_steps,
        legal_at_eval=legal,
        amp_evals=amp_evals,
        wall_seconds=time.perf_counter() - started,
        self_play_seconds=self_play_seconds,
        train_seconds=train_seconds,
    )


def project_games_per_hour(
    net_evals_per_second: float,
    forward_ratio: float,
    sims: int,
    plies_per_game: int,
) -> float:
    """Project the full-game self-play rate: ``3600 E / (r S P)``.

    The §12 M2.5 formula, in one place so it is testable in isolation from any
    measurement. See the module docstring for what each factor assumes and
    which assumptions are weak.

    Args:
        net_evals_per_second: ``E`` — measured micro net-evals/sec.
        forward_ratio: ``r`` — measured ``t_full / t_micro`` at batch 1.
        sims: ``S`` — the projection's sim count (M3's fixed 128).
        plies_per_game: ``P`` — the projection's assumed plies/game.

    Returns:
        Projected full-game games/hour.

    Raises:
        ValueError: If any argument is not positive; every factor is a rate or
            a count, and a zero or negative one is a measurement bug, not a
            number to divide by.
    """
    values = {
        "net_evals_per_second": net_evals_per_second,
        "forward_ratio": forward_ratio,
        "sims": sims,
        "plies_per_game": plies_per_game,
    }
    bad = [f"{name}={value!r}" for name, value in values.items() if value <= 0]
    if bad:
        raise ValueError(f"projection inputs must all be positive; got {', '.join(bad)}")
    return 3600.0 * net_evals_per_second / (forward_ratio * sims * plies_per_game)


def is_go(projected_games_per_hour: float, floor: float) -> bool:
    """Apply the pinned GO predicate: **at or above** the floor is a GO.

    The comparison is ``>=``, not ``>`` — §12 M2.5 reads "GO iff
    ``games_per_hour_full >= 100``", so a projection landing exactly on the
    floor is a GO.

    Args:
        projected_games_per_hour: The projected full-game rate.
        floor: ``throughput.min_projected_games_per_hour``.

    Returns:
        True iff the projection reaches the floor.
    """
    return projected_games_per_hour >= floor


def make_projection(spike: SpikeResult, ratio: ForwardRatio, cfg: RunConfig) -> Projection:
    """Combine the spike and the forward ratio into the verdict object.

    Args:
        spike: The measurement interval's counters.
        ratio: The measured batch-1 forward-time ratio.
        cfg: The run config supplying ``S``, ``P`` and the floor.

    Returns:
        The :class:`Projection`, verdict included.
    """
    sims = cfg.throughput.projection_sims
    plies = cfg.throughput.projection_plies_per_game
    floor = cfg.throughput.min_projected_games_per_hour
    projected = project_games_per_hour(spike.net_evals_per_second, ratio.ratio, sims, plies)
    return Projection(
        net_evals_per_second=spike.net_evals_per_second,
        forward_ratio=ratio.ratio,
        sims=sims,
        plies_per_game=plies,
        games_per_hour=projected,
        floor=floor,
        go=is_go(projected, floor),
    )


def verdict_line(projection: Projection, official: bool) -> str:
    """Render the one-line verdict, provisional runs marked as such.

    Args:
        projection: The projection and its mechanical outcome.
        official: Whether this run may sign a verdict.

    Returns:
        A single line beginning ``GO``, ``NO-GO`` or ``NO VERDICT``.
    """
    call = "GO" if projection.go else "NO-GO"
    body = (
        f"projected {projection.games_per_hour:,.1f} full-game games/hour "
        f"vs. a floor of {projection.floor:,.0f} "
        f"(E = {projection.net_evals_per_second:,.1f} net-evals/s, r = "
        f"{projection.forward_ratio:.3f}, S = {projection.sims}, P = {projection.plies_per_game})"
    )
    if official:
        return f"{call}: {body}."
    return (
        f"NO VERDICT ({UNOFFICIAL_TAG}) — the arithmetic would read {call}, but this run "
        f"was not on the pinned RTX {CANONICAL_GPU_SUBSTRING} 16 GB: {body}."
    )


def verdict_exit_code(projection: Projection, official: bool) -> int:
    """Map the verdict onto the process exit status.

    An unofficial run returns :data:`EXIT_UNOFFICIAL` **whatever** the
    arithmetic says: a CPU number is not a decision in either direction, and a
    provisional GO exiting 0 would be indistinguishable from the signed one.

    Args:
        projection: The projection and its mechanical outcome.
        official: Whether this run may sign a verdict.

    Returns:
        :data:`EXIT_GO`, :data:`EXIT_NO_GO` or :data:`EXIT_UNOFFICIAL`.
    """
    if not official:
        return EXIT_UNOFFICIAL
    return EXIT_GO if projection.go else EXIT_NO_GO


def ratio_rows(
    micro: Game,
    full: Game,
    micro_stats: PlayoutStats,
    full_stats: PlayoutStats,
    ratio: ForwardRatio,
) -> list[tuple[str, str, str, str]]:
    """Build the micro:full ratio table rows (§12 M2.5's inspectability list).

    Every geometric figure is read off the two adapters rather than hardcoded,
    so the table cannot drift from the instances actually measured.

    Args:
        micro: The micro adapter.
        full: The full 14×14 adapter.
        micro_stats: Micro random-playout statistics.
        full_stats: Full-game random-playout statistics.
        ratio: The measured batch-1 forward-time ratio.

    Returns:
        ``(quantity, micro, full, full:micro)`` rows.
    """

    def area(game: Game) -> int:
        """Board cells of ``game``."""
        h, w = game.input_shape
        return h * w

    def actions(game: Game) -> int:
        """Raw policy-head width of ``game``."""
        total = 1
        for dim in game.policy_shape:
            total *= dim
        return total

    def row(name: str, m: float, f: float, fmt: str = ",.0f") -> tuple[str, str, str, str]:
        """One table row with its full:micro ratio."""
        return (name, f"{m:{fmt}}", f"{f:{fmt}}", f"{f / m:.2f}×")

    return [
        row("board cells", area(micro), area(full)),
        row("input planes", micro.input_planes, full.input_planes),
        row("raw actions", actions(micro), actions(full)),
        row(
            "mean legal-set size (random playouts)",
            micro_stats.mean_legal,
            full_stats.mean_legal,
            ".1f",
        ),
        row(
            "mean plies/game (random playouts)",
            micro_stats.mean_plies,
            full_stats.mean_plies,
            ".1f",
        ),
        row("batch-1 forward (ms, measured)", ratio.micro_ms, ratio.full_ms, ".3f"),
    ]


def build_report(
    meta: RunMeta,
    spike: SpikeResult,
    ratio: ForwardRatio,
    projection: Projection,
    rows: list[tuple[str, str, str, str]],
) -> str:
    """Render the markdown report — the gate's artifact.

    An unofficial run stamps ``UNOFFICIAL / PROVISIONAL`` in the title, the
    intro, the measurement heading and the verdict, and carries a **pending**
    section holding the 4060 Ti verdict's placeholder plus the exact command
    that fills it. An official run replaces that section with the signed
    verdict.

    Args:
        meta: Run header facts.
        spike: The measurement interval's counters.
        ratio: The measured batch-1 forward-time ratio.
        projection: The projection and its mechanical outcome.
        rows: The micro:full ratio table rows.

    Returns:
        The complete markdown report.
    """
    title = "# M2.5 throughput go/no-go — micro-Blokus self-play spike"
    if not meta.official:
        title += f" [{UNOFFICIAL_TAG}]"
    memory = "" if meta.device_memory_gib is None else f", {meta.device_memory_gib:.1f} GiB"
    intro = (
        [
            "The §12 M2.5 early feasibility gate, measured on the pinned hardware.",
            "",
            "```",
            f"python3 scripts/bench_micro_throughput.py --out {CANONICAL_OUT}",
            "```",
        ]
        if meta.official
        else [
            f"**{UNOFFICIAL_TAG} — this is not the gate verdict.** §12 M2.5 pins the"
            f" measurement to the RTX {CANONICAL_GPU_SUBSTRING} 16 GB with CUDA + AMP;"
            f" this run was on `{meta.device_name}`, so every number below is provisional"
            " and the verdict slot stays **PENDING**.",
            "",
            "```",
            "python3 scripts/bench_micro_throughput.py --allow-unverified-hardware"
            f" --out {CANONICAL_OUT}",
            "```",
        ]
    )
    amp_note = (
        "AMP **on** — autocast observed live inside the timed batch-1 forwards and at"
        " every leaf evaluation"
        if meta.amp
        else "AMP **off** (an exact no-op off CUDA) — observed, not assumed"
    )
    lines = [
        title,
        "",
        *intro,
        "",
        f"- **Device:** {meta.device_name}{memory} (`{meta.device}`), {amp_note}",
        f"- **torch:** {meta.torch_version} (CUDA {meta.cuda_version})",
        f"- **Date:** {meta.date}",
        f"- **Config:** `{meta.config_path}` — run_seed {meta.run_seed}, fresh weights"
        f" (net-init seed {meta.net_init_seed}), {meta.sims} sims/move, batch-1 leaf inference.",
        f"- **Protocol:** {meta.warmup_games} warm-up games discarded, then"
        f" {meta.measure_games} measured games (§12 M2.5's pinned interval).",
        f"- **Game identity:** {meta.identity['game']} / {meta.identity['game_config']},"
        f" orientation hash `{meta.identity['orientation_hash']}`.",
        "",
        "## The pinned predicate",
        "",
        "Pre-registered in §12 M2.5 and carried in `configs/blokus_micro.json`"
        " (`throughput`), fixed before any run:",
        "",
        "```",
        "games_per_hour_full = 3600 * E / (r * S * P)",
        f"GO iff games_per_hour_full >= {projection.floor:,.0f}",
        "```",
        "",
        f"with `S = {projection.sims}` (M3's fixed sims), `P = {projection.plies_per_game}`"
        " (assumed full-game plies), `E` the measured micro net-evals/sec, and"
        " `r = t_full / t_micro` the measured batch-1 forward-time ratio of the two nets on"
        " the same device. A NO-GO routes back to the design doc — sims budget, config size,"
        " or pulling M5 levers forward — before M3 starts; it is never a reason to move the"
        " floor.",
        "",
        "## Method",
        "",
        "A dedicated spike, not a training run: `scripts/run_micro.py`'s exact pacing (one"
        " self-play game, then one learner step drawing from the same replay window) with"
        " that module's learner step imported rather than re-implemented, freshly"
        " initialized weights (self-play throughput is weight-independent), and nothing"
        " persisted. Net evaluations and leaf legal-set sizes are counted by a wrapper"
        " around the evaluator — no `core/mcts.py` change; M3's observability task"
        " formalizes these counters in `core/metrics.py` and these script-local ones do not"
        " satisfy it. Warm-up games are played and discarded, then the counters are reset"
        " and the clock started.",
        "",
        f"## Measured — micro loop{'' if meta.official else f' ({UNOFFICIAL_TAG})'}",
        "",
        "| quantity | value |",
        "|---|--:|",
        f"| end-to-end games/hour | {spike.games_per_hour:,.1f} |",
        f"| mean plies/game | {spike.plies_per_game:.2f} |",
        f"| sims/sec (end-to-end) | {spike.sims_per_second:,.1f} |",
        f"| net-evals/sec (end-to-end) — **E** | {spike.net_evals_per_second:,.1f} |",
        f"| net-evals/sec (self-play time only) | {spike.self_play_net_evals_per_second:,.1f} |",
        f"| learner steps/sec | {spike.learner_steps_per_second:,.2f} |",
        f"| net-evals per sim | {spike.net_evals_per_sim:.3f} |",
        f"| mean legal-set size at evaluated leaves | {spike.mean_legal_at_eval:.2f} |",
        f"| wall clock, measured interval (s) | {spike.wall_seconds:,.1f} |",
        f"| — self-play share | {spike.self_play_seconds / spike.wall_seconds:.1%} |",
        f"| — learner share | {spike.train_seconds / spike.wall_seconds:.1%} |",
        "",
        "The per-phase split is what makes a NO-GO diagnosable: a learner-dominated interval"
        " points at the M3 actor–learner split, a self-play-dominated one at batched"
        " inference (the M5 lever) or the sims budget.",
        "",
        "## Micro:full ratio table",
        "",
        "| quantity | micro | full 14×14 | full:micro |",
        "|---|--:|--:|--:|",
        *[f"| {name} | {m} | {f} | {r} |" for name, m, f, r in rows],
        "",
        f"`r = {ratio.ratio:.3f}` is the last row's ratio — median of {ratio.trials} timed"
        " batch-1 forwards per net (after warm-up) on the same device, both nets carrying the"
        " identical D5 8×128 trunk (§5.3 keeps the trunk unchanged so exactly this number"
        " transfers).",
        "",
        "## Scaling model, and where it is weak",
        "",
        "`E` is measured **end-to-end** — self-play and the interleaved learner steps — so"
        " the loop's non-network cost is already inside it. `r` then rescales the"
        " per-simulation **network** cost from the micro net's"
        " `12×5×5 → (5,5,9)` shape to the full net's `46×14×14 → (14,14,91)` shape, and"
        " `S`/`P` substitute M3's sim count and the assumed full-game length. One net"
        " evaluation per simulation is assumed (batch-1 leaf inference — the known M2.5/M3"
        " configuration, recorded here as the M5 lever rather than optimized); the measured"
        " net-evals-per-sim above is the check on that.",
        "",
        "Five assumptions carry the projection, and **the first four all lean optimistic** —"
        " the true full-game rate is likely below the projected number:",
        "",
        "1. **Dividing the whole loop's cost by `r` alone.** `E` contains tree descent, move"
        " generation, `apply`, state encoding and the learner step, none of which scale like"
        " the network. The full game's non-network cost grows much faster than `r` — see the"
        " ratio table: 79× the raw actions, ~17× the mean legal-set size (828 legal openings"
        " at the root vs. 42), ~8× the board cells, ~4× the planes encoded per leaf, against"
        f" an `r` of {ratio.ratio:.2f}. **This is the weakest assumption in the gate.**",
        "2. **`r` measured at batch 1 on a GPU is latency-bound.** Both forwards are"
        " dominated by kernel-launch overhead there, so the measured `r` can sit near 1"
        " while the true compute ratio is several-fold.",
        "3. **The learner step is not rescaled at all.** The micro step is batch 32 on 5×5;"
        " M3's is batch 256 on 14×14. Folding the micro learner into `E` and then dividing"
        " by `r` understates the real learner share.",
        "4. **`P = 35` is an assumption, not a measurement.** The measured full-game"
        " random-playout mean plies is in the ratio table as a check; the projection scales"
        " inversely with `P`.",
        "5. **Net-evals-per-sim differs between the two boards** — direction ambiguous,"
        " unlike 1–4. The micro tree is small enough that many simulations end in an"
        " already-expanded or terminal node and evaluate nothing, so the measured ratio"
        f" above ({spike.net_evals_per_sim:.3f}) sits below 1; the full game's tree is"
        " nowhere near exhausted at 128 sims, so its ratio is ≈1. `E` therefore carries"
        " more non-network work per evaluation than the full game will, while the full"
        " game's per-sim non-network work is itself far larger.",
        "",
        "Consequence for reading the result: a projection comfortably **above** the floor is"
        " weaker evidence than it looks, and a projection **below** it is strong evidence —"
        " the optimistic model still failed.",
        "",
        "## Projection",
        "",
        "```",
        f"E = {projection.net_evals_per_second:,.4f} net-evals/s",
        f"r = {projection.forward_ratio:.4f}",
        f"S = {projection.sims}   P = {projection.plies_per_game}",
        f"games_per_hour_full = 3600 * {projection.net_evals_per_second:,.4f}"
        f" / ({projection.forward_ratio:.4f} * {projection.sims} * {projection.plies_per_game})"
        f" = {projection.games_per_hour:,.2f}",
        f"floor = {projection.floor:,.0f}",
        "```",
        "",
        "## Verdict",
        "",
        f"**{verdict_line(projection, meta.official)}**",
        "",
    ]
    if meta.official:
        lines += [
            "This is the signed gate evidence for starting M3."
            if projection.go
            else "NO-GO: per §12 M2.5 this routes back to the design doc (sims budget,"
            " micro/full config size, or pulling M5 levers forward) before M3 starts.",
            "",
        ]
    else:
        lines += [
            "### RTX 4060 Ti verdict — PENDING",
            "",
            "The signed verdict requires one run on the pinned hardware. Until then this"
            " section is a placeholder and nothing above is gate evidence.",
            "",
            "| field | value |",
            "|---|--:|",
            "| device | _pending — RTX 4060 Ti 16 GB, CUDA + AMP_ |",
            "| date | _pending_ |",
            "| E (net-evals/s) | _pending_ |",
            "| r (t_full / t_micro) | _pending_ |",
            "| projected full-game games/hour | _pending_ |",
            f"| floor | {projection.floor:,.0f} |",
            "| verdict | _pending — GO / NO-GO_ |",
            "",
            "Fill it by running, on the 4060 Ti box:",
            "",
            "```",
            f"python3 scripts/bench_micro_throughput.py --out {CANONICAL_OUT}",
            "```",
            "",
            "which refuses to run anywhere else, rewrites this file in place with the"
            " official heading, and exits"
            f" {EXIT_GO} on GO / {EXIT_NO_GO} on NO-GO.",
            "",
        ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the CLI.

    Args:
        argv: Argument list, or ``None`` for ``sys.argv``.

    Returns:
        The parsed namespace.

    Raises:
        SystemExit: On invalid arguments — nonpositive trial/playout counts, or
            a measurement-interval override without
            ``--allow-unverified-hardware`` (the pinned 50/200 interval is part
            of the protocol an official verdict is signed against).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=MICRO_RUN_CONFIG_PATH,
        help="run config JSON (default: the pinned configs/blokus_micro.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the report here; an official run without it persists to the committed"
        f" artifact {CANONICAL_OUT} (§12 M2.5 gates on persisted evidence), while an"
        " unofficial run without it writes nothing",
    )
    parser.add_argument(
        "--allow-unverified-hardware",
        action="store_true",
        help="opt in to a smoke run off the pinned RTX 4060 Ti 16 GB: every output is"
        f" labelled {UNOFFICIAL_TAG} and the process exits {EXIT_UNOFFICIAL} (no verdict)",
    )
    parser.add_argument(
        "--warmup-games",
        type=int,
        default=None,
        help="override throughput.warmup_games (smoke runs only)",
    )
    parser.add_argument(
        "--measure-games",
        type=int,
        default=None,
        help="override throughput.measure_games (smoke runs only)",
    )
    parser.add_argument(
        "--forward-trials", type=int, default=50, help="timed batch-1 forwards per net (r)"
    )
    parser.add_argument(
        "--forward-warmup", type=int, default=10, help="untimed batch-1 forwards per net"
    )
    parser.add_argument(
        "--playouts", type=int, default=4, help="random playouts per instance for the ratio table"
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-game progress lines")
    args = parser.parse_args(argv)
    if args.forward_trials < 1 or args.playouts < 1:
        parser.error("--forward-trials and --playouts must be >= 1")
    if args.forward_warmup < 0:
        parser.error("--forward-warmup must be >= 0")
    if args.warmup_games is not None and args.warmup_games < 0:
        parser.error("--warmup-games must be >= 0")
    if args.measure_games is not None and args.measure_games < 1:
        parser.error("--measure-games must be >= 1")
    if (
        args.warmup_games is not None or args.measure_games is not None
    ) and not args.allow_unverified_hardware:
        parser.error(
            "the measurement interval is pinned (§12 M2.5: 50 warm-up games discarded, then "
            "200 measured), and an official verdict is signed against it; "
            "--warmup-games/--measure-games require --allow-unverified-hardware"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the spike, project, and emit the report and the mechanical verdict.

    AMP is resolved once (:class:`~run_micro.AmpMode`) and that one value opens
    every autocast context; the header's AMP flag comes back out of the two
    measurements via :func:`observed_amp`, never from the device. An official
    run always persists its report (:func:`resolve_out_path`).

    Args:
        argv: Argument list, or ``None`` for ``sys.argv``.

    Returns:
        The process exit code (:func:`verdict_exit_code`).

    Raises:
        SystemExit: From :func:`require_official_hardware` when the hardware is
            not the pinned 4060 Ti 16 GB and the opt-in flag is absent.
        ValueError: From :func:`observed_amp` if the timed forwards and the
            leaf evaluations did not run at the same precision.
    """
    args = parse_args(argv)
    official, device = require_official_hardware(args.allow_unverified_hardware)
    cfg = load_run_config(args.config)
    warmup_games = (
        args.warmup_games if args.warmup_games is not None else cfg.throughput.warmup_games
    )
    measure_games = (
        args.measure_games if args.measure_games is not None else cfg.throughput.measure_games
    )

    micro = build_game(cfg)
    full = BlokusDuo(config=FULL_CONFIG)
    # Resolved once, here: the same object opens every autocast context below
    # and supplies (via the observations it enables) the reported AMP flag.
    amp = AmpMode.resolve(device)
    ratio = measure_forward_ratio(
        micro, full, device, args.forward_trials, args.forward_warmup, amp
    )
    print(
        f"[spike] batch-1 forward: micro {ratio.micro_ms:.3f} ms, full {ratio.full_ms:.3f} ms,"
        f" r = {ratio.ratio:.3f}",
        file=sys.stderr,
        flush=True,
    )
    micro_stats = playout_stats(micro, cfg.run_seed, args.playouts)
    full_stats = playout_stats(full, cfg.run_seed, args.playouts)
    spike = run_spike(cfg, micro, device, warmup_games, measure_games, amp, verbose=not args.quiet)
    projection = make_projection(spike, ratio, cfg)

    meta = RunMeta(
        official=official,
        device=str(device),
        device_name=(
            torch.cuda.get_device_name(device) if device.type == "cuda" else f"{device.type} host"
        ),
        device_memory_gib=(
            _gib(torch.cuda.get_device_properties(device).total_memory)
            if device.type == "cuda"
            else None
        ),
        amp=observed_amp(ratio, spike),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "n/a",
        config_path=display_path(args.config),
        run_seed=cfg.run_seed,
        net_init_seed=net_init_seed(cfg.run_seed),
        identity=game_identity(cfg),
        warmup_games=warmup_games,
        measure_games=measure_games,
        sims=cfg.self_play.sims,
        date=time.strftime("%Y-%m-%d"),
    )
    report = build_report(
        meta, spike, ratio, projection, ratio_rows(micro, full, micro_stats, full_stats, ratio)
    )
    print(report, end="")
    out_path = resolve_out_path(args.out, official)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        if args.out is None:
            # Loud, on stdout: an official verdict that left no artifact behind
            # would not be gate evidence, so say where the evidence went.
            print(
                f"[spike] no --out given on an official run: the gate evidence was written"
                f" to {display_path(out_path)} (§12 M2.5 gates on persisted evidence)."
            )
        print(f"[spike] wrote {out_path}", file=sys.stderr)
    return verdict_exit_code(projection, official)


if __name__ == "__main__":
    raise SystemExit(main())
