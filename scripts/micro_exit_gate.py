"""Evaluate the pre-registered M2.5 exit gate against a completed run (§12 M2.5, task 7).

M2.5 gates the far more expensive M3 build, so its exit test is *falsifiable*:
every threshold, window and seed was pinned in the design doc (§12 M2.5) and
mirrored into ``configs/blokus_micro.json`` **before** any run existed, and this
script only reads them. It computes nothing it could have tuned: the three
predicates are

1. **Strength** — the trained net, playing the rung-7 agent form (MCTS with the
   network supplying both priors and value) at the pinned eval sims with **no
   Dirichlet noise** and **argmax-N** move choice, scores at least
   ``evaluation.min_score_rate`` over ``evaluation.n_pairs`` mirrored pairs
   against rung 1 (uniform random), draws counted 0.5, played through the M1.6
   paired runner at ``evaluation.eval_seed`` with the start-square opening
   balancer keyed on the *micro* start squares.
2. **Policy loss** — ``mean(policy_loss over the last tail_window_steps)`` is at
   most ``policy_max_ratio ×`` the mean over the first ``head_window_steps``.
3. **Value loss** — the same relation at ``value_max_ratio``.

Verdict = PASS iff all three hold. Both loss predicates read the *persisted*
run record task 6 wrote (``core.selfplay.load_run_record``) — never a number
recomputed here, and never a single end-of-run minibatch.

**Identity is checked before anything is scored.** The run record's and the
checkpoint's ``(game, game_config, orientation_hash)`` must agree with each
other and with the hash re-derived from the config (Invariant 4), and the
record's embedded config must equal the pinned config file. A gate run against
a mismatched checkpoint exits ``2`` with a diagnostic rather than quietly
producing a score for the wrong instance.

Exit codes: ``0`` PASS, ``1`` FAIL, ``2`` the gate could not be evaluated.

Usage::

    python3 scripts/micro_exit_gate.py                       # the pinned run + config
    python3 scripts/micro_exit_gate.py --run-dir runs/try1 --config configs/blokus_micro.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.agents import Agent, RandomAgent  # noqa: E402
from core.elo import fit_elo, matches_from_pairs  # noqa: E402
from core.game import Action, Game, State  # noqa: E402
from core.mcts import MCTS  # noqa: E402
from core.network import Network, NetworkConfig, make_network_evaluator  # noqa: E402
from core.runconfig import MICRO_RUN_CONFIG_PATH, RunConfig, load_run_config  # noqa: E402
from core.runner import PairResult, play_pairs  # noqa: E402
from core.selfplay import load_run_record  # noqa: E402
from games.blokus_duo import BlokusDuo  # noqa: E402
from games.blokus_duo.baselines import start_square_balancer  # noqa: E402
from games.blokus_duo.pieces import orientation_table_hash  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Artifact names/tags. ``CHECKPOINT_SCHEMA`` and ``RUN_RECORD_NAME`` are the
# writer's (``scripts/run_micro.py``) — restated here as what this reader
# accepts, and pinned equal to the writer's by ``tests/test_micro_exit_gate.py``
# so the two can never drift apart silently.
CHECKPOINT_SCHEMA = "alpha-games/micro-checkpoint/v1"
RUN_RECORD_NAME = "run_record.json"
VERDICT_NAME = "exit_gate_verdict.json"
VERDICT_SCHEMA = "alpha-games/exit-gate-verdict/v1"

# The only evaluation protocol this gate knows how to realize (§12 M2.5). A
# config naming anything else is a doc-first change plus a branch here, never a
# silent substitution of whatever agent happens to be available.
SUPPORTED_AGENT_FORM = "rung7_mcts_policy_value"
SUPPORTED_OPPONENT = "rung1_uniform_random"
SUPPORTED_MOVE_SELECTION = "argmax_n"

# Predicate names, as they appear in the verdict record and the printed table.
STRENGTH = "strength"
POLICY_LOSS = "policy_loss"
VALUE_LOSS = "value_loss"


class NetworkMCTSAgent(Agent):
    """Ladder rung 7: MCTS with the network supplying priors *and* value.

    The evaluation-mode form §12 M2.5 pins: a fixed sim budget, **no** root
    Dirichlet noise (D7 is self-play-only), and deterministic argmax-N move
    choice with ``MCTS.best_action``'s tie-break (most visits, ties to the lowest
    action id). One search tree per move — no subtree reuse across moves — so a
    move is a pure function of the position and the budget, which is what makes
    the fixed paired set reproducible.

    The search is built on the ``game`` handed to :meth:`select_action`, not on
    a captured one: the pair runner plays game 2 through an opening-restricted
    wrapper, and the search must see that restriction.

    Args:
        evaluator: The network leaf evaluator
            (``core.network.make_network_evaluator``).
        sims: Simulations per move (``evaluation.sims``).
        seed: Accepted so the agent fits the runner's ``AgentFactory`` signature;
            deliberately **unused and unstored** — this form is deterministic (no
            noise, argmax N, and both the PUCT and argmax tie-breaks are
            id-ordered), so there is no stream to seed and no hidden state that
            could make two identically-configured agents differ.
    """

    def __init__(self, evaluator, sims: int, seed: int = 0):
        self._evaluator = evaluator
        self._sims = sims

    @property
    def name(self) -> str:
        return SUPPORTED_AGENT_FORM

    def select_action(self, game: Game, state: State) -> Action:
        search = MCTS(game, evaluate=self._evaluator, root_noise=None)
        search.run(self._sims, state)
        return search.best_action()


@dataclass(frozen=True)
class Predicate:
    """One pre-registered pass/fail predicate and the numbers it consumed.

    Attributes:
        name: Predicate id (:data:`STRENGTH`, :data:`POLICY_LOSS`,
            :data:`VALUE_LOSS`).
        passed: Whether the predicate holds.
        measured: The statistic the predicate compares.
        threshold: The pinned bound it is compared against.
        comparator: ``">="`` or ``"<="`` — the direction, recorded rather than
            implied, so a reader of the verdict file needs no source access.
        detail: The inputs behind ``measured`` (window means, totals, ...).
    """

    name: str
    passed: bool
    measured: float
    threshold: float
    comparator: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the predicate in its persisted layout.

        Returns:
            A plain ``dict`` with the verdict, the statistic, the bound, the
            comparator, and the detail block.
        """
        return {
            "name": self.name,
            "verdict": "PASS" if self.passed else "FAIL",
            "measured": self.measured,
            "threshold": self.threshold,
            "comparator": self.comparator,
            "detail": dict(self.detail),
        }

    def render(self) -> str:
        """Return the predicate as one line of the printed table.

        Returns:
            ``"PASS  name  measured <= threshold  (detail...)"``.
        """
        detail = "  ".join(f"{k}={_fmt(v)}" for k, v in self.detail.items())
        return (
            f"{'PASS' if self.passed else 'FAIL'}  {self.name:<11} "
            f"{self.measured:.4f} {self.comparator} {self.threshold:.4f}   {detail}"
        )


def _fmt(value: Any) -> str:
    """Format one detail value for the printed table.

    Args:
        value: A detail entry.

    Returns:
        Four-decimal text for floats, ``str`` otherwise.
    """
    return f"{value:.4f}" if isinstance(value, float) else str(value)


@dataclass(frozen=True)
class Verdict:
    """The gate's complete, persistable outcome.

    Attributes:
        passed: PASS iff every predicate holds (the §12 M2.5 conjunction).
        predicates: The three predicates, in protocol order.
        inputs: Every input the verdict rests on — config, seeds, checkpoint and
            game identity, and the full evaluation-protocol scalars.
    """

    passed: bool
    predicates: tuple[Predicate, ...]
    inputs: dict[str, Any]

    def predicate(self, name: str) -> Predicate:
        """Return one predicate by name.

        Args:
            name: Predicate id.

        Returns:
            The matching :class:`Predicate`.

        Raises:
            KeyError: If no predicate carries that name.
        """
        for predicate in self.predicates:
            if predicate.name == name:
                return predicate
        raise KeyError(f"no predicate named {name!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return the verdict in its persisted JSON layout.

        Returns:
            A plain nested ``dict``, schema tag first.
        """
        return {
            "schema": VERDICT_SCHEMA,
            "verdict": "PASS" if self.passed else "FAIL",
            "predicates": [p.to_dict() for p in self.predicates],
            "inputs": self.inputs,
        }

    def write(self, path: Path | str) -> Path:
        """Write the verdict as JSON, creating parent directories.

        Args:
            path: Destination file.

        Returns:
            The path written.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return out

    def render(self) -> str:
        """Return the human-readable verdict block.

        Returns:
            The header lines, one line per predicate, and the overall verdict.
        """
        inputs = self.inputs
        evaluation = inputs["evaluation"]
        lines = [
            f"M2.5 exit gate — {inputs['run_name']} (run seed {inputs['run_seed']})",
            f"  run record:  {inputs['run_record']}",
            f"  checkpoint:  {inputs['checkpoint']['path']} "
            f"({inputs['checkpoint']['kind']}, step {inputs['checkpoint']['step']})",
            f"  identity:    {inputs['game_identity']['game']}/"
            f"{inputs['game_identity']['game_config']} "
            f"orientation {inputs['game_identity']['orientation_hash']}",
            f"  evaluation:  {evaluation['agent_form']} @{evaluation['sims']} sims, "
            f"noise {'on' if evaluation['root_noise'] else 'off'}, "
            f"{evaluation['move_selection']} vs {evaluation['opponent']}, "
            f"{evaluation['n_pairs']} pairs @ seed {evaluation['eval_seed']}",
        ]
        lines += [p.render() for p in self.predicates]
        lines.append(f"VERDICT: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def combine(predicates: Sequence[Predicate], inputs: dict[str, Any]) -> Verdict:
    """Combine the evaluated predicates into the overall verdict.

    The §12 M2.5 rule, in one place: **PASS iff the conjunction holds** — every
    predicate, no weighting, no "two out of three".

    Args:
        predicates: The evaluated predicates, in protocol order.
        inputs: The inputs the verdict rests on.

    Returns:
        The :class:`Verdict`.

    Raises:
        ValueError: If no predicates were evaluated — an empty conjunction is
            vacuously true, which is the one way a gate must never pass.
    """
    if not predicates:
        raise ValueError("no predicates evaluated; an empty gate cannot return PASS")
    return Verdict(
        passed=all(p.passed for p in predicates),
        predicates=tuple(predicates),
        inputs=inputs,
    )


def loss_series(record: dict[str, Any], name: str) -> list[float]:
    """Read one loss component out of a persisted run record, in step order.

    The file-side mirror of ``core.selfplay.RunRecord.loss_series``: the gate's
    predicates are defined over what was *written*, so they are computed from
    the parsed record and never from a live loop.

    Args:
        record: The parsed run record (``core.selfplay.load_run_record``).
        name: ``"policy_loss"``, ``"value_loss"``, ``"aux_loss"`` or
            ``"total_loss"``.

    Returns:
        The per-step values, in recorded step order.

    Raises:
        KeyError: If ``name`` is not a recorded loss component.
        ValueError: If the record carries no ``steps`` list, a step is missing
            the component, or a value is missing/non-numeric — a hole in the
            evidence must fail loudly, never average to something plausible.
    """
    if name not in ("policy_loss", "value_loss", "aux_loss", "total_loss"):
        raise KeyError(f"unknown loss component {name!r}")
    steps = record.get("steps")
    if not isinstance(steps, list):
        raise ValueError("run record carries no 'steps' list")
    series = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or name not in step:
            raise ValueError(f"run record step {i} has no {name!r} entry")
        value = step[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"run record step {i} has a non-numeric {name} ({value!r})")
        series.append(float(value))
    return series


def window_means(
    series: Sequence[float], head_window_steps: int, tail_window_steps: int
) -> tuple[float, float]:
    """Return the (head mean, tail mean) of a loss series over the pinned windows.

    The windows must be **disjoint**: a record shorter than
    ``head + tail`` steps cannot support the pinned comparison, and overlapping
    windows would drag the ratio toward 1 (or, at full overlap, to exactly 1)
    instead of failing — a truncated run must be rejected, not scored.

    Args:
        series: The per-step loss values, in step order.
        head_window_steps: Length of the leading window.
        tail_window_steps: Length of the trailing window.

    Returns:
        ``(head_mean, tail_mean)``.

    Raises:
        ValueError: If either window is not positive, the series is shorter than
            the two windows combined, or the head mean is not positive (the
            ratio would be meaningless or undefined).
    """
    if head_window_steps <= 0 or tail_window_steps <= 0:
        raise ValueError(
            f"loss windows must be positive, got head={head_window_steps} tail={tail_window_steps}"
        )
    if len(series) < head_window_steps + tail_window_steps:
        raise ValueError(
            f"run record has {len(series)} recorded steps, fewer than the pinned "
            f"head+tail windows ({head_window_steps}+{tail_window_steps}); the exit "
            "predicate is not evaluable on a truncated run"
        )
    head = sum(series[:head_window_steps]) / head_window_steps
    tail = sum(series[-tail_window_steps:]) / tail_window_steps
    if not head > 0.0:
        raise ValueError(f"head-window mean is {head}; the loss ratio is not meaningful")
    return head, tail


def loss_predicate(
    record: dict[str, Any],
    name: str,
    head_window_steps: int,
    tail_window_steps: int,
    max_ratio: float,
) -> Predicate:
    """Evaluate one tail-vs-head loss predicate (§12 M2.5 predicates 2–3).

    The comparator is ``<=``: a ratio *exactly* at the pinned bound **passes**.

    Args:
        record: The parsed run record.
        name: The loss component (``"policy_loss"`` / ``"value_loss"``).
        head_window_steps: Length of the head window, in recorded steps.
        tail_window_steps: Length of the tail window, in recorded steps.
        max_ratio: The pinned bound on ``tail_mean / head_mean``.

    Returns:
        The evaluated :class:`Predicate`.

    Raises:
        KeyError: If ``name`` is not a recorded loss component.
        ValueError: If the record cannot support the pinned windows (see
            :func:`window_means` and :func:`loss_series`).
    """
    series = loss_series(record, name)
    head, tail = window_means(series, head_window_steps, tail_window_steps)
    ratio = tail / head
    return Predicate(
        name=name,
        passed=ratio <= max_ratio,
        measured=ratio,
        threshold=max_ratio,
        comparator="<=",
        detail={
            "head_mean": head,
            "tail_mean": tail,
            "head_window_steps": head_window_steps,
            "tail_window_steps": tail_window_steps,
            "recorded_steps": len(series),
        },
    )


def strength_predicate(pairs: Sequence[PairResult], min_score_rate: float) -> Predicate:
    """Evaluate the win-rate predicate (§12 M2.5 predicate 1).

    ``score_rate = total_score_a / (2 × n_pairs)`` with a draw counted 0.5 (the
    runner's ``PairResult.score_a`` already does), compared with ``>=``: a rate
    *exactly* at the pinned floor **passes**.

    Args:
        pairs: The mirrored-pair results, trained side as agent A.
        min_score_rate: The pinned floor.

    Returns:
        The evaluated :class:`Predicate`.

    Raises:
        ValueError: If ``pairs`` is empty — an empty match must not report a
            score rate at all.
    """
    if not pairs:
        raise ValueError("no pairs played; the strength predicate has no evidence")
    total = sum(p.score_a for p in pairs)
    games = 2 * len(pairs)
    rate = total / games
    return Predicate(
        name=STRENGTH,
        passed=rate >= min_score_rate,
        measured=rate,
        threshold=min_score_rate,
        comparator=">=",
        detail={"total_score": total, "games": games, "pairs": len(pairs)},
    )


def check_identity(expected: dict[str, str], found: dict[str, Any], source: str) -> None:
    """Assert one artifact's game identity matches the config's (Invariant 4).

    Args:
        expected: The identity re-derived from the run config —
            ``{"game", "game_config", "orientation_hash"}``.
        found: The artifact's identity block.
        source: Where ``found`` came from, for the error message.

    Raises:
        ValueError: If any field is missing or disagrees. Scoring a checkpoint
            built from a different instance (or a different orientation table)
            would produce a number that looks fine and means nothing.
    """
    for key, want in expected.items():
        got = found.get(key)
        if got != want:
            raise ValueError(
                f"{source}: {key} is {got!r}, but the run config implies {want!r}; "
                "refusing to evaluate the gate against a mismatched instance"
            )


def expected_identity(cfg: RunConfig) -> dict[str, str]:
    """Return the game identity a run of ``cfg`` must carry.

    Args:
        cfg: The run config.

    Returns:
        ``{"game", "game_config", "orientation_hash"}``, the hash re-derived
        from the instance config rather than trusted from the artifacts.

    Raises:
        ValueError: If the config names a game this gate cannot construct.
    """
    if cfg.game != "blokus_duo":
        raise ValueError(f"scripts/micro_exit_gate.py drives blokus_duo only; got {cfg.game!r}")
    return {
        "game": cfg.game,
        "game_config": cfg.game_config,
        "orientation_hash": orientation_table_hash(cfg.resolve_game_config()),
    }


def select_checkpoint(record: dict[str, Any], kind: str, run_dir: Path) -> tuple[Path, dict]:
    """Return the checkpoint the config selects (``training.checkpoint_selection``).

    Args:
        record: The parsed run record.
        kind: The checkpoint kind to select (M2.5 pins ``"final"``).
        run_dir: The run directory, used to relocate a checkpoint whose recorded
            absolute path does not exist (the run may have been produced on
            another machine and the directory copied).

    Returns:
        ``(path, entry)`` — the resolved file and its run-record entry.

    Raises:
        ValueError: If the record carries no checkpoints, or not exactly one of
            the requested kind.
        FileNotFoundError: If the selected checkpoint file cannot be found.
    """
    entries = [c for c in record.get("checkpoints", []) if c.get("kind") == kind]
    if len(entries) != 1:
        raise ValueError(
            f"run record carries {len(entries)} {kind!r} checkpoint(s); the gate evaluates "
            "exactly one"
        )
    entry = entries[0]
    path = Path(entry["path"])
    if not path.exists():
        fallback = run_dir / path.name
        if not fallback.exists():
            raise FileNotFoundError(
                f"checkpoint {path} does not exist (nor {fallback}); the gate needs the "
                "run's own weights"
            )
        path = fallback
    return path, entry


def load_checkpoint(path: Path, cfg: RunConfig, identity: dict[str, str]) -> dict[str, Any]:
    """Load and validate the evaluated checkpoint.

    Args:
        path: The checkpoint file.
        cfg: The run config the record carried.
        identity: The expected game identity (:func:`expected_identity`).

    Returns:
        The parsed checkpoint blob.

    Raises:
        ValueError: If the schema tag is unknown, the identity disagrees with
            the config's, or the checkpoint's own run seed / config disagree
            with the record's — all of which mean the weights did not come from
            the run whose losses are being scored.
    """
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"{path}: unknown checkpoint schema {blob.get('schema')!r} "
            f"(expected {CHECKPOINT_SCHEMA!r})"
        )
    check_identity(identity, blob, f"checkpoint {path}")
    if blob.get("run_seed") != cfg.run_seed:
        raise ValueError(
            f"{path}: checkpoint run_seed {blob.get('run_seed')!r} does not match the "
            f"run record's {cfg.run_seed!r}"
        )
    if blob.get("config") != cfg.to_dict():
        raise ValueError(
            f"{path}: the checkpoint's embedded config differs from the run record's; "
            "the weights and the loss series come from different protocols"
        )
    return blob


def build_network(blob: dict[str, Any], device: str) -> Network:
    """Rebuild the trained network from a checkpoint blob.

    Args:
        blob: The parsed checkpoint.
        device: Torch device string.

    Returns:
        The network in ``eval`` mode on ``device``, weights loaded strictly.

    Raises:
        ValueError: If the blob carries no network config or weights.
        RuntimeError: Propagated from ``load_state_dict`` if the weights do not
            fit the declared architecture.
    """
    raw = blob.get("network_config")
    state = blob.get("model_state_dict")
    if not raw or not state:
        raise ValueError("checkpoint carries no network_config/model_state_dict")
    net_cfg = NetworkConfig(
        input_planes=int(raw["input_planes"]),
        input_shape=tuple(raw["input_shape"]),
        policy_shape=tuple(raw["policy_shape"]),
        trunk_blocks=int(raw["trunk_blocks"]),
        trunk_channels=int(raw["trunk_channels"]),
        num_aux=int(raw["num_aux"]),
    )
    net = Network(net_cfg)
    net.load_state_dict(state)
    return net.to(torch.device(device)).eval()


def check_protocol(cfg: RunConfig) -> None:
    """Assert the config's evaluation protocol is the one this gate realizes.

    Args:
        cfg: The run config.

    Raises:
        ValueError: If the agent form, opponent, move rule, or root-noise flag
            is not the §12 M2.5 pin. Substituting a different agent form would
            silently answer a different question.
    """
    evaluation = cfg.evaluation
    if evaluation.agent_form != SUPPORTED_AGENT_FORM:
        raise ValueError(
            f"evaluation.agent_form is {evaluation.agent_form!r}; this gate realizes "
            f"{SUPPORTED_AGENT_FORM!r} only"
        )
    if evaluation.opponent != SUPPORTED_OPPONENT:
        raise ValueError(
            f"evaluation.opponent is {evaluation.opponent!r}; this gate realizes "
            f"{SUPPORTED_OPPONENT!r} only"
        )
    if evaluation.move_selection != SUPPORTED_MOVE_SELECTION:
        raise ValueError(
            f"evaluation.move_selection is {evaluation.move_selection!r}; the pinned "
            f"evaluation form is {SUPPORTED_MOVE_SELECTION!r}"
        )
    if evaluation.root_noise:
        raise ValueError("evaluation.root_noise is true; D7 noise is self-play-only (§12 M2.5)")


def play_evaluation_set(
    game: Game, net: Network, cfg: RunConfig, device: str = "cpu"
) -> list[PairResult]:
    """Play the fixed paired set the strength predicate is scored on.

    The M1.6 machinery, unchanged: ``core.runner.play_pairs`` for mirrored pairs
    with seats swapped and per-pair seeds, and the Blokus start-square balancer
    (now keyed on the *configured* start squares) so both games of a pair open on
    the same square. The trained side is agent **A**, so ``PairResult.score_a``
    is its score.

    Args:
        game: The micro adapter.
        net: The trained network.
        cfg: The run config (evaluation scalars).
        device: Torch device string for leaf inference.

    Returns:
        One :class:`~core.runner.PairResult` per pair, in play order.
    """
    evaluator = make_network_evaluator(net, game, device=device)
    sims = cfg.evaluation.sims
    return play_pairs(
        game,
        lambda seed: NetworkMCTSAgent(evaluator, sims, seed),
        lambda seed: RandomAgent(seed),
        n_pairs=cfg.evaluation.n_pairs,
        seed=cfg.evaluation.eval_seed,
        opening_balancer=start_square_balancer,
    )


def run_gate(
    run_dir: Path | str,
    *,
    config_path: Path | str = MICRO_RUN_CONFIG_PATH,
    device: str = "cpu",
) -> Verdict:
    """Evaluate the pre-registered gate against a completed run.

    Order matters: identity and protocol are validated **before** a single game
    is played, so a mismatched checkpoint costs a diagnostic rather than a
    meaningless score.

    Args:
        run_dir: The run directory holding ``run_record.json`` and the
            checkpoints.
        config_path: The pre-registered config file the record's embedded config
            must equal — the pinned ``configs/blokus_micro.json`` by default.
        device: Torch device string for leaf inference.

    Returns:
        The evaluated :class:`Verdict` (not yet written).

    Raises:
        FileNotFoundError: If the run record or the selected checkpoint is
            missing.
        ValueError: If the record's schema/config, the game identity, the
            evaluation protocol, or the recorded loss series cannot support the
            pinned gate.
    """
    run_path = Path(run_dir)
    record_path = run_path / RUN_RECORD_NAME
    record = load_run_record(record_path)

    cfg = RunConfig.from_dict(record["config"])
    pinned = load_run_config(config_path)
    if cfg != pinned:
        raise ValueError(
            f"the run record's config does not match the pre-registered {Path(config_path)}; "
            "the gate scores a run only against the protocol it was registered under"
        )
    check_protocol(cfg)

    identity = expected_identity(cfg)
    check_identity(identity, record.get("game_identity", {}), f"run record {record_path}")
    checkpoint_path, entry = select_checkpoint(record, cfg.training.checkpoint_selection, run_path)
    blob = load_checkpoint(checkpoint_path, cfg, identity)

    # Losses first: they need no GPU, no games, and a truncated record should
    # fail before the expensive half runs.
    predicates = [
        loss_predicate(
            record,
            POLICY_LOSS,
            cfg.loss_predicates.head_window_steps,
            cfg.loss_predicates.tail_window_steps,
            cfg.loss_predicates.policy_max_ratio,
        ),
        loss_predicate(
            record,
            VALUE_LOSS,
            cfg.loss_predicates.head_window_steps,
            cfg.loss_predicates.tail_window_steps,
            cfg.loss_predicates.value_max_ratio,
        ),
    ]

    game = BlokusDuo(config=cfg.resolve_game_config())
    net = build_network(blob, device)
    pairs = play_evaluation_set(game, net, cfg, device=device)
    # The M1.6 anchored-Elo scaffolding, with rung 1 pinned at 0. Observational:
    # the predicate is the score rate, not the rating.
    ratings = fit_elo(
        matches_from_pairs(cfg.evaluation.agent_form, cfg.evaluation.opponent, pairs),
        anchor=cfg.evaluation.opponent,
    )
    strength = strength_predicate(pairs, cfg.evaluation.min_score_rate)
    strength = replace(
        strength,
        detail={**strength.detail, "elo_vs_anchor": ratings[cfg.evaluation.agent_form]},
    )
    predicates.insert(0, strength)

    inputs = {
        "run_name": cfg.name,
        "run_seed": cfg.run_seed,
        "run_dir": str(run_path),
        "run_record": str(record_path),
        "config_path": str(Path(config_path)),
        "config": cfg.to_dict(),
        "game_identity": identity,
        "device": device,
        "checkpoint": {
            "path": str(checkpoint_path),
            "kind": entry["kind"],
            "step": entry["step"],
            "schema": blob["schema"],
        },
        "evaluation": {
            "agent_form": cfg.evaluation.agent_form,
            "sims": cfg.evaluation.sims,
            "root_noise": cfg.evaluation.root_noise,
            "move_selection": cfg.evaluation.move_selection,
            "opponent": cfg.evaluation.opponent,
            "n_pairs": cfg.evaluation.n_pairs,
            "eval_seed": cfg.evaluation.eval_seed,
            "min_score_rate": cfg.evaluation.min_score_rate,
        },
        "loss_predicates": {
            "head_window_steps": cfg.loss_predicates.head_window_steps,
            "tail_window_steps": cfg.loss_predicates.tail_window_steps,
            "policy_max_ratio": cfg.loss_predicates.policy_max_ratio,
            "value_max_ratio": cfg.loss_predicates.value_max_ratio,
        },
        "measurements": {
            "pair_scores": [p.score_a for p in pairs],
            "elo": ratings,
        },
    }
    return combine(predicates, inputs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the gate's command line.

    Args:
        argv: Argument list; ``None`` reads ``sys.argv``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the pre-registered M2.5 exit gate against a completed run."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=MICRO_RUN_CONFIG_PATH,
        help="pre-registered run config the record must match "
        "(default: the pinned configs/blokus_micro.json)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="run directory holding run_record.json (default: the config's run_dir)",
    )
    parser.add_argument("--device", default="cpu", help="torch device for leaf inference")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"verdict file to write (default: <run-dir>/{VERDICT_NAME})",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the printed verdict block")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Evaluate the gate, persist the verdict, and report it.

    Args:
        argv: Argument list; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code: ``0`` PASS, ``1`` FAIL, ``2`` the gate could not be
        evaluated (missing/mismatched artifacts, truncated record) — a gate must
        never exit 0 on evidence it could not check.
    """
    args = parse_args(argv)
    try:
        cfg = load_run_config(args.config)
        run_dir = args.run_dir if args.run_dir is not None else ROOT / cfg.run_dir
        verdict = run_gate(run_dir, config_path=args.config, device=args.device)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"exit gate could not be evaluated: {exc}", file=sys.stderr)
        return 2
    out = args.out if args.out is not None else Path(run_dir) / VERDICT_NAME
    verdict.write(out)
    if not args.quiet:
        print(verdict.render())
        print(f"verdict record: {out}")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
