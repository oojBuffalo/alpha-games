"""Inference mixed precision as a run policy (D5 AMP; design doc §7, §12 M2.5/M5).

Self-play leaf inference runs under the same CUDA-only autocast rule the train
step follows (``core.train.make_scaler`` / ``core.train.train_step``): live on
CUDA, an exact no-op everywhere else. The rule is *derived from the device* --
:meth:`AmpMode.resolve` is the one place it is decided -- and is deliberately
not a config field: precision is how a run executes on its hardware, not part
of the protocol it records.

Two pieces:

* :class:`AmpMode` -- the resolved setting. One instance opens every autocast
  context of a run *and* answers "was autocast actually live?" by reading
  torch's own state from inside that context (:meth:`AmpMode.observed`), so a
  recorded AMP flag is a measurement, never a claim re-derived from the device.
* :func:`amp_evaluator` -- wraps a leaf evaluator so its forward runs inside
  the context. Applied at the call site (``core.ipc`` for the production
  actor, the micro scripts for theirs) rather than inside
  ``core.network``'s evaluator factories, whose other callers (evaluation)
  must not change precision as a side effect.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from core.game import Game


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
    def resolve(cls, device: torch.device | str) -> AmpMode:
        """Resolve AMP for a device — the one place the on/off rule is decided.

        Args:
            device: The device the run's forwards execute on, as a
                ``torch.device`` or a torch device string (``"cuda:0"``).

        Returns:
            The mode: enabled on CUDA, disabled everywhere else.
        """
        device_type = torch.device(device).type
        return cls(device_type=device_type, enabled=device_type == "cuda")

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


def amp_evaluator(
    evaluator: Callable[[Game, Any], Any],
    amp: AmpMode,
    *,
    observe: Callable[[bool], None] | None = None,
) -> Callable[[Game, Any], Any]:
    """Wrap a leaf evaluator so its forward runs under the run's AMP setting.

    Applied at the call site rather than inside ``core.network``'s evaluator
    factories: precision is a *run* policy, not a property of the net→MCTS
    bridge, and ``core/`` has other callers whose behaviour must not move.
    Autocast is thread-local dispatch state, so the inner evaluator's
    ``torch.inference_mode`` forward picks it up unchanged — inference-only, no
    ``GradScaler`` anywhere on this path.

    The wrapper forwards its second argument untouched, so it serves both
    engine seams: ``core.mcts.Evaluator`` (one state) and
    ``core.mcts.BatchEvaluator`` (a sequence of states).

    Args:
        evaluator: The evaluator to wrap (``core.network.make_network_evaluator``
            or ``make_batched_network_evaluator``'s result, possibly already
            wrapped by a counter — anything wrapped here runs inside the
            context).
        amp: The run's resolved AMP mode.
        observe: Optional. Called exactly once, on the wrapper's first call,
            from *inside* the entered context and before the forward, with
            :meth:`AmpMode.observed` — the live state the forward is about to
            run under, for the caller to record.

    Returns:
        A callable with ``evaluator``'s signature and identical semantics,
        evaluated under ``amp``.
    """
    pending = observe

    def evaluate(game: Game, state: Any) -> Any:
        """Evaluate inside the run's autocast context.

        Args:
            game: The adapter, passed through untouched.
            state: The leaf state (or the batch's states), passed through.

        Returns:
            The wrapped evaluator's result.
        """
        nonlocal pending
        with amp.autocast():
            if pending is not None:
                pending, report = None, pending
                report(amp.observed())
            return evaluator(game, state)

    return evaluate
