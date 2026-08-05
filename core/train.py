"""Collate boundary, D5 optimizer recipe, and the AMP train step (§7, §12 M2).

Torch and NumPy live here (and in ``core/network.py`` / ``core/losses.py``)
only — the pyproject confinement pin; adapters and the rest of ``core/`` stay
stdlib-pure. This module is deliberately not exported from ``core/__init__``
so that ``import core`` never pulls torch.

``collate`` is the single point where stdlib data becomes tensors: everything
upstream (adapters, self-play, replay) speaks nested tuples and sparse D12
``(action_id, visit_count)`` pairs; everything downstream speaks tensors. Aux
handling is spec-driven — a game whose ``ValueTargetSpec`` declares no aux
heads (Othello) carries no aux slot in its samples and collate emits no aux
tensors, not zero-filled placeholders.

The optimizer recipe is D5: SGD momentum 0.9, weight decay 1e-4 (the §7
``c‖θ‖²`` term — it lives in the optimizer and only there), base LR 0.02
under linear warmup + cosine decay. Warmup length and total steps are config
arguments: M2 pins only the schedule's shape; M3 pins the run-length numbers.
``train_step`` runs mixed precision (autocast + GradScaler) on CUDA and
degrades both to exact no-ops elsewhere, so the battery runs GPU-free (CI has
no GPU); AMP is exercised for real by the M2 GPU benchmark on the 4060 Ti.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from core.game import Game, assert_v1_envelope
from core.losses import LossBreakdown, composite_loss, pad_sparse_targets, sparse_policy_loss

# torch.amp.GradScaler is the 2.3+ AMP surface (2.2's torch.amp exposes
# autocast only) — this matches the pyproject lower bound; fail at import
# with a clear message rather than an AttributeError mid-train-step on an
# out-of-date install.
if not hasattr(torch.amp, "GradScaler"):
    raise ImportError(
        f"core.train requires torch >= 2.3 (torch.amp.GradScaler); found {torch.__version__}"
    )

# D5 default batch size (§7: "Batch 256 (benchmark 128/256/512)") — an explicit
# constant the M3 loop consumes, not an implicit property of whatever batch a
# caller passes. ``train_step`` itself is batch-size-agnostic: the CPU battery
# uses small synthetic batches, and the 128/256/512 sweep is the M2 GPU
# benchmark's job.
BATCH_SIZE = 256

# D5 base learning rate (§7: "LR ≈ 0.02 warmup + cosine"); the schedule from
# ``make_lr_scheduler`` multiplies it.
LEARNING_RATE = 0.02


@dataclass(frozen=True)
class Batch:
    """One collated training batch — the tensor side of the D12 sample shape.

    Attributes:
        planes: ``(N, input_planes, H, W)`` float32 state planes.
        legal_ids: ``(N, L)`` int64 legal action ids, ragged rows padded with
            ``core.losses.PAD_ID``.
        visit_counts: ``(N, L)`` float32 root visit counts, zero in pad slots.
        z: ``(N,)`` float32 game-outcome targets (D1).
        aux: ``(N, num_aux)`` float32 aux targets, or ``None`` for a game
            whose spec declares no aux heads — absent, never zero-filled.
        aux_weights: The spec's ``aux_loss_weights``, carried so the batch is
            self-describing: ``train_step`` reads them here rather than
            re-querying the game.
    """

    planes: torch.Tensor
    legal_ids: torch.Tensor
    visit_counts: torch.Tensor
    z: torch.Tensor
    aux: torch.Tensor | None
    aux_weights: tuple[float, ...]

    def to(self, device: Any) -> Batch:
        """Return a copy of the batch with every tensor moved to ``device``.

        Args:
            device: Target device, as accepted by ``torch.Tensor.to`` (e.g.
                ``"cuda"`` for the GPU benchmark path).

        Returns:
            A new ``Batch`` on ``device``; ``aux_weights`` carries over as-is.
        """
        return Batch(
            planes=self.planes.to(device),
            legal_ids=self.legal_ids.to(device),
            visit_counts=self.visit_counts.to(device),
            z=self.z.to(device),
            aux=None if self.aux is None else self.aux.to(device),
            aux_weights=self.aux_weights,
        )


def collate(game: Game, samples: Sequence[Sequence[Any]]) -> Batch:
    """Collate stdlib-encoded samples into the tensor batch the loss consumes.

    The single stdlib→tensor conversion point (§12 M2): planes go through
    ``numpy.asarray`` → ``torch.as_tensor``; sparse policy targets go through
    ``core.losses.pad_sparse_targets`` into the padded legal-id/count/mask
    contract of ``sparse_policy_loss``. Sample arity is spec-driven: a game
    with declared aux heads carries a fourth slot of per-head targets, a
    no-aux game (Othello) carries none — either way collate emits exactly
    what the spec declares.

    Args:
        game: Adapter declaring the encoding surface and ``value_targets``
            (§6.1); the stacked planes are validated against its declared
            ``input_planes``/``input_shape``.
        samples: Per-sample tuples ``(planes, sparse_pi, z, aux)`` when the
            spec declares aux heads, ``(planes, sparse_pi, z)`` otherwise.
            ``planes`` are the nested tuples from ``encode_state``;
            ``sparse_pi`` is the full legal set as D12 ``(action_id,
            visit_count)`` pairs; ``z`` is the D1 scalar outcome; ``aux`` is
            a sequence of per-head targets parallel to the spec's
            ``aux_names`` (Blokus: ``(score_diff/109,)``).

    Returns:
        The collated ``Batch``; ``aux`` is ``None`` iff the spec declares no
        aux heads.

    Raises:
        EnvelopeError: If ``game`` breaches the v1 engine envelope (§2 —
            "asserted in code, not just prose"): collate is a core boundary
            and asserts it exactly as the search engine does.
        ValueError: If the batch is empty, a sample's arity disagrees with
            the spec (an aux slot on a no-aux game, or a missing or mis-sized
            one on an aux game), the planes cannot be stacked, or the stacked
            shape disagrees with the game's declared
            ``(input_planes, *input_shape)``.
    """
    assert_v1_envelope(game)
    spec = game.value_targets
    num_aux = len(spec.aux_names)
    rows = list(samples)
    if not rows:
        raise ValueError("empty batch of samples")
    arity = 4 if num_aux else 3
    if any(len(row) != arity for row in rows):
        slots = "(planes, sparse_pi, z, aux)" if num_aux else "(planes, sparse_pi, z)"
        raise ValueError(
            f"{type(game).__name__} declares {num_aux} aux head(s): every sample must be {slots}"
        )
    planes = torch.as_tensor(np.asarray([row[0] for row in rows], dtype=np.float32))
    expected = (len(rows), game.input_planes, *game.input_shape)
    if tuple(planes.shape) != expected:
        raise ValueError(f"stacked planes shape {tuple(planes.shape)} != declared {expected}")
    legal_ids, visit_counts = pad_sparse_targets([row[1] for row in rows])
    z = torch.as_tensor([float(row[2]) for row in rows], dtype=torch.float32)
    if num_aux == 0:
        aux = None
    else:
        try:
            aux_rows = [tuple(float(v) for v in row[3]) for row in rows]
        except TypeError as err:
            raise ValueError(
                f"aux must be a sequence of per-head targets parallel to {spec.aux_names}"
            ) from err
        if any(len(row) != num_aux for row in aux_rows):
            raise ValueError(f"an aux row is not {num_aux} wide (spec aux_names {spec.aux_names})")
        aux = torch.as_tensor(aux_rows, dtype=torch.float32)
    return Batch(
        planes=planes,
        legal_ids=legal_ids,
        visit_counts=visit_counts,
        z=z,
        aux=aux,
        aux_weights=tuple(spec.aux_loss_weights),
    )


def make_optimizer(net: torch.nn.Module, lr: float = LEARNING_RATE) -> torch.optim.SGD:
    """Build the D5 optimizer: SGD, momentum 0.9, weight decay 1e-4 (§7).

    The weight-decay term realizes the ``c‖θ‖²`` of the §7 composite loss —
    here and only here; ``core.losses.composite_loss`` deliberately does not
    compute it (adding it there as well would double-count it).

    Args:
        net: The network whose parameters to optimize.
        lr: Base learning rate (D5: 0.02); the warmup+cosine schedule
            multiplies it.

    Returns:
        The configured ``torch.optim.SGD``.
    """
    return torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)


def make_lr_lambda(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    """Build the D5 LR-shape multiplier: linear warmup, then cosine decay to 0.

    The multiplier climbs linearly from 0 toward 1 over ``warmup_steps``
    steps — exactly 1.0 *at* step ``warmup_steps`` — then follows a half
    cosine down to exactly 0.0 at ``total_steps``, staying 0 beyond. M2 pins
    only this shape; the run-length numbers are pinned at M3.

    Args:
        warmup_steps: Steps of linear warmup; 0 starts at the cosine peak.
        total_steps: Step at which the cosine reaches 0; must exceed
            ``warmup_steps``.

    Returns:
        ``step → multiplier`` in ``[0, 1]``, for ``LambdaLR`` or direct use.

    Raises:
        ValueError: If ``warmup_steps`` is negative or ``total_steps`` does
            not exceed it.
    """
    if warmup_steps < 0 or total_steps <= warmup_steps:
        raise ValueError(
            f"need 0 <= warmup_steps < total_steps, got {warmup_steps=}, {total_steps=}"
        )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        if step >= total_steps:
            return 0.0
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Attach the D5 warmup+cosine schedule to an optimizer.

    Args:
        optimizer: The optimizer whose LR to schedule (``make_optimizer``).
        warmup_steps: See ``make_lr_lambda``.
        total_steps: See ``make_lr_lambda``.

    Returns:
        A ``LambdaLR`` applying the shape multiplier to the base LR; call
        ``.step()`` once per training step.

    Raises:
        ValueError: Propagated from ``make_lr_lambda`` on a bad shape config.
    """
    return torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(warmup_steps, total_steps))


def make_scaler(device_type: str = "cpu") -> torch.amp.GradScaler:
    """Build the AMP loss scaler, enabled on CUDA only (D5 mixed precision).

    A disabled ``GradScaler`` is an exact pass-through — ``scale`` returns the
    loss unchanged and ``step``/``update`` reduce to a plain
    ``optimizer.step()`` — so the one ``train_step`` code path runs GPU-free
    on CPU (CI) and AMP-for-real on the 4060 Ti.

    Args:
        device_type: ``"cuda"`` for real loss scaling; anything else yields a
            disabled scaler.

    Returns:
        The (possibly disabled) ``torch.amp.GradScaler``.
    """
    enabled = device_type == "cuda"
    return torch.amp.GradScaler("cuda" if enabled else "cpu", enabled=enabled)


def train_step(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: Batch,
) -> LossBreakdown:
    """Run one optimization step of the §7 composite loss under AMP.

    Forward and loss run under ``autocast`` — float16 on CUDA, an exact no-op
    elsewhere, so the CPU battery computes plain float32 — and backward/step
    go through ``scaler`` (identity when disabled, ``make_scaler``). The
    device is read from the batch; the caller keeps net and batch co-located
    (``Batch.to``). Aux plumbing is spec-driven end to end: the batch carries
    its spec's ``aux_weights``, and ``composite_loss`` rejects any
    net-vs-batch aux disagreement loudly.

    Args:
        net: The D5 network, on the batch's device, in train mode.
        optimizer: The D5 optimizer (``make_optimizer``).
        scaler: The AMP scaler (``make_scaler``); disabled off-CUDA.
        batch: A collated ``Batch`` (``collate``).

    Returns:
        The detached ``LossBreakdown`` — per-component losses for
        observability (§12 M3 counters), holding no graph references.

    Raises:
        ValueError: If the net's aux output disagrees with the batch's aux
            spec (e.g. an aux-emitting net on a no-aux batch), from
            ``composite_loss``.
    """
    device_type = batch.planes.device.type
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device_type, enabled=device_type == "cuda"):
        logits, value, aux_pred = net(batch.planes)
        policy = sparse_policy_loss(logits, batch.legal_ids, batch.visit_counts)
        parts = composite_loss(policy, value, batch.z, aux_pred, batch.aux, batch.aux_weights)
    scaler.scale(parts.total).backward()
    scaler.step(optimizer)
    scaler.update()
    return LossBreakdown(
        total=parts.total.detach(),
        value=parts.value.detach(),
        policy=parts.policy.detach(),
        aux=None if parts.aux is None else parts.aux.detach(),
    )
