"""Sparse policy cross-entropy and the §7 composite loss (§12 M2).

Torch lives here (and in ``core/network.py`` / ``core/train.py``) only — the
pyproject confinement pin; adapters and the rest of ``core/`` stay stdlib-pure.
This module is deliberately not exported from ``core/__init__`` so that
``import core`` never pulls torch.

The policy loss is the sparse gather §5.1 anticipates: nearly all of the
17,836 raw logits are illegal, so the cross-entropy gathers each sample's
legal-action logits from the flat head and renormalizes (``log_softmax``)
over the legal set only — masking + renormalization is load-bearing. Targets
are ``π_train(a) = N(a)/ΣN`` (D10), consumed directly from the D12 replay
shape — sparse ``(action_id, visit_count)`` pairs — with no dense policy
vector materialized anywhere (Invariant 3: sparse everywhere).

Batching pads ragged legal sets to ``(N, L)`` with ``PAD_ID = -1`` ids and
zero counts. Pad slots get an additive ``-inf`` mask before the softmax, and
their log-probabilities are zeroed *after* it — the output is masked, never a
``0 · (-inf)`` product or a difference of two ``-inf``s — so a pad slot
contributes exactly zero to the loss and exactly zero gradient.

Validation ownership: ``pad_sparse_targets`` is the structural validator —
every malformed-sparse-row failure mode (duplicate ids, negative ids,
non-finite or negative counts, zero-total rows, empty rows/batches) is
rejected there, in Python on CPU at collate time. The loss bodies check only
tensor *metadata* (shapes) and perform no tensor-value tests, so the AMP hot
path never forces a device-to-host synchronization per training step.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

# Sentinel action id marking a pad slot in a padded legal-id tensor. The pad
# mask is exactly ``legal_ids == PAD_ID``; any other negative id is not
# treated as padding and fails the loss's gather loudly (out-of-range index)
# rather than being silently discarded.
PAD_ID = -1


def pad_sparse_targets(
    batch: Sequence[Sequence[tuple[int, int]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of sparse D12 policy targets into loss-ready tensors.

    This is the structural validation boundary (see the module docstring):
    every malformed-row failure mode is rejected here, in Python on CPU at
    collate time, so ``sparse_policy_loss`` can stay synchronization-free.

    Args:
        batch: Per-sample sequences of ``(action_id, visit_count)`` pairs —
            the D12 replay storage shape (§6.2). Each sample must list its
            full legal set: zero-count legal actions add nothing to the CE
            sum but do shape the legal-set renormalization.

    Returns:
        ``(legal_ids, visit_counts)``: ``(N, L)`` int64 action ids padded
        with ``PAD_ID`` and ``(N, L)`` float32 counts padded with zero,
        where ``L`` is the widest legal set in the batch.

    Raises:
        ValueError: If the batch is empty; a sample has no pairs (the pass
            invariant guarantees >= 1 legal action at every stored
            position); an action id is negative (indistinguishable from the
            pad sentinel) or duplicated within its sample (a duplicate would
            enter the softmax as two legal slots and skew the D10 target);
            or a visit count is non-finite, negative, or the sample's counts
            sum to zero (D10: ``π_train ∝ N`` needs a positive total).
    """
    rows = [tuple(sample) for sample in batch]
    if not rows:
        raise ValueError("empty batch of policy targets")
    for i, row in enumerate(rows):
        if not row:
            raise ValueError(f"sample {i} has no (action_id, visit_count) pairs (pass invariant)")
        ids = [action for action, _ in row]
        if any(action < 0 for action in ids):
            raise ValueError(
                f"sample {i}: negative action id collides with the pad sentinel {PAD_ID}"
            )
        if len(set(ids)) != len(ids):
            raise ValueError(f"sample {i}: duplicate action id in a sparse policy row")
        counts = [count for _, count in row]
        if any(not math.isfinite(count) or count < 0 for count in counts):
            raise ValueError(f"sample {i}: visit counts must be finite and nonnegative")
        if sum(counts) <= 0:
            raise ValueError(f"sample {i}: visit counts sum to zero (D10: π_train ∝ N)")
    width = max(len(row) for row in rows)
    legal_ids = torch.full((len(rows), width), PAD_ID, dtype=torch.int64)
    visit_counts = torch.zeros((len(rows), width), dtype=torch.float32)
    for i, row in enumerate(rows):
        legal_ids[i, : len(row)] = torch.tensor([a for a, _ in row], dtype=torch.int64)
        visit_counts[i, : len(row)] = torch.tensor([c for _, c in row], dtype=torch.float32)
    return legal_ids, visit_counts


def sparse_policy_loss(
    logits: torch.Tensor,
    legal_ids: torch.Tensor,
    visit_counts: torch.Tensor,
) -> torch.Tensor:
    """Legal-set-only policy cross-entropy over the flat head (§5.1, §7, D10).

    Gathers each sample's legal-action logits from the full head and applies
    ``log_softmax`` over the legal set only; the batch-mean cross-entropy is
    taken against ``π_train(a) = N(a)/ΣN``. Illegal logits are never touched:
    perturbing one changes neither the loss bits nor any gradient, and their
    own gradient is exactly zero. Pad slots are masked out of the softmax
    (additive ``-inf``) and out of the output (log-probs zeroed), so they
    contribute exactly zero — no ``0 · (-inf) = nan`` enters the graph.

    Structural row validity — unique nonnegative ids, finite nonnegative
    counts, >= 1 real slot and a positive count total per row — is the
    caller's contract, owned by ``pad_sparse_targets`` at collate time. This
    body checks tensor metadata only and runs no tensor-value test, so it
    never forces a device-to-host synchronization on the AMP hot path. A
    stray negative id other than ``PAD_ID`` is not treated as padding: it
    reaches the gather out of range and fails loudly.

    Args:
        logits: ``(N, num_actions)`` raw policy-head logits, unmasked.
        legal_ids: ``(N, L)`` int64 legal action ids, ragged rows padded
            with ``PAD_ID`` — from ``pad_sparse_targets``, which validated
            them.
        visit_counts: ``(N, L)`` root visit counts aligned with
            ``legal_ids``; values in pad slots are ignored.

    Returns:
        Scalar batch-mean cross-entropy ``-Σ_a π_train(a) · log p(a)``.

    Raises:
        ValueError: On shape disagreement, an empty batch, or zero-width
            targets (either would otherwise reduce to ``nan``).
        RuntimeError: From the gather, if an id is out of range for the
            head — including any negative id that is not ``PAD_ID``.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be (N, num_actions), got {tuple(logits.shape)}")
    if legal_ids.shape != visit_counts.shape or legal_ids.dim() != 2:
        raise ValueError(
            f"legal_ids {tuple(legal_ids.shape)} and visit_counts "
            f"{tuple(visit_counts.shape)} must be equal-shaped (N, L) tensors"
        )
    if legal_ids.shape[0] != logits.shape[0]:
        raise ValueError(
            f"batch mismatch: {logits.shape[0]} logit rows, {legal_ids.shape[0]} target rows"
        )
    if logits.shape[0] == 0:
        raise ValueError("empty batch: the batch-mean policy loss is undefined")
    if legal_ids.shape[1] == 0:
        raise ValueError("zero-width targets: every sample needs >= 1 legal slot")
    pad = legal_ids.eq(PAD_ID)
    counts = visit_counts.to(logits.dtype).masked_fill(pad, 0.0)
    totals = counts.sum(dim=1, keepdim=True)
    gathered = logits.gather(1, legal_ids.masked_fill(pad, 0))
    # Additive -inf mask on pad slots: the softmax renormalizes over the
    # legal set only. masked_fill's backward is an exact zero at filled
    # positions, so the pad slots' id-0 gathers leak no gradient.
    log_p = torch.log_softmax(gathered.masked_fill(pad, float("-inf")), dim=1)
    # Mask the *output*: pad log-probs are -inf here; zeroing them (rather
    # than multiplying by a zero target) keeps 0 · (-inf) = nan out.
    log_p = log_p.masked_fill(pad, 0.0)
    per_sample = -(counts / totals * log_p).sum(dim=1)
    return per_sample.mean()


@dataclass(frozen=True)
class LossBreakdown:
    """Composite-loss components, kept separate for observability (§12 M3).

    Attributes:
        total: The scalar to backpropagate: ``value + policy + aux``.
        value: Batch-mean value MSE ``(z − v)²``.
        policy: The policy cross-entropy, passed through unchanged.
        aux: Weighted aux term ``Σ λ_i · MSE(aux_i)`` exactly as it enters
            ``total``, or ``None`` when the spec declares no aux heads.
    """

    total: torch.Tensor
    value: torch.Tensor
    policy: torch.Tensor
    aux: torch.Tensor | None


def composite_loss(
    policy_loss: torch.Tensor,
    value: torch.Tensor,
    z: torch.Tensor,
    aux_pred: torch.Tensor | None = None,
    aux_target: torch.Tensor | None = None,
    aux_weights: Sequence[float] = (),
) -> LossBreakdown:
    """Assemble the §7 composite loss: value MSE + policy CE + weighted aux MSEs.

    ``l = (z − v)² − πᵀ log p + Σ_i λ_i · MSE(aux_i)``, each term batch-mean.
    The §7 ``c‖θ‖²`` term is deliberately *not* computed here: weight decay
    lives in the SGD optimizer (``weight_decay=1e-4``, D5), and adding it to
    the loss as well would double-count it.

    ``aux_weights`` comes from the adapter's declared
    ``ValueTargetSpec.aux_loss_weights`` (§6.1) — read from the spec, never
    hardcoded. The empty spec (``aux_weights=()`` with no aux tensors) is a
    first-class path, not an edge case: Othello declares zero aux heads, and
    M3 drives it through exactly this code, where the total reduces
    bit-exactly to value MSE + policy CE.

    Args:
        policy_loss: Scalar policy cross-entropy, e.g. ``sparse_policy_loss``.
        value: ``(N,)`` value-head outputs ``v``.
        z: ``(N,)`` game-outcome targets (D1), same shape as ``value``.
        aux_pred: ``(N, num_aux)`` aux-head outputs, or ``None`` when the
            spec declares no aux heads (the network's "absent" convention).
        aux_target: ``(N, num_aux)`` aux targets (Blokus: ``score_diff/109``,
            D1), or ``None`` alongside ``aux_pred``.
        aux_weights: Per-head weights ``λ_i``, parallel to the aux columns —
            pass the adapter's ``ValueTargetSpec.aux_loss_weights``.

    Returns:
        ``LossBreakdown`` carrying the total and each component.

    Raises:
        ValueError: If ``value`` and ``z`` shapes disagree, the batch is
            empty (the batch-mean would be ``nan``), aux tensors are present
            under an empty spec (or absent under a nonempty one), or an aux
            tensor's shape disagrees with ``(N, len(aux_weights))``.
    """
    if value.shape != z.shape:
        raise ValueError(f"value shape {tuple(value.shape)} != z shape {tuple(z.shape)}")
    if value.numel() == 0:
        raise ValueError("empty batch: the batch-mean value loss is undefined")
    value_mse = torch.mean((z - value) ** 2)
    if len(aux_weights) == 0:
        if aux_pred is not None or aux_target is not None:
            raise ValueError("aux tensors passed under an empty aux spec")
        return LossBreakdown(
            total=value_mse + policy_loss, value=value_mse, policy=policy_loss, aux=None
        )
    if aux_pred is None or aux_target is None:
        raise ValueError(f"spec declares {len(aux_weights)} aux head(s) but aux tensors are absent")
    expected = (value.shape[0], len(aux_weights))
    if tuple(aux_pred.shape) != expected or tuple(aux_target.shape) != expected:
        raise ValueError(
            f"aux shapes {tuple(aux_pred.shape)}/{tuple(aux_target.shape)} != {expected} "
            "(batch, len(aux_weights))"
        )
    per_head = ((aux_target - aux_pred) ** 2).mean(dim=0)
    weights = torch.as_tensor(aux_weights, dtype=per_head.dtype, device=per_head.device)
    aux_term = (weights * per_head).sum()
    return LossBreakdown(
        total=value_mse + policy_loss + aux_term,
        value=value_mse,
        policy=policy_loss,
        aux=aux_term,
    )
