"""Loss battery (§12 M2): sparse policy CE + §7 composite loss.

Hand-computed goldens on tiny legal sets, an inline dense renormalized-softmax
reference, and the invariance battery: perturbing an *illegal* logit leaves
loss and gradients bit-identical; pad slots contribute exactly zero loss and
gradient; ``aux_weights=(0,)`` removes the aux term bit-exactly; the empty
aux spec (``aux_weights=()``, no aux tensors — the Othello path) equals the
value+policy-only loss. Seeded, CPU-only.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from core.losses import PAD_ID, composite_loss, pad_sparse_targets, sparse_policy_loss
from games.blokus_duo.targets import AUX_LOSS_WEIGHT, value_target_spec
from games.othello import Othello

torch.manual_seed(0)

NUM_ACTIONS = 40


def dense_reference(logits_row, pairs):
    """Per-sample CE via an explicit renormalized softmax over the legal set."""
    legal = torch.stack([logits_row[a] for a, _ in pairs])
    exp = torch.exp(legal - legal.max())
    p = exp / exp.sum()
    total = sum(c for _, c in pairs)
    pi = torch.tensor([c / total for _, c in pairs], dtype=logits_row.dtype)
    return -(pi * torch.log(p)).sum()


def random_batch(seed, n=6, num_actions=NUM_ACTIONS):
    """Seeded ragged batch: logits + per-sample D12 (action_id, count) pairs."""
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, num_actions, generator=g)
    samples = []
    for _ in range(n):
        ids = rng.sample(range(num_actions), rng.randint(1, 8))
        counts = [rng.randint(0, 5) for _ in ids]
        counts[rng.randrange(len(ids))] += 1  # keep ΣN > 0 (D10)
        samples.append(list(zip(ids, counts, strict=True)))
    return logits, samples


# --- pad_sparse_targets --------------------------------------------------------------


def test_pad_sparse_targets_shapes_and_padding():
    legal_ids, visit_counts = pad_sparse_targets([[(3, 2), (7, 0), (1, 5)], [(4, 1)]])
    assert torch.equal(legal_ids, torch.tensor([[3, 7, 1], [4, PAD_ID, PAD_ID]]))
    assert torch.equal(visit_counts, torch.tensor([[2.0, 0.0, 5.0], [1.0, 0.0, 0.0]]))
    assert legal_ids.dtype == torch.int64 and visit_counts.dtype == torch.float32


def test_pad_sparse_targets_rejects_bad_input():
    with pytest.raises(ValueError):
        pad_sparse_targets([])
    with pytest.raises(ValueError):
        pad_sparse_targets([[(3, 2)], []])  # empty sample breaks the pass invariant
    with pytest.raises(ValueError):
        pad_sparse_targets([[(-1, 2)]])  # collides with the pad sentinel
    with pytest.raises(ValueError):
        pad_sparse_targets([[(-2, 2)]])  # any negative id, not just the sentinel


def test_pad_sparse_targets_rejects_malformed_rows():
    # The collate boundary owns structural validation (the loss body is
    # synchronization-free): duplicates, bad counts, and zero totals must
    # all die here, before any tensor reaches the hot path.
    with pytest.raises(ValueError, match="duplicate"):
        pad_sparse_targets([[(1, 1), (1, 2)]])  # would enter the softmax twice
    with pytest.raises(ValueError, match="finite"):
        pad_sparse_targets([[(3, float("inf"))]])
    with pytest.raises(ValueError, match="finite"):
        pad_sparse_targets([[(3, float("nan"))]])
    with pytest.raises(ValueError, match="finite"):
        pad_sparse_targets([[(3, 2), (5, -1)]])  # negative visit count
    with pytest.raises(ValueError, match="sum to zero"):
        pad_sparse_targets([[(3, 0), (5, 0)]])  # D10: π_train ∝ N needs ΣN > 0


# --- sparse_policy_loss goldens ------------------------------------------------------


def test_hand_computed_golden_two_actions():
    # Legal logits (ln 3, 0) → p = (3/4, 1/4); counts (1, 1) → π = (1/2, 1/2).
    logits = torch.zeros(1, NUM_ACTIONS)
    logits[0, 11] = math.log(3.0)
    legal_ids, visit_counts = pad_sparse_targets([[(11, 1), (29, 1)]])
    loss = sparse_policy_loss(logits, legal_ids, visit_counts)
    assert loss.item() == pytest.approx(-0.5 * (math.log(0.75) + math.log(0.25)), rel=1e-6)


def test_batch_mean_of_per_sample_losses():
    logits, samples = random_batch(1)
    legal_ids, visit_counts = pad_sparse_targets(samples)
    loss = sparse_policy_loss(logits, legal_ids, visit_counts)
    expected = torch.stack(
        [dense_reference(row, pairs) for row, pairs in zip(logits, samples, strict=True)]
    ).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_matches_inline_dense_reference_across_seeds():
    for seed in range(2, 6):
        logits, samples = random_batch(seed)
        legal_ids, visit_counts = pad_sparse_targets(samples)
        loss = sparse_policy_loss(logits, legal_ids, visit_counts)
        expected = torch.stack(
            [dense_reference(row, pairs) for row, pairs in zip(logits, samples, strict=True)]
        ).mean()
        assert torch.allclose(loss, expected, atol=1e-6), seed


def test_zero_count_legal_action_shapes_renormalization():
    # A zero-count legal action adds nothing to the CE sum but stays in the
    # softmax support: dropping it from the pairs must change the loss.
    logits, _ = random_batch(7, n=1)
    with_zero = [[(3, 2), (5, 0), (9, 1)]]
    without = [[(3, 2), (9, 1)]]
    loss_with = sparse_policy_loss(logits, *pad_sparse_targets(with_zero))
    loss_without = sparse_policy_loss(logits, *pad_sparse_targets(without))
    assert not torch.isclose(loss_with, loss_without)
    assert torch.allclose(loss_with, dense_reference(logits[0], with_zero[0]), atol=1e-6)


# --- invariance battery --------------------------------------------------------------


def test_illegal_logit_perturbation_is_bit_identical():
    logits, samples = random_batch(10)
    legal = {a for pairs in samples for a, _ in pairs}
    illegal = [a for a in range(NUM_ACTIONS) if a not in legal]
    assert illegal  # the batch must leave some actions untouched
    legal_ids, visit_counts = pad_sparse_targets(samples)

    base = logits.clone().requires_grad_()
    loss = sparse_policy_loss(base, legal_ids, visit_counts)
    loss.backward()

    perturbed = logits.clone()
    perturbed[:, illegal] += 1e6 * torch.randn(len(logits), len(illegal))
    perturbed.requires_grad_()
    loss_p = sparse_policy_loss(perturbed, legal_ids, visit_counts)
    loss_p.backward()

    assert torch.equal(loss, loss_p)  # bit-identical, not merely close
    assert torch.equal(base.grad, perturbed.grad)
    assert torch.equal(base.grad[:, illegal], torch.zeros(len(logits), len(illegal)))


def test_pad_slot_gradient_exactly_zero():
    # Row 1 pads out to row 0's width; pad slots gather action id 0, which is
    # illegal in every row — its gradient must still be exactly zero.
    logits = torch.randn(2, NUM_ACTIONS, generator=torch.Generator().manual_seed(11))
    logits.requires_grad_()
    legal_ids, visit_counts = pad_sparse_targets([[(3, 1), (5, 2), (7, 1)], [(9, 4)]])
    loss = sparse_policy_loss(logits, legal_ids, visit_counts)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[:, 0], torch.zeros(2))


def test_extra_padding_is_inert():
    logits, samples = random_batch(12)
    legal_ids, visit_counts = pad_sparse_targets(samples)
    widened_ids = torch.cat([legal_ids, torch.full((len(logits), 3), PAD_ID)], dim=1)
    widened_counts = torch.cat([visit_counts, torch.zeros(len(logits), 3)], dim=1)
    assert torch.equal(
        sparse_policy_loss(logits, legal_ids, visit_counts),
        sparse_policy_loss(logits, widened_ids, widened_counts),
    )


def test_gradients_finite_on_extreme_logits():
    logits, samples = random_batch(13)
    logits = (logits * 40.0).requires_grad_()  # widely separated legal logits
    legal_ids, visit_counts = pad_sparse_targets(samples)
    loss = sparse_policy_loss(logits, legal_ids, visit_counts)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_sparse_policy_loss_rejects_bad_metadata():
    # The loss checks tensor metadata only (row validity is owned by
    # pad_sparse_targets — no tensor-value test may run on the hot path);
    # both degenerate shapes would otherwise reduce to nan, not raise.
    logits, _ = random_batch(14, n=2)
    ids = torch.tensor([[3, 5], [4, 6]])
    with pytest.raises(ValueError):
        sparse_policy_loss(logits, ids, torch.tensor([[1.0], [1.0]]))  # shape mismatch
    with pytest.raises(ValueError):
        sparse_policy_loss(
            torch.zeros(0, NUM_ACTIONS),
            torch.zeros(0, 2, dtype=torch.int64),
            torch.zeros(0, 2),
        )  # empty batch
    with pytest.raises(ValueError):
        sparse_policy_loss(
            logits,
            torch.zeros(2, 0, dtype=torch.int64),
            torch.zeros(2, 0),
        )  # zero-width targets


def test_stray_negative_id_fails_loudly_not_silently():
    # Only PAD_ID marks padding. A hand-built -2 must not be silently
    # discarded as a pad slot: it reaches the gather out of range and raises.
    logits, _ = random_batch(15, n=1)
    ids = torch.tensor([[3, -2]])
    counts = torch.tensor([[1.0, 1.0]])
    with pytest.raises((RuntimeError, IndexError)):
        sparse_policy_loss(logits, ids, counts)


# --- composite_loss ------------------------------------------------------------------


def make_heads(seed, n=4, num_aux=1):
    g = torch.Generator().manual_seed(seed)
    value = torch.tanh(torch.randn(n, generator=g))
    z = torch.tensor([1.0, -1.0, 0.0, 1.0][:n])
    aux_pred = torch.randn(n, num_aux, generator=g)
    aux_target = torch.randn(n, num_aux, generator=g)
    return value, z, aux_pred, aux_target


def test_composite_golden_with_aux():
    policy = torch.tensor(0.625)
    value, z, aux_pred, aux_target = make_heads(20)
    spec = value_target_spec()  # Blokus: aux_loss_weights == (0.25,)
    parts = composite_loss(policy, value, z, aux_pred, aux_target, spec.aux_loss_weights)
    value_mse = ((z - value) ** 2).mean()
    aux_mse = ((aux_target - aux_pred) ** 2).mean()
    assert torch.allclose(parts.value, value_mse)
    assert torch.equal(parts.policy, policy)
    assert torch.allclose(parts.aux, AUX_LOSS_WEIGHT * aux_mse)
    assert torch.allclose(parts.total, value_mse + policy + AUX_LOSS_WEIGHT * aux_mse)


def test_empty_aux_spec_reduces_to_value_plus_policy():
    # The Othello no-aux path: aux_weights=() and no aux tensors at all —
    # first-class, and bit-exactly the value+policy-only loss.
    policy = torch.tensor(1.375)
    value, z, _, _ = make_heads(21)
    spec = Othello().value_targets
    assert spec.aux_loss_weights == ()
    parts = composite_loss(policy, value, z, aux_weights=spec.aux_loss_weights)
    assert parts.aux is None
    assert torch.equal(parts.total, ((z - value) ** 2).mean() + policy)


def test_zero_weight_removes_aux_term_bit_exactly():
    # Distinct from the empty spec: a present-but-zero-weighted head still
    # flows through the aux branch, yet must not move the total by a bit.
    policy = torch.tensor(0.75)
    value, z, aux_pred, aux_target = make_heads(22)
    weighted = composite_loss(policy, value, z, aux_pred, aux_target, (0.0,))
    bare = composite_loss(policy, value, z)
    assert torch.equal(weighted.aux, torch.tensor(0.0))
    assert torch.equal(weighted.total, bare.total)


def test_multi_head_aux_weights_apply_per_head():
    policy = torch.tensor(0.5)
    value, z, aux_pred, aux_target = make_heads(23, num_aux=2)
    parts = composite_loss(policy, value, z, aux_pred, aux_target, (0.25, 2.0))
    per_head = ((aux_target - aux_pred) ** 2).mean(dim=0)
    assert torch.allclose(parts.aux, 0.25 * per_head[0] + 2.0 * per_head[1])


def test_composite_rejects_mismatches():
    policy = torch.tensor(0.5)
    value, z, aux_pred, aux_target = make_heads(24)
    with pytest.raises(ValueError):
        composite_loss(policy, value, z[:-1])  # value/z shape disagreement
    with pytest.raises(ValueError):
        composite_loss(policy, value[:0], z[:0])  # empty batch: mean() would be nan
    with pytest.raises(ValueError):
        composite_loss(policy, value, z, aux_pred, aux_target, ())  # tensors, empty spec
    with pytest.raises(ValueError):
        composite_loss(policy, value, z, aux_weights=(0.25,))  # spec, no tensors
    with pytest.raises(ValueError):
        composite_loss(policy, value, z, aux_pred, aux_target, (0.25, 0.5))  # width


def test_composite_gradients_finite_end_to_end():
    logits, samples = random_batch(25, n=4)
    logits.requires_grad_()
    value, z, aux_pred, aux_target = make_heads(26)
    value = value.detach().requires_grad_()
    aux_pred = aux_pred.detach().requires_grad_()
    legal_ids, visit_counts = pad_sparse_targets(samples)
    policy = sparse_policy_loss(logits, legal_ids, visit_counts)
    parts = composite_loss(policy, value, z, aux_pred, aux_target, (AUX_LOSS_WEIGHT,))
    parts.total.backward()
    for leaf in (logits, value, aux_pred):
        assert torch.isfinite(leaf.grad).all()
