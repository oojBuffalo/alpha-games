---
id: 8
title: Implement sparse policy and composite loss
status: pending
priority: high
dependencies: [7]
complexity: 5
recommended_subtasks: 3
---

## Description
The sparse policy cross-entropy (legal-actions-only masking + renormalization) and the §7
composite loss `l = (z − v)² − πᵀ log p + λ_aux·(aux term) + c‖θ‖²`.

## Details
- New `core/losses.py` (torch).
- `sparse_policy_loss(logits, legal_ids, visit_counts)`: gather each sample's legal-action logits
  from the flat head (the sparse gather §5.1 anticipates), `log_softmax` over the legal set only
  (nearly all of the 17,836 logits are illegal — masking + renormalization is load-bearing),
  targets `π_train(a) = N(a)/ΣN` (D10). Batch via padded legal-id tensors with an additive `-inf`
  mask for pad slots.
- `composite_loss(policy_loss, v, z, aux_pred, aux_target, aux_weights)`: value MSE `(z − v)²` +
  policy CE + `Σ λ_i · MSE(aux_i)` with `aux_weights` taken from the adapter's
  `ValueTargetSpec.aux_loss_weights` (pinned in task 1 — read it from the spec, don't hardcode).
  The `c‖θ‖²` term is *not* computed in the loss: it lives in SGD `weight_decay=1e-4` (D5);
  say so in the docstring so nobody double-counts it.
- Sparse targets come straight from the D12 replay shape `(action_id, visit_count)` — no dense
  17,836-vector materialization anywhere.

## Test Strategy
New `tests/test_losses.py`: hand-computed goldens on tiny cases (2–3 legal actions, known
logits/counts, values verified by a dense reference computed inline with explicit renormalized
softmax); perturbing an *illegal* logit leaves the loss bit-identical; `aux_weights=(0,)` removes
the aux term exactly; gradients are finite. Seeded, CPU-only.

## Complexity Analysis
Compact code with subtle failure modes: padded gathers with `-inf` masking interact with
`log_softmax` numerics (a stray `nan` from `-inf − -inf` is the classic bug), the legal-set-only
renormalization must leave illegal logits with exactly zero gradient, and the weight-decay term
must stay out of the loss to avoid double-counting against SGD. The invariant tests (illegal-logit
perturbation, dense-reference agreement) are what make this trustworthy, and they take comparable
effort to the implementation.

**Suggested expansion approach:** split three ways — (1) `sparse_policy_loss` with the padded
gather/mask batching; (2) `composite_loss` assembly reading `ValueTargetSpec.aux_loss_weights`;
(3) `tests/test_losses.py` goldens including the inline dense reference and invariance checks.
