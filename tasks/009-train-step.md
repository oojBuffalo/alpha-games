---
id: 9
title: Implement the AMP train step
status: pending
priority: high
dependencies: [8]
complexity:
recommended_subtasks:
---

## Description
One working train step with the D5 optimizer recipe: SGD momentum 0.9, weight decay 1e-4,
LR ≈0.02 with warmup+cosine, mixed precision (AMP). Plus the collate boundary that turns
stdlib-encoded samples into tensors.

## Details
- New `core/train.py` (torch).
- `make_optimizer(net, lr=0.02)`: `torch.optim.SGD(momentum=0.9, weight_decay=1e-4)` — the
  weight-decay term is the `c‖θ‖²` of the §7 loss (see task 8's docstring note).
- LR schedule factory: linear warmup then cosine decay (D5); warmup length and total steps are
  config arguments — M2 only needs the shape, M3 pins the run-length numbers.
- `collate(game, samples)`: converts a batch of `(planes, sparse_pi, z, aux)` — planes as the
  nested tuples from `encode_state`, π as `(action_id, visit_count)` pairs — into the tensor
  batch task 8's loss consumes (`numpy.asarray` → `torch.as_tensor`, padded legal-id/mask
  tensors). This is the single point where stdlib data becomes tensors.
- `train_step(net, optimizer, scaler, batch)`: AMP `autocast` + `GradScaler`, forward, composite
  loss, backward, step, return the loss components for observability. Device-aware: autocast and
  scaler must degrade to no-ops on CPU so the battery runs GPU-free (CI has no GPU); AMP is
  exercised for real on the 4060 Ti.

## Test Strategy
New `tests/test_train_step.py`: one step on a small seeded synthetic batch runs end-to-end on
CPU with finite loss and updated parameters (assert some parameter tensor changed); two steps on
the same batch reduce the loss; LR-schedule goldens (warmup endpoint hits the base LR, cosine
decays toward ~0); collate round-trip golden on a couple of real Blokus samples.
