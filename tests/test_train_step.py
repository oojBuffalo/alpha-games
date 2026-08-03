"""Train-step battery (§12 M2): collate boundary, D5 recipe, AMP step.

Collate round-trip goldens on real Blokus samples (planes byte-for-byte, the
padded legal-id/count/mask contract, spec-carried aux weights); the no-aux
path end-to-end — an Othello-shaped batch with no aux slot collates and
completes a train step (M3's zero-``core/``-diff requirement reaching the
training stack); LR-schedule endpoint goldens (warmup peak exactly at the
configured step, cosine to exactly 0); one seeded CPU step with finite loss
and changed parameters; two steps on the same batch reducing the loss. The
whole battery runs GPU-free — autocast and GradScaler degrade to no-ops on
CPU — with one CUDA-guarded test for the AMP-real path. Seeded.
"""

from __future__ import annotations

import dataclasses
import random

import pytest
import torch

from core.losses import PAD_ID, pad_sparse_targets
from core.network import Network, NetworkConfig
from core.train import (
    BATCH_SIZE,
    LEARNING_RATE,
    collate,
    make_lr_lambda,
    make_lr_scheduler,
    make_optimizer,
    make_scaler,
    train_step,
)
from games.blokus_duo import BlokusDuo
from games.othello import Othello

torch.manual_seed(0)

BLOKUS = BlokusDuo()
OTHELLO = Othello()

# Micro trunks over the real games' declared dims: every head shape still
# matches the adapter (planes, policy width, aux count), but a step runs in
# milliseconds on CPU. The D5 trunk/batch are exercised for real by the M2
# GPU benchmark, not here.
MICRO_BLOKUS = NetworkConfig(
    46, (14, 14), (14, 14, 91), trunk_blocks=1, trunk_channels=8, num_aux=1
)
MICRO_OTHELLO = NetworkConfig(2, (8, 8), (65,), trunk_blocks=1, trunk_channels=8)


def game_states(game, n):
    """The initial state plus ``n - 1`` min-action successors."""
    states = [game.initial_state()]
    while len(states) < n:
        states.append(game.apply(states[-1], min(game.legal_moves(states[-1]))))
    return states


def make_samples(game, n, seed):
    """Seeded collate-input samples from real game states.

    Planes come from ``encode_state``; sparse π draws a ragged legal subset
    with ΣN > 0 (D10); z/aux targets are synthetic. The aux slot is present
    iff the game's spec declares aux heads.
    """
    rng = random.Random(seed)
    num_aux = len(game.value_targets.aux_names)
    samples = []
    for state in game_states(game, n):
        legal = list(game.legal_moves(state))
        ids = rng.sample(legal, min(len(legal), rng.randint(1, 6)))
        counts = [rng.randint(0, 5) for _ in ids]
        counts[rng.randrange(len(ids))] += 1  # keep ΣN > 0 (D10)
        pairs = list(zip(ids, counts, strict=True))
        z = rng.choice([-1, 0, 1])
        if num_aux:
            aux = tuple(rng.uniform(-1.0, 1.0) for _ in range(num_aux))
            samples.append((game.encode_state(state), pairs, z, aux))
        else:
            samples.append((game.encode_state(state), pairs, z))
    return samples


# --- collate -------------------------------------------------------------------------


def test_collate_blokus_round_trip_golden():
    samples = make_samples(BLOKUS, 3, seed=1)
    batch = collate(BLOKUS, samples)
    assert batch.planes.shape == (3, 46, 14, 14)
    assert batch.planes.dtype == torch.float32
    for i, (planes, pairs, z, aux) in enumerate(samples):
        # Byte-for-byte round trip of the stdlib encoding (values in {0, 1}).
        assert torch.equal(batch.planes[i], torch.tensor(planes, dtype=torch.float32))
        assert batch.z[i].item() == float(z)
        assert torch.equal(batch.aux[i], torch.tensor(aux, dtype=torch.float32))
        row_ids = batch.legal_ids[i, : len(pairs)].tolist()
        row_counts = batch.visit_counts[i, : len(pairs)].tolist()
        assert list(zip(row_ids, row_counts, strict=True)) == [(a, float(c)) for a, c in pairs]
    ref_ids, ref_counts = pad_sparse_targets([pairs for _, pairs, _, _ in samples])
    assert torch.equal(batch.legal_ids, ref_ids)
    assert torch.equal(batch.visit_counts, ref_counts)
    # Aux weights ride along from the declared spec — never hardcoded.
    assert batch.aux_weights == BLOKUS.value_targets.aux_loss_weights == (0.25,)


def test_collate_padding_mask_matches_legal_counts():
    samples = make_samples(BLOKUS, 4, seed=2)
    batch = collate(BLOKUS, samples)
    widths = [len(pairs) for _, pairs, _, _ in samples]
    assert batch.legal_ids.shape[1] == max(widths)
    mask = batch.legal_ids >= 0
    assert mask.sum(dim=1).tolist() == widths
    assert (batch.legal_ids[~mask] == PAD_ID).all()
    assert (batch.visit_counts[~mask] == 0).all()


def test_collate_othello_emits_no_aux():
    # The spec-driven no-aux path: 3-tuple samples, no aux tensor, empty
    # weights — absence, not zero-fills.
    samples = make_samples(OTHELLO, 3, seed=3)
    assert all(len(s) == 3 for s in samples)
    batch = collate(OTHELLO, samples)
    assert batch.planes.shape == (3, 2, 8, 8)
    assert batch.aux is None
    assert batch.aux_weights == ()


def test_collate_rejects_spec_arity_mismatches():
    blokus = make_samples(BLOKUS, 2, seed=4)
    othello = make_samples(OTHELLO, 2, seed=4)
    with pytest.raises(ValueError):
        collate(BLOKUS, [])  # empty batch
    with pytest.raises(ValueError):
        collate(BLOKUS, [s[:3] for s in blokus])  # aux game, aux slot missing
    with pytest.raises(ValueError):
        collate(OTHELLO, [(*s, (0.5,)) for s in othello])  # no-aux game, aux slot present
    with pytest.raises(ValueError):
        collate(BLOKUS, [(p, pi, z, (*a, 0.5)) for p, pi, z, a in blokus])  # aux too wide
    with pytest.raises(ValueError):
        collate(BLOKUS, [(p, pi, z, 0.5) for p, pi, z, _ in blokus])  # bare-float aux


def test_collate_rejects_planes_off_the_declared_shape():
    # Othello-encoded planes under the Blokus declaration: right arity, wrong
    # (input_planes, H, W).
    blokus = make_samples(BLOKUS, 2, seed=5)
    othello = make_samples(OTHELLO, 2, seed=5)
    mixed = [(o[0], pi, z, a) for o, (_, pi, z, a) in zip(othello, blokus, strict=True)]
    with pytest.raises(ValueError):
        collate(BLOKUS, mixed)


# --- D5 constants and optimizer ------------------------------------------------------


def test_d5_defaults_are_explicit_constants():
    assert BATCH_SIZE == 256  # D5: batch 256, benchmark 128/256/512
    assert LEARNING_RATE == 0.02


def test_make_optimizer_d5_recipe():
    torch.manual_seed(10)
    net = Network(MICRO_OTHELLO)
    opt = make_optimizer(net)
    assert isinstance(opt, torch.optim.SGD)
    (group,) = opt.param_groups
    assert group["lr"] == LEARNING_RATE
    assert group["momentum"] == 0.9
    assert group["weight_decay"] == 1e-4
    assert len(group["params"]) == len(list(net.parameters()))


# --- LR schedule goldens -------------------------------------------------------------


def test_lr_lambda_endpoint_goldens():
    fn = make_lr_lambda(warmup_steps=4, total_steps=16)
    assert fn(0) == 0.0
    assert fn(1) == 0.25  # linear warmup
    assert fn(2) == 0.5
    assert fn(4) == 1.0  # warmup peak exactly at the configured step
    assert fn(10) == pytest.approx(0.5)  # cosine midpoint
    assert fn(16) == 0.0  # cosine floor exactly at total_steps
    assert fn(100) == 0.0  # and stays there


def test_lr_lambda_monotone_up_then_down():
    fn = make_lr_lambda(warmup_steps=5, total_steps=40)
    warmup = [fn(s) for s in range(6)]
    assert warmup == sorted(warmup) and warmup[-1] == 1.0
    cosine = [fn(s) for s in range(5, 41)]
    assert cosine == sorted(cosine, reverse=True)
    assert all(f > 0 for f in cosine[:-1]) and cosine[-1] == 0.0


def test_lr_lambda_zero_warmup_starts_at_peak():
    fn = make_lr_lambda(warmup_steps=0, total_steps=8)
    assert fn(0) == 1.0
    assert fn(8) == 0.0


def test_lr_lambda_rejects_bad_shape_config():
    with pytest.raises(ValueError):
        make_lr_lambda(warmup_steps=-1, total_steps=10)
    with pytest.raises(ValueError):
        make_lr_lambda(warmup_steps=4, total_steps=4)  # warmup must end before total
    with pytest.raises(ValueError):
        make_lr_lambda(warmup_steps=0, total_steps=0)


def test_lr_scheduler_applies_shape_to_base_lr():
    torch.manual_seed(11)
    opt = make_optimizer(Network(MICRO_OTHELLO))
    sched = make_lr_scheduler(opt, warmup_steps=3, total_steps=12)
    opt.step()  # a first optimizer step, as in the real loop
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(12):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] == 0.0
    assert lrs[3] == LEARNING_RATE  # warmup peak == the D5 base LR
    assert lrs[12] == 0.0


# --- train_step ----------------------------------------------------------------------


def test_train_step_cpu_finite_and_updates_params():
    torch.manual_seed(12)
    net = Network(MICRO_BLOKUS)
    opt = make_optimizer(net)
    scaler = make_scaler("cpu")
    assert not scaler.is_enabled()  # degraded to a no-op: the battery is GPU-free
    batch = collate(BLOKUS, make_samples(BLOKUS, 4, seed=6))
    before = net.stem[0].weight.clone()
    parts = train_step(net, opt, scaler, batch)
    for term in (parts.total, parts.value, parts.policy, parts.aux):
        assert torch.isfinite(term)
        assert not term.requires_grad  # detached: observability only
    assert torch.allclose(parts.total, parts.value + parts.policy + parts.aux)
    assert not torch.equal(net.stem[0].weight, before)


def test_two_steps_on_same_batch_reduce_loss():
    # A gentler LR than the D5 base: 0.02 is a batch-256 setting and
    # overshoots on this 4-sample micro batch; the property under test is
    # that the step descends, not the D5 tuning.
    torch.manual_seed(13)
    net = Network(MICRO_BLOKUS)
    opt = make_optimizer(net, lr=1e-3)
    scaler = make_scaler("cpu")
    batch = collate(BLOKUS, make_samples(BLOKUS, 4, seed=7))
    first = train_step(net, opt, scaler, batch)
    second = train_step(net, opt, scaler, batch)
    assert second.total < first.total


def test_othello_no_aux_batch_end_to_end():
    # The M3 zero-core-diff requirement reaching the training stack: an
    # Othello-shaped batch (no aux anywhere) collates and completes a step.
    torch.manual_seed(14)
    net = Network(MICRO_OTHELLO)
    opt = make_optimizer(net)
    batch = collate(OTHELLO, make_samples(OTHELLO, 4, seed=8))
    before = net.stem[0].weight.clone()
    parts = train_step(net, opt, make_scaler("cpu"), batch)
    assert parts.aux is None
    assert torch.isfinite(parts.total)
    assert torch.allclose(parts.total, parts.value + parts.policy)
    assert not torch.equal(net.stem[0].weight, before)


def test_train_step_rejects_net_batch_aux_mismatch():
    # An aux-emitting net on a batch whose spec declares no aux heads must
    # fail loudly (composite_loss is the shared spec checkpoint).
    torch.manual_seed(15)
    net = Network(MICRO_BLOKUS)
    batch = collate(BLOKUS, make_samples(BLOKUS, 2, seed=9))
    stripped = dataclasses.replace(batch, aux=None, aux_weights=())
    with pytest.raises(ValueError):
        train_step(net, make_optimizer(net), make_scaler("cpu"), stripped)


def test_batch_to_returns_equal_batch_on_cpu():
    batch = collate(BLOKUS, make_samples(BLOKUS, 2, seed=10))
    moved = batch.to("cpu")
    assert torch.equal(moved.planes, batch.planes)
    assert torch.equal(moved.legal_ids, batch.legal_ids)
    assert torch.equal(moved.aux, batch.aux)
    assert moved.aux_weights == batch.aux_weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP-real path needs CUDA")
def test_train_step_amp_real_on_cuda():
    torch.manual_seed(16)
    net = Network(MICRO_BLOKUS).cuda()
    opt = make_optimizer(net)
    scaler = make_scaler("cuda")
    assert scaler.is_enabled()
    batch = collate(BLOKUS, make_samples(BLOKUS, 4, seed=11)).to("cuda")
    parts = train_step(net, opt, scaler, batch)
    assert torch.isfinite(parts.total)
