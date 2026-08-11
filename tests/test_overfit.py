"""Overfit-one-batch: the §12 M2 exit test — the whole model memorizes one batch.

"The overfit-one-batch test presupposes the whole model": one small batch of
real Blokus positions goes through ``encode_state`` → ``collate`` → the *full
D5 network* (``NetworkConfig.from_game`` — 8×128 trunk, spatial policy head,
tanh value, score-diff aux) → sparse policy CE + §7 composite loss (the
spec-carried ``λ_aux = 0.25``) → the D5 SGD recipe at the base LR, repeatedly
on the same batch, until the model memorizes it. Any wiring break anywhere in
the M2 stack — plane order, §5.1 flatten, sparse gather, loss assembly,
optimizer — keeps the loss high and fails the assertions.

Everything is seeded and CPU-only, full precision by construction: on CPU
``train_step``'s autocast is disabled and the GradScaler is a no-op
(determinism beats speed here; AMP-for-real is the M2 GPU benchmark's job).
100 steps at the D5 base LR suffice — observed convergence is total ≈ 1e-4
with 12/12 policy argmax, |v − z| ≤ 0.01, and aux MSE ≈ 2.5e-5, so the
asserted thresholds carry roughly two orders of magnitude of cross-platform
slack — and keep the runtime in tens of seconds on CPU, which is why the
module is ``slow``-marked.

All three heads are asserted independently in eval mode. The composite loss
alone cannot vouch for the aux head: at λ_aux = 0.25 and this fixture's
±10/109 targets, a dead constant-zero aux head costs only
``0.25 · (10/109)² ≈ 2e-3`` — invisible under ``LOSS_THRESHOLD``.

What the aux assertion does and does not prove: this batch comes from a single
decisive playout, so its aux targets are exactly ``z · 10/109`` — perfectly
correlated with the value targets. Memorizing ``score_diff`` and merely
rescaling the value signal are therefore indistinguishable *on this fixture*,
and no assertion over it could separate them. What the check does prove is
that the head is neither dead nor constant: the targets' own variance puts
every constant predictor (including the batch mean, MSE ≈ 8.2e-3) above
``AUX_MSE_THRESHOLD``. Separating the two would need a fixture spanning
several playouts with differing score margins at equal z — a discrimination
this exit test does not claim.
"""

from __future__ import annotations

import random
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F

from core.network import Network, NetworkConfig
from core.train import collate, make_optimizer, make_scaler, train_step
from games.blokus_duo import BlokusDuo
from games.blokus_duo.bitboard import BitboardEngine
from games.blokus_duo.targets import AUX_LOSS_WEIGHT, value_targets

SEED = 0

# Batch/step budget, tuned empirically on CPU: 12 positions memorize in well
# under 100 steps at the D5 base LR (no higher LR needed), and 100 full-D5
# steps run in tens of seconds. The seed-0 playout is a 24-ply decisive game;
# its 12 sampled plies span both z signs and legal widths from 3 to 617.
BATCH_POSITIONS = 12
TRAIN_STEPS = 100

# All of the played position's visits on the played action (D12 pairs; the
# magnitude is irrelevant — π_train normalizes to the one-hot either way).
VISITS = 256

# Assertion thresholds, far above the observed converged values (≈ 1e-4 /
# 12⁄12 / 0.01) yet far below any broken-wiring outcome: the untrained loss
# starts ≈ 6.6, an unlearned tanh value sits ≈ 1 from its ±1 target, and
# chance argmax over the ≥ 3-wide legal sets is nowhere near 90%.
LOSS_THRESHOLD = 0.05
MIN_POLICY_ACCURACY = 0.9
VALUE_TOLERANCE = 0.25

# Unweighted aux-MSE threshold, calibrated on this fixture: converged
# ≈ 2.5e-5, a dead constant-zero head ≈ (10/109)² ≈ 8.4e-3 — the threshold
# sits 80× above the former and 4× below the latter, so a head that predicts
# nothing fails even though λ_aux hides it inside the composite loss.
AUX_MSE_THRESHOLD = 0.002


class _PlayoutPly(NamedTuple):
    """One recorded ply of the seeded playout — the batch's raw material.

    Attributes:
        state: The position before the action was played.
        mover: ``current_player`` at that position; the perspective
            ``value_targets`` is computed from.
        action: The uniform-random action the playout took — the trained-on
            π target and the assertion-side ground truth.
        legal: The legal action ids at the position, in adapter order.
    """

    state: object
    mover: int
    action: int
    legal: list[int]


def playout_batch(game, engine, rng, n_positions):
    """Sample a training batch from the plies of one seeded random playout.

    Plays one full game of uniform-random legal actions, then samples
    ``n_positions`` distinct plies. Distinct plies of a single game are
    distinct states (occupancy strictly grows), so no two samples can carry
    identical planes with conflicting targets — the degenerate fixture that
    sampling early plies of *several* playouts produces (every game shares
    the initial state). Targets are synthetic-but-plausible: sparse π puts
    all visits on the action the playout took, and ``z``/``aux`` come from
    ``value_targets`` on the played-out final scores, mover-relative.

    Args:
        game: The ``BlokusDuo`` adapter to play through.
        engine: The rules engine backing ``game``, injected [F8] so the
            terminal score pair is reachable for ``value_targets``.
        rng: Seeded ``random.Random`` driving playout and ply sampling.
        n_positions: Number of plies to sample; the playout must be at least
            this long.

    Returns:
        ``(samples, played)``: collate-ready ``(planes, sparse_pi, z, aux)``
        tuples, and the played action id per position — the assertion-side
        ground truth, held apart from the trained-on policy targets.
    """
    trail = []
    state = game.initial_state()
    while not game.is_terminal(state):
        legal = list(game.legal_moves(state))
        action = rng.choice(legal)
        trail.append(_PlayoutPly(state, game.current_player(state), action, legal))
        state = game.apply(state, action)
    scores = engine.scores(state)
    samples, played = [], []
    for index in sorted(rng.sample(range(len(trail)), n_positions)):
        ply = trail[index]
        z, aux = value_targets(scores[ply.mover], scores[1 - ply.mover])
        pairs = [(a, VISITS if a == ply.action else 0) for a in ply.legal]
        samples.append((game.encode_state(ply.state), pairs, float(z), (aux,)))
        played.append(ply.action)
    return samples, played


@pytest.mark.slow
def test_overfit_one_batch():
    """The full D5 model + loss + optimizer memorizes one real batch (§12 M2)."""
    torch.manual_seed(SEED)
    engine = BitboardEngine()
    game = BlokusDuo(engine)
    samples, played = playout_batch(game, engine, random.Random(SEED), BATCH_POSITIONS)
    batch = collate(game, samples)
    # Fixture sanity: a decisive game seen from both movers — the value head
    # must separate the +1 and −1 positions, not collapse to a constant.
    assert set(batch.z.tolist()) == {-1.0, 1.0}
    # The §7-pinned λ_aux, consumed via the declared spec — this test is why
    # the value had to be pinned doc-first.
    assert batch.aux_weights == (AUX_LOSS_WEIGHT,) == (0.25,)

    # The *full* D5 config from the game — 8×128 trunk, (14, 14, 91) head,
    # one aux — and the D5 recipe verbatim (SGD 0.9/1e-4 at base LR 0.02).
    net = Network(NetworkConfig.from_game(game))
    optimizer = make_optimizer(net)
    scaler = make_scaler("cpu")  # disabled: full precision, deterministic
    for _ in range(TRAIN_STEPS):
        parts = train_step(net, optimizer, scaler, batch)
    assert parts.total.item() < LOSS_THRESHOLD

    net.eval()
    with torch.no_grad():
        logits, value, aux_pred = net(batch.planes)
    # Argmax over each position's legal set (illegal logits never enter the
    # loss and mean nothing) against the *played* action, not the trained-on
    # pairs — so sabotaged policy targets cannot re-green this assertion.
    mask = batch.legal_ids >= 0
    legal_logits = logits.gather(1, batch.legal_ids.clamp(min=0))
    best = legal_logits.masked_fill(~mask, float("-inf")).argmax(dim=1, keepdim=True)
    predicted = batch.legal_ids.gather(1, best).squeeze(1)
    accuracy = (predicted == torch.tensor(played)).float().mean().item()
    assert accuracy >= MIN_POLICY_ACCURACY
    assert (value - batch.z).abs().max().item() <= VALUE_TOLERANCE
    # The aux head must memorize too, judged unweighted: inside the composite,
    # λ_aux and the small ±10/109 targets make a dead zero head cost ≈ 2e-3 —
    # under LOSS_THRESHOLD — so only this direct MSE check can catch it.
    assert F.mse_loss(aux_pred, batch.aux).item() < AUX_MSE_THRESHOLD
