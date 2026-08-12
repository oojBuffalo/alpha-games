"""Micro-Blokus network battery: shape goldens, §5.1 flatten, overfit (§12 M2.5).

The §5.3 micro instance is the same ``games/blokus_duo/`` package under a
different config, so the network half of M2.5 has exactly one thing to prove:
that *no full-game constant leaks*. Every number below is a §5.3 golden —
12 planes, 5×5, ``(5, 5, 9)`` = 225 raw actions, aux divisor 29 — and none of
them is 46 / 14×14 / 91 / 17,836 / 109. The trunk is the one dimension that
deliberately does **not** shrink: §5.3 pins "the micro loop reuses the D5 8×128
trunk unchanged" so the M2.5 throughput number stays transferable to M3, so
8×128 is asserted here rather than assumed.

Three layers, mirroring ``tests/test_network.py`` and ``tests/test_overfit.py``:

1. ``NetworkConfig.from_game`` goldens plus a forward pass on genuinely
   micro-encoded states — the head must emit exactly ``prod(policy_shape)``
   logits, and the aux head's width and normalization must come from the
   adapter's declared ``ValueTargetSpec`` and its config-derived divisor, never
   from a hardcoded 109 (or a hardcoded 29).
2. The §5.1 HWC flatten at micro shapes: pre-flatten logit ``(o, r, c)`` must
   land at flat index ``(r*5 + c)*9 + o``, which must be the action id the
   adapter's own ``encode_action`` produces for that placement. A channel-major
   flatten passes every shape check while silently training each legal id
   against another placement's logit.
3. Overfit-one-batch on micro — the §12 M2 exit test re-run through the reduced
   instance, at ``tests/test_overfit.py``'s thresholds.

CPU-only, seeded, full precision by construction (``train_step``'s autocast is
disabled off-CUDA and the GradScaler is a no-op).
"""

from __future__ import annotations

import random
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F

from core.game import ValueTargetSpec
from core.network import Network, NetworkConfig
from core.train import collate, make_optimizer, make_scaler, train_step
from games.blokus_duo import BlokusDuo
from games.blokus_duo.actions import action_codec
from games.blokus_duo.config import FULL_CONFIG, MICRO_CONFIG
from games.blokus_duo.targets import AUX_LOSS_WEIGHT, MAX_SCORE_DIFF, max_score_diff, value_targets

torch.manual_seed(0)

# The §5.3 micro instance, built the documented way: a config argument, not a
# fork. (The default engine is the bitboard engine for this same config.)
MICRO = BlokusDuo(config=MICRO_CONFIG)
MICRO_CODEC = action_codec(MICRO_CONFIG)
MICRO_CFG = NetworkConfig.from_game(MICRO)
MICRO_NET = Network(MICRO_CFG).eval()

# §5.3 goldens (independently enumerated by scripts/enumerate_micro_config.py).
MICRO_PLANES = 12
MICRO_SHAPE = (5, 5)
MICRO_POLICY_SHAPE = (5, 5, 9)
MICRO_NUM_ACTIONS = 225
MICRO_IN_BOUNDS = 159
MICRO_OPENINGS = 42
MICRO_AUX_DIVISOR = 29

# D5 trunk (§5.3: pinned unchanged for the micro loop, not a defaulted leftover).
D5_TRUNK = (8, 128)

# Full-game constants that must not appear anywhere in the micro surface (§12
# M2.5: "none of the full-game golden constants carry over").
FULL_GAME_CONSTANTS = frozenset({14, 46, 91, 109, 17836})


def encode_batch(game, n):
    """Stack ``n`` encoded states (initial + successors) as a float batch.

    Args:
        game: The adapter to walk; successors are taken along the lowest legal
            action id, so the walk is deterministic without an RNG.
        n: Number of states to stack; the walk must not terminate before then.

    Returns:
        A ``(n, input_planes, *input_shape)`` float32 tensor.
    """
    states = [game.initial_state()]
    while len(states) < n:
        states.append(game.apply(states[-1], min(game.legal_moves(states[-1]))))
    return torch.tensor([game.encode_state(s) for s in states], dtype=torch.float32)


def policy_preflatten(net, x):
    """Recompute the spatial policy head up to — not including — the flatten.

    Args:
        net: A ``Network`` with a spatial policy head.
        x: Its float input batch.

    Returns:
        The ``(N, C, H, W)`` pre-flatten logits.
    """
    return net.policy_conv(net.blocks(net.stem(x)))


# --- 1. shape goldens: no full-game constant leaks -----------------------------------


def test_from_game_micro_golden():
    """``from_game`` on the micro adapter yields the §5.3 dims and the D5 trunk."""
    assert MICRO_CFG == NetworkConfig(
        input_planes=MICRO_PLANES,
        input_shape=MICRO_SHAPE,
        policy_shape=MICRO_POLICY_SHAPE,
        trunk_blocks=D5_TRUNK[0],
        trunk_channels=D5_TRUNK[1],
        num_aux=1,
    )
    # The trunk is pinned, not merely defaulted (§5.3): re-pinning a smaller
    # trunk was considered and rejected to keep the throughput number
    # transferable to M3.
    assert (MICRO_CFG.trunk_blocks, MICRO_CFG.trunk_channels) == D5_TRUNK
    # prod(policy_shape) is the micro raw-action count, and it is the codec's.
    assert MICRO_CFG.num_actions == MICRO_NUM_ACTIONS == MICRO_CODEC.num_actions
    assert MICRO_CODEC.num_orientations == MICRO_POLICY_SHAPE[2]
    assert len(MICRO_CODEC.in_bounds_actions) == MICRO_IN_BOUNDS


def test_no_full_game_constant_leaks_into_the_micro_config():
    """None of 46 / 14 / 91 / 17,836 / 109 survives into the micro surface."""
    surface = {
        MICRO_CFG.input_planes,
        *MICRO_CFG.input_shape,
        *MICRO_CFG.policy_shape,
        MICRO_CFG.num_actions,
        max_score_diff(MICRO_CONFIG),
    }
    assert not surface & FULL_GAME_CONSTANTS
    # ... and the full game still carries them, so the two instances are
    # genuinely parameterized apart rather than both quietly reduced.
    assert MAX_SCORE_DIFF == 109
    assert max_score_diff(MICRO_CONFIG) == MICRO_AUX_DIVISOR == 29


def test_micro_forward_shapes_and_value_range():
    """A forward pass on micro-encoded states emits 225 logits, ``(N,)``, ``(N, 1)``."""
    x = encode_batch(MICRO, 4)
    assert x.shape == (4, MICRO_PLANES, *MICRO_SHAPE)
    with torch.no_grad():
        policy, value, aux = MICRO_NET(x)
    assert policy.shape == (4, MICRO_NUM_ACTIONS)
    assert value.shape == (4,)
    assert aux.shape == (4, 1)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_aux_head_and_normalization_ride_the_declared_spec():
    """Aux width comes from ``value_targets``; the divisor from the instance config.

    The head is one wide because the adapter *declares* one aux name, and the
    targets it is trained on are divided by this instance's own bound — the
    check that the D1 ``/109`` did not ride along into micro. λ_aux and the aux
    name ride the same declaration.
    """
    spec = MICRO.value_targets
    assert spec == ValueTargetSpec("z", ("score_diff",), (AUX_LOSS_WEIGHT,))
    assert spec.aux_loss_weights == (0.25,)  # §7, pinned doc-first at M2
    assert MICRO_CFG.num_aux == len(spec.aux_names) == 1
    # The extreme micro score pair (§5.3: −9 against +20) normalizes to exactly
    # ±1 under the micro divisor; the same pair under the full game's 109 does
    # not — so a leaked divisor is visible in the target, not just in a comment.
    assert value_targets(20, -9, MICRO_CONFIG) == (1, 1.0)
    assert value_targets(-9, 20, MICRO_CONFIG) == (-1, -1.0)
    assert value_targets(20, -9, FULL_CONFIG)[1] == MICRO_AUX_DIVISOR / MAX_SCORE_DIFF
    # The adapter's generic terminal-target surface carries the same pair,
    # one aux value per declared head.
    terminal = MICRO.initial_state()
    while not MICRO.is_terminal(terminal):
        terminal = MICRO.apply(terminal, min(MICRO.legal_moves(terminal)))
    z, aux = MICRO.training_targets(terminal, 0)
    assert len(aux) == len(spec.aux_names)
    assert z in (-1.0, 0.0, 1.0) and abs(aux[0]) <= 1.0
    assert aux[0] * MICRO_AUX_DIVISOR == round(aux[0] * MICRO_AUX_DIVISOR)  # integer score diff


# --- 2. §5.1 spatial flatten at micro shapes -----------------------------------------


def test_flatten_order_golden_at_micro_shapes():
    """Pre-flatten ``(o, r, c)`` lands at ``(r*5 + c)*9 + o`` — exhaustively."""
    x = encode_batch(MICRO, 1)
    with torch.no_grad():
        policy, _, _ = MICRO_NET(x)
        pre = policy_preflatten(MICRO_NET, x)
    h, w, k = MICRO_POLICY_SHAPE
    assert pre.shape == (1, k, h, w)
    for r in range(h):
        for c in range(w):
            for o in range(k):
                assert policy[0, (r * w + c) * k + o] == pre[0, o, r, c]


def test_flat_index_is_the_adapter_encoded_action_id():
    """Every in-bounds micro action id *is* its logit's flat index (§5.1, D2).

    Closes the loop the shape checks cannot: the flatten formula is exercised
    against ids produced by the adapter's own ``encode_action`` (cells → id),
    reached from ``decode_action`` (id → cells), with the anchor re-derived from
    the absolute cells as the D2 bounding-box top-left. The named failure mode
    is a stride or anchor mismatch that leaves every id pointing at some other
    placement's logit while every tensor shape stays correct.
    """
    h, w, k = MICRO_POLICY_SHAPE
    for action in MICRO_CODEC.in_bounds_actions:
        cells = MICRO.decode_action(action)
        assert MICRO.encode_action(cells) == action
        r, c = min(cell[0] for cell in cells), min(cell[1] for cell in cells)
        o = action % k
        assert action == (r * w + c) * k + o
        assert 0 <= r < h and 0 <= c < w


def test_legal_micro_ids_index_the_intended_logits():
    """The 42 opening legal ids gather the logits their ``(r, c, o)`` cells own."""
    state = MICRO.initial_state()
    legal = list(MICRO.legal_moves(state))
    assert len(legal) == MICRO_OPENINGS
    x = torch.tensor([MICRO.encode_state(state)], dtype=torch.float32)
    with torch.no_grad():
        policy, _, _ = MICRO_NET(x)
        pre = policy_preflatten(MICRO_NET, x)
    _, w, k = MICRO_POLICY_SHAPE
    for action in legal:
        cells = MICRO.decode_action(action)
        # Every opening covers a start square (§4/§5.3: 21 per square).
        assert set(cells) & set(MICRO_CONFIG.start_squares)
        r, c = min(cell[0] for cell in cells), min(cell[1] for cell in cells)
        o = action % k
        assert action == (r * w + c) * k + o
        assert policy[0, action] == pre[0, o, r, c]


# --- 3. overfit-one-batch on micro (§12 M2 exit test, reduced instance) --------------

SEED = 1

# Batch/step budget. Micro games are 5–8 plies (§5.3: mean ≈ 6.2), so unlike the
# full-game fixture a single playout cannot fill a 12-position batch — the
# builder walks consecutive seeded playouts instead, deduplicating on encoded
# planes. 200 D5 steps at the base LR memorize it; the composite loss is already
# two orders under its threshold well before then, and the extra steps buy the
# aux head's margin (see AUX_MSE_THRESHOLD).
BATCH_POSITIONS = 12
TRAIN_STEPS = 200

# All of the played position's visits on the played action (D12 pairs; the
# magnitude is irrelevant — π_train normalizes to the one-hot either way).
VISITS = 256

# Assertion thresholds, carried over verbatim from tests/test_overfit.py so the
# micro exit test is judged exactly as the full-game one is. Observed converged
# values across five torch inits: total ≤ 2.0e-4, 12/12 policy argmax,
# |v − z| ≤ 0.013 — roughly two orders of slack, as at 14×14.
LOSS_THRESHOLD = 0.05
MIN_POLICY_ACCURACY = 0.9
VALUE_TOLERANCE = 0.25

# Unweighted aux-MSE threshold, recalibrated for this fixture (the full game's
# 0.002 is tuned to its own ±10/109 targets and means nothing here). On this
# batch: converged ≤ 1.2e-3 across five torch inits; a dead constant-zero head
# scores 0.193; and — the discrimination the M2 fixture explicitly could not
# claim — the *best possible* predictor that sees only z scores 0.078, because
# the batch spans two score margins (1/29 and 18/29) within each z sign. The
# threshold sits ~8× above the worst converged run and ~65× below that z-only
# floor, so an aux head that merely rescales the value signal fails here.
AUX_MSE_THRESHOLD = 0.01


class _PlayoutPly(NamedTuple):
    """One recorded ply of a seeded micro playout — the batch's raw material.

    Attributes:
        state: The position before the action was played.
        mover: ``current_player`` at that position; the perspective
            ``training_targets`` is computed from.
        action: The uniform-random action the playout took — the trained-on
            π target and the assertion-side ground truth.
        legal: The legal action ids at the position, in adapter order.
    """

    state: object
    mover: int
    action: int
    legal: list[int]


def micro_playout_batch(game, rng, n_positions):
    """Sample a training batch from the plies of seeded random micro playouts.

    The micro analogue of ``tests/test_overfit.py``'s ``playout_batch``, with
    one structural difference forced by §5.3: a micro game is 5–8 plies, so the
    batch is filled from consecutive playouts rather than one. That reopens the
    degenerate-fixture risk the full-game builder avoided by construction (every
    playout shares the initial state, and two playouts can revisit a position
    with different final scores), so plies are deduplicated on their *encoded
    planes*: no two samples can carry identical planes with conflicting targets.

    Targets are synthetic-but-plausible in the same way: sparse π puts all
    visits on the action the playout took, over the position's full legal set,
    and ``(z, aux)`` come from the adapter's ``training_targets`` on the
    played-out terminal state, mover-relative — so the aux divisor is the
    instance's own 29, taken from the adapter rather than restated here.

    Args:
        game: The micro ``BlokusDuo`` adapter to play through.
        rng: Seeded ``random.Random`` driving the playouts.
        n_positions: Number of distinct positions to collect.

    Returns:
        ``(samples, played)``: collate-ready ``(planes, sparse_pi, z, aux)``
        tuples, and the played action id per position — the assertion-side
        ground truth, held apart from the trained-on policy targets.
    """
    samples, played, seen = [], [], set()
    while len(samples) < n_positions:
        trail = []
        state = game.initial_state()
        while not game.is_terminal(state):
            legal = list(game.legal_moves(state))
            action = rng.choice(legal)
            trail.append(_PlayoutPly(state, game.current_player(state), action, legal))
            state = game.apply(state, action)
        for ply in trail:
            planes = game.encode_state(ply.state)
            if planes in seen:
                continue
            seen.add(planes)
            z, aux = game.training_targets(state, ply.mover)
            pairs = [(a, VISITS if a == ply.action else 0) for a in ply.legal]
            samples.append((planes, pairs, z, aux))
            played.append(ply.action)
            if len(samples) == n_positions:
                break
    return samples, played


@pytest.mark.slow
def test_overfit_one_batch_micro():
    """The full D5 model + loss + optimizer memorizes one real micro batch (§12 M2.5)."""
    torch.manual_seed(SEED)
    samples, played = micro_playout_batch(MICRO, random.Random(SEED), BATCH_POSITIONS)
    batch = collate(MICRO, samples)
    assert batch.planes.shape == (BATCH_POSITIONS, MICRO_PLANES, *MICRO_SHAPE)
    # Fixture sanity. Micro draws are common (§5.3's short games end level
    # ~36% of the time under random play), so unlike the full-game fixture this
    # batch spans all three z values — a value head collapsing to a constant, or
    # to ±1, fails VALUE_TOLERANCE.
    assert set(batch.z.tolist()) == {-1.0, 0.0, 1.0}
    # Two distinct win margins at the same z: what makes AUX_MSE_THRESHOLD able
    # to reject a head that only rescales the value signal (see the constant).
    aux_by_z = [(z, abs(a)) for z, (a,) in zip(batch.z.tolist(), batch.aux.tolist(), strict=True)]
    assert len({a for z, a in aux_by_z if z > 0}) >= 2
    # The widest row is the full micro opening set (§5.3: 42) — the batch
    # exercises real legal widths, down to forced single-action positions.
    widths = (batch.legal_ids >= 0).sum(dim=1)
    assert int(widths.max()) == MICRO_OPENINGS
    # The §7-pinned λ_aux, consumed via the declared spec.
    assert batch.aux_weights == (AUX_LOSS_WEIGHT,) == (0.25,)

    # The *micro* D5 config from the game — 8×128 trunk, (5, 5, 9) head, one
    # aux — and the D5 recipe verbatim (SGD 0.9/1e-4 at base LR 0.02).
    net = Network(NetworkConfig.from_game(MICRO))
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
    # The aux head must memorize too, judged unweighted: λ_aux hides a dead head
    # inside the composite loss. What this can and cannot prove: the two margins
    # per z sign rule out both a dead head and a rescaled-value head *on this
    # batch*; it says nothing about generalization, which no one-batch test can.
    assert F.mse_loss(aux_pred, batch.aux).item() < AUX_MSE_THRESHOLD
