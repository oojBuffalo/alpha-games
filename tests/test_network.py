"""D5 network battery (§12 M2).

Forward shapes, the §5.1 flatten-order golden against ``actions.encode``, the
``from_game`` goldens (Blokus 46/(14,14)/(14,14,91)/1; Othello 2/(8,8)/(65,)/0),
and synthetic micro configs — square and non-square, spatial and flat heads —
proving nothing hardcodes 46/14/91. CPU-only, seeded.
"""

from __future__ import annotations

import random

import pytest
import torch

from core.network import Network, NetworkConfig
from games.blokus_duo import BlokusDuo
from games.blokus_duo.actions import NUM_ACTIONS, encode
from games.othello import Othello

torch.manual_seed(0)

BLOKUS = BlokusDuo()
BLOKUS_CFG = NetworkConfig.from_game(BLOKUS)
BLOKUS_NET = Network(BLOKUS_CFG).eval()
OTHELLO = Othello()
OTHELLO_NET = Network(NetworkConfig.from_game(OTHELLO)).eval()

# Synthetic micro configs (§12 M2.5 parameterization proof): reduced trunks
# keep them cheap; every head dimension still derives from the config.
MICRO_SQUARE = NetworkConfig(7, (5, 5), (5, 5, 3), trunk_blocks=2, trunk_channels=8, num_aux=2)
MICRO_WIDE = NetworkConfig(3, (6, 7), (6, 7, 4), trunk_blocks=2, trunk_channels=8)
MICRO_WIDE_FLAT = NetworkConfig(3, (6, 7), (43,), trunk_blocks=2, trunk_channels=8)


def encode_batch(game, n):
    """Stack ``n`` encoded states (initial + successors) as a float batch."""
    states = [game.initial_state()]
    while len(states) < n:
        states.append(game.apply(states[-1], min(game.legal_moves(states[-1]))))
    return torch.tensor([game.encode_state(s) for s in states], dtype=torch.float32)


def policy_preflatten(net, x):
    """Recompute the spatial policy head up to — not including — the flatten."""
    return net.policy_conv(net.blocks(net.stem(x)))


# --- config goldens and rejection ---------------------------------------------------


def test_from_game_blokus_golden():
    assert BLOKUS_CFG == NetworkConfig(
        input_planes=46,
        input_shape=(14, 14),
        policy_shape=(14, 14, 91),
        trunk_blocks=8,
        trunk_channels=128,
        num_aux=1,
    )
    assert BLOKUS_CFG.num_actions == NUM_ACTIONS == 17836


def test_from_game_othello_golden():
    cfg = NetworkConfig.from_game(OTHELLO)
    assert (cfg.input_planes, cfg.input_shape, cfg.policy_shape) == (2, (8, 8), (65,))
    assert cfg.num_aux == 0
    assert cfg.num_actions == 65
    assert (cfg.trunk_blocks, cfg.trunk_channels) == (8, 128)  # D5 defaults


def test_config_rejects_undeclared_shapes():
    with pytest.raises(ValueError):
        NetworkConfig(7, (5, 5), (5, 5))  # 2-tuple: neither spatial nor flat
    with pytest.raises(ValueError):
        NetworkConfig(7, (5, 5), (6, 5, 3))  # spatial head off input_shape in H
    with pytest.raises(ValueError):
        NetworkConfig(7, (5, 5, 1), (5, 5, 3))  # input_shape is not (H, W)
    with pytest.raises(ValueError):
        NetworkConfig(0, (5, 5), (25,))  # nonpositive dimension


# --- forward shapes ------------------------------------------------------------------


def test_blokus_forward_shapes_and_value_range():
    x = encode_batch(BLOKUS, 2)
    assert x.shape == (2, 46, 14, 14)
    with torch.no_grad():
        policy, value, aux = BLOKUS_NET(x)
    assert policy.shape == (2, 17836)
    assert value.shape == (2,)
    assert aux.shape == (2, 1)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_othello_flat_head_and_no_aux():
    x = encode_batch(OTHELLO, 2)
    assert x.shape == (2, 2, 8, 8)
    with torch.no_grad():
        policy, value, aux = OTHELLO_NET(x)
    assert policy.shape == (2, 65)
    assert value.shape == (2,)
    assert aux is None
    # num_aux = 0 builds no aux parameters at all; Blokus's num_aux = 1 does.
    assert not any("aux" in name for name, _ in OTHELLO_NET.named_parameters())
    assert any("aux" in name for name, _ in BLOKUS_NET.named_parameters())


def test_trunk_preserves_grid():
    torch.manual_seed(1)
    for cfg, net in ((BLOKUS_CFG, BLOKUS_NET), (MICRO_WIDE, Network(MICRO_WIDE).eval())):
        x = torch.randn(2, cfg.input_planes, *cfg.input_shape)
        with torch.no_grad():
            trunk = net.blocks(net.stem(x))
        assert trunk.shape == (2, cfg.trunk_channels, *cfg.input_shape)


# --- flatten-order goldens (§5.1) ----------------------------------------------------


def test_flatten_order_golden_matches_actions_encode():
    # The non-negotiable golden: the spatial head's pre-flatten (N, C, H, W)
    # logit at (o, r, c) must land at flat index (r*14+c)*91+o — a silent
    # channel-major flatten passes every shape check while corrupting every
    # training target.
    x = encode_batch(BLOKUS, 1)
    with torch.no_grad():
        policy, _, _ = BLOKUS_NET(x)
        pre = policy_preflatten(BLOKUS_NET, x)
    assert pre.shape == (1, 91, 14, 14)
    rng = random.Random(5)
    triples = [(r, c, o) for r in (0, 13) for c in (0, 13) for o in (0, 1, 90)]
    triples += [(rng.randrange(14), rng.randrange(14), rng.randrange(91)) for _ in range(200)]
    for r, c, o in triples:
        assert policy[0, encode(r, c, o)] == pre[0, o, r, c]


def test_flatten_order_on_nonsquare_grid():
    # Exhaustive on the 6×7 micro head: flat index (r*W+c)*C+o must use W=7 —
    # an H/W mixup is invisible on square boards.
    torch.manual_seed(2)
    net = Network(MICRO_WIDE).eval()
    x = torch.randn(1, 3, 6, 7)
    with torch.no_grad():
        policy, _, _ = net(x)
        pre = policy_preflatten(net, x)
    h, w, c = MICRO_WIDE.policy_shape
    assert pre.shape == (1, c, h, w)
    for r in range(h):
        for col in range(w):
            for o in range(c):
                assert policy[0, (r * w + col) * c + o] == pre[0, o, r, col]


# --- micro-config parameterization (§12 M2.5) ----------------------------------------


def test_micro_square_forward():
    torch.manual_seed(3)
    net = Network(MICRO_SQUARE).eval()
    with torch.no_grad():
        policy, value, aux = net(torch.randn(8, 7, 5, 5) * 50)
    assert policy.shape == (8, 75)
    assert value.shape == (8,)
    assert aux.shape == (8, 2)  # num_aux parameterizes the head width
    assert torch.all(value.abs() <= 1.0)  # tanh bound holds on extreme inputs


def test_micro_flat_head_on_nonsquare_grid():
    torch.manual_seed(4)
    net = Network(MICRO_WIDE_FLAT).eval()
    with torch.no_grad():
        policy, value, aux = net(torch.randn(3, 3, 6, 7))
    assert policy.shape == (3, 43)
    assert value.shape == (3,)
    assert aux is None


def test_gradients_flow_through_all_heads():
    torch.manual_seed(5)
    net = Network(MICRO_SQUARE)  # train mode
    policy, value, aux = net(torch.randn(4, 7, 5, 5))
    (policy.sum() + value.sum() + aux.sum()).backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, name
    assert net.stem[0].weight.grad.abs().sum() > 0
