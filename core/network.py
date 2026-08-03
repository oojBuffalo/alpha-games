"""The D5 network: conv stem, residual trunk, policy/value/aux heads (§7, §12 M2).

Torch lives here (and in ``core/losses.py`` / ``core/train.py``) only — the
pyproject confinement pin; adapters and the rest of ``core/`` stay stdlib-pure.

Every dimension derives from ``NetworkConfig`` — nothing hardcodes Blokus's
46 planes / 14×14 grid / 91 policy channels: §12 M2.5 requires a
config-parameterized net whose dims derive from the game, and M3's
zero-``core/``-diff Othello re-check drives a flat ``(65,)`` head and an
aux-free config through this same class, so both declared head shapes,
``num_aux = 0``, and non-square grids are first-class, not rejection branches.

The load-bearing convention is the spatial policy head's flatten (§5.1): the
``(N, C, H, W)`` conv logits are permuted to HWC before flattening so flat
index ``(r*W + c)*C + o`` matches ``games.blokus_duo.actions.encode`` — "M2
pays one tensor ``permute`` before the sparse gather; there is no perf
argument for channel-major."
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from core.game import Action, Game, State
from core.mcts import Evaluator

# Head reduction widths — 1×1 conv channels ahead of each head's FC,
# AlphaZero-style: two planes feed the flat policy FC, one plane each feeds
# the value/aux FCs. (The spatial policy head has no FC: its 1×1 conv emits
# the logits directly.)
_FLAT_POLICY_CHANNELS = 2
_SCALAR_HEAD_CHANNELS = 1


@dataclass(frozen=True)
class NetworkConfig:
    """Every dimension the D5 net needs, declared per game (§12 M2/M2.5).

    Attributes:
        input_planes: Number of input planes (Blokus: 46, D3).
        input_shape: ``(height, width)`` of every input plane — declared, never
            derived, and never assumed square (Connect 4 is 6×7); a flat policy
            head carries no geometry to recover it from (§6.1).
        policy_shape: The adapter's declared head shape — spatial ``(H, W, C)``
            like Blokus's ``(14, 14, 91)``, validated to match ``input_shape``
            in ``(H, W)``, or flat ``(K,)`` like Othello's ``(65,)``.
        trunk_blocks: Residual blocks in the trunk (D5: 8).
        trunk_channels: Trunk width (D5: 128).
        num_aux: Auxiliary value heads (Blokus: 1, the normalized score diff;
            Othello: 0 — §12 M1.5 pins "no aux head").
    """

    input_planes: int
    input_shape: tuple[int, int]
    policy_shape: tuple[int, ...]
    trunk_blocks: int = 8
    trunk_channels: int = 128
    num_aux: int = 0

    def __post_init__(self) -> None:
        """Rejects undeclared shapes loudly (§6.1 contract).

        Raises:
            ValueError: If ``input_shape`` is not a 2-tuple, ``policy_shape``
                is neither spatial ``(H, W, C)`` nor flat ``(K,)``, a spatial
                head disagrees with ``input_shape`` in ``(H, W)``, any
                dimension is nonpositive, or ``num_aux`` is negative.
        """
        if len(self.input_shape) != 2:
            raise ValueError(f"input_shape must be (height, width), got {self.input_shape!r}")
        if len(self.policy_shape) == 3:
            if tuple(self.policy_shape[:2]) != tuple(self.input_shape):
                raise ValueError(
                    f"spatial policy head {self.policy_shape} disagrees with "
                    f"input_shape {self.input_shape} in (H, W)"
                )
        elif len(self.policy_shape) != 1:
            raise ValueError(
                f"policy_shape must be spatial (H, W, C) or flat (K,), got {self.policy_shape!r}"
            )
        dims = (
            self.input_planes,
            *self.input_shape,
            *self.policy_shape,
            self.trunk_blocks,
            self.trunk_channels,
        )
        if any(d < 1 for d in dims) or self.num_aux < 0:
            raise ValueError(f"nonpositive dimension in {self}")

    @property
    def num_actions(self) -> int:
        """Total policy logits, ``prod(policy_shape)`` (Blokus: 17,836)."""
        return math.prod(self.policy_shape)

    @property
    def spatial_policy(self) -> bool:
        """Whether the policy head is spatial ``(H, W, C)`` rather than flat ``(K,)``."""
        return len(self.policy_shape) == 3

    @classmethod
    def from_game(cls, game: Game) -> NetworkConfig:
        """Builds the D5-default config from a game's declared encoding surface.

        Args:
            game: Adapter declaring ``input_planes``, ``input_shape``,
                ``policy_shape``, and ``value_targets`` (§6.1).

        Returns:
            A config carrying the game's dimensions and the D5 trunk defaults
            (Blokus: 46/(14, 14)/(14, 14, 91)/1 aux; Othello: 2/(8, 8)/(65,)/0).
        """
        return cls(
            input_planes=game.input_planes,
            input_shape=tuple(game.input_shape),
            policy_shape=tuple(game.policy_shape),
            num_aux=len(game.value_targets.aux_names),
        )


class ResidualBlock(nn.Module):
    """One trunk block: conv-BN-ReLU ×2 with identity skip (D5, AlphaZero-style)."""

    def __init__(self, channels: int) -> None:
        """Builds the block.

        Args:
            channels: Trunk width; input and output channel count alike.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the block; the ``(N, C, H, W)`` grid is preserved.

        Args:
            x: Trunk activations, ``(N, trunk_channels, H, W)``.

        Returns:
            Activations of the same shape.
        """
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(x + self.bn2(self.conv2(y)))


class Network(nn.Module):
    """The D5 policy/value/aux network: 3×3 conv stem, residual trunk, three heads.

    Heads (all raw — masking/renormalization is the sparse policy loss's job):

    * **Policy, spatial ``(H, W, C)``:** 1×1 conv to ``C`` channels, then the
      §5.1 HWC permute/flatten — flat index ``(r*W + c)*C + o`` matches
      ``games.blokus_duo.actions.encode``.
    * **Policy, flat ``(K,)``:** 1×1 conv reduction → FC → ``K`` logits, no
      permute — a flat head has no spatial factorization to preserve
      (Othello's pass id lives outside the board grid). Both branches end in
      the same ``(N, num_actions)`` logits contract.
    * **Value:** 1×1 conv → FC → scalar ``tanh`` (D1/D5).
    * **Aux:** built only when ``num_aux > 0`` (Othello declares none): 1×1
      conv → FC → ``num_aux`` linear outputs (Blokus target
      ``score_diff/109 ∈ [−1, 1]``, trained by MSE; tanh is not pinned for
      aux — linear is the direct reading of "normalized score-diff aux").
    """

    def __init__(self, config: NetworkConfig) -> None:
        """Builds the net; every dimension comes from ``config``.

        Args:
            config: Declared dimensions (see ``NetworkConfig.from_game``).
        """
        super().__init__()
        self.config = config
        h, w = config.input_shape
        ch = config.trunk_channels
        self.stem = nn.Sequential(
            nn.Conv2d(config.input_planes, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*(ResidualBlock(ch) for _ in range(config.trunk_blocks)))
        if config.spatial_policy:
            self.policy_conv = nn.Conv2d(ch, config.policy_shape[2], 1)
            self.policy_fc = None
        else:
            self.policy_conv = nn.Sequential(
                nn.Conv2d(ch, _FLAT_POLICY_CHANNELS, 1, bias=False),
                nn.BatchNorm2d(_FLAT_POLICY_CHANNELS),
                nn.ReLU(inplace=True),
            )
            self.policy_fc = nn.Linear(_FLAT_POLICY_CHANNELS * h * w, config.num_actions)
        self.value_conv = nn.Sequential(
            nn.Conv2d(ch, _SCALAR_HEAD_CHANNELS, 1, bias=False),
            nn.BatchNorm2d(_SCALAR_HEAD_CHANNELS),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Linear(_SCALAR_HEAD_CHANNELS * h * w, 1)
        if config.num_aux > 0:
            self.aux_conv = nn.Sequential(
                nn.Conv2d(ch, _SCALAR_HEAD_CHANNELS, 1, bias=False),
                nn.BatchNorm2d(_SCALAR_HEAD_CHANNELS),
                nn.ReLU(inplace=True),
            )
            self.aux_fc = nn.Linear(_SCALAR_HEAD_CHANNELS * h * w, config.num_aux)
        else:
            # No aux parameters at all (§12 M1.5: Othello pins "no aux head").
            self.aux_conv = None
            self.aux_fc = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Runs the trunk and all heads.

        Args:
            x: Float input batch, ``(N, input_planes, height, width)``.

        Returns:
            Tuple ``(policy_logits, value, aux)`` —
            ``policy_logits``: raw ``(N, num_actions)`` logits over the full
            head, no masking; for a spatial head, flat index
            ``(r*W + c)*C + o`` is the §5.1 cell-major action id.
            ``value``: ``(N,)`` scalars in ``[−1, 1]`` (``tanh``, D1).
            ``aux``: ``(N, num_aux)`` linear outputs, or ``None`` when the
            config declares no aux head (the pinned "absent" convention).
        """
        trunk = self.blocks(self.stem(x))
        if self.config.spatial_policy:
            # (N, C, H, W) → HWC → flat: the one §5.1 permute before the gather.
            pre = self.policy_conv(trunk)
            policy = pre.permute(0, 2, 3, 1).reshape(pre.shape[0], -1)
        else:
            policy = self.policy_fc(self.policy_conv(trunk).flatten(1))
        value = torch.tanh(self.value_fc(self.value_conv(trunk).flatten(1))).squeeze(-1)
        aux = None if self.aux_fc is None else self.aux_fc(self.aux_conv(trunk).flatten(1))
        return policy, value, aux


def make_network_evaluator(net: Network, game: Game, device: str = "cpu") -> Evaluator:
    """Bridge ``net`` into the ``MCTS.evaluate`` seam (§12 M0: "from M2, the network").

    The returned callable matches the M0 ``Evaluator`` seam exactly — the same
    seam, no new abstraction — so ``MCTS(game, evaluate=make_network_evaluator(...))``
    is the whole wiring. Batch-1 per-leaf inference is the M2/M3 functional
    path; batched/asynchronous inference (queueing across concurrent descents)
    is explicitly M5 scope and does not live here.

    **Value sign convention (the bug this docstring exists to prevent):** the
    scalar ``tanh`` head is returned as-is because it is *mover-relative by
    construction* — the §5.2 encoding is own/opponent from the side to move
    (no side-to-move plane), and training targets ``z`` are stored from the
    mover's perspective over that same encoding. That is precisely the seam's
    ``value_from_movers_perspective`` contract; the player-aware backup owns
    every sign flip from there. Any absolute-player (or opponent-relative)
    value returned here would corrupt search silently, not crash.

    **Priors are raw logits, not probabilities:** ``legal_moves`` yields flat
    action ids and the flat policy vector is indexed by action id (the §5.1
    flatten golden — the legal ids *are* the logit indices; nothing
    re-encodes), so the priors dict is ``{action_id: logit}`` with an entry
    for **every** legal id (``MCTS._priors`` defaults missing ids to logit
    ``0.0``). Only the legal logits are gathered on-device and transferred —
    never the dense head (sparse-everywhere; Blokus is 17,836 wide).
    ``MCTS._priors`` owns the single legal-subset softmax; normalizing here
    as well would softmax the policy twice and skew every prior toward
    uniform. ``uniform_prior=True`` on the engine (ladder rung 6) discards
    these priors while keeping the value.

    **Cross-wiring guard:** states are encoded with the factory-validated
    adapter — the pairing the net was checked against — never with the
    callback-time ``game``, and a callback-time ``game`` whose declared
    encoding surface disagrees with the validated one is rejected before any
    inference (``MCTS(game_b, evaluate=make_network_evaluator(net_a, game_a))``
    constructs fine — the evaluator is an opaque callable — so the first
    search call is the earliest loud failure). Delegating wrappers that
    preserve the surface (e.g. the runner's opening restriction) pass the
    guard: legal ids come from the callback-time ``game``, encoding from the
    validated adapter. An equal-surface *different* game is undetectable here
    (states are opaque); that pairing remains the caller's contract.

    Args:
        net: The network to evaluate leaves with. Moved to ``device`` and
            switched to ``eval()`` mode in place; forwards run under
            ``torch.inference_mode()``.
        game: The adapter ``net`` was built for — validated against
            ``net.config`` so a mismatched pairing fails loudly here instead
            of silently indexing the wrong logits.
        device: Torch device for inference (default ``"cpu"``).

    Returns:
        An ``Evaluator``: ``(game, state) -> (value, {action_id: raw_logit})``
        with the value from the mover's perspective.

    Raises:
        ValueError: If ``net.config`` disagrees with ``game``'s declared
            ``input_planes`` / ``input_shape`` / ``policy_shape``.
    """
    cfg = net.config
    adapter = game
    surface = (game.input_planes, tuple(game.input_shape), tuple(game.policy_shape))
    if (cfg.input_planes, cfg.input_shape, cfg.policy_shape) != surface:
        raise ValueError(
            f"net config ({cfg.input_planes}, {cfg.input_shape}, {cfg.policy_shape}) does "
            f"not match {type(game).__name__}'s declared encoding surface {surface}"
        )
    dev = torch.device(device)
    net.to(dev).eval()

    def evaluate(game: Game, state: State) -> tuple[float, dict[Action, float]]:
        if game is not adapter:
            declared = (game.input_planes, tuple(game.input_shape), tuple(game.policy_shape))
            if declared != surface:
                raise ValueError(
                    f"evaluator cross-wired: built for {type(adapter).__name__} with "
                    f"encoding surface {surface}, called with {type(game).__name__} "
                    f"declaring {declared}"
                )
        legal = list(game.legal_moves(state))
        x = torch.as_tensor(adapter.encode_state(state), dtype=torch.float32, device=dev)
        with torch.inference_mode():
            policy, value, _ = net(x.unsqueeze(0))
        idx = torch.as_tensor(legal, dtype=torch.long, device=dev)
        return value.item(), dict(zip(legal, policy[0, idx].tolist(), strict=True))

    return evaluate
