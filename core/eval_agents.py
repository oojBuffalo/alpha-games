"""Checkpoint-backed evaluator load path + rung-5 network-policy agent (§9, M4).

Torch lives here (like ``core/network.py`` / ``core/checkpoint.py``) — the
pyproject confinement pin; ``core/agents.py`` stays pure stdlib and torch-free,
so :class:`NetworkPolicyAgent` sits beside the load path it depends on rather
than beside ``RandomAgent``/``MobilityAgent``.

Every network rung (5 here; 6/7 at a later task) stands on the same
load-bearing seam: reconstruct a published checkpoint's *exact* trained
architecture, restore its weights, validate its fingerprint, and wrap it as an
MCTS :data:`~core.mcts.Evaluator` — never a freshly initialized net wearing a
borrowed version label. That is the review-flagged failure mode this module
exists to close (P1): a checkpoint's fingerprint only pins *encoding-surface
agreement* (game identity, shapes, orientation hash) — it says nothing about
whether ``load_state_dict`` was ever actually called — so a load path that
skips step 3 below would build a randomly initialized net, label it with the
checkpoint's ``model_version``, and pass every identity check while every
downstream network rung silently measures a random net instead of the
checkpoint. :func:`load_eval_network` makes that step explicit and
un-skippable, and ``tests/test_eval_agents.py``'s distinct-weights golden
proves the weights actually moved.

**Reconstructing the architecture without a stored ``NetworkConfig`` field.**
``core.checkpoint.CheckpointBundle`` bundles ``run_config`` (``RunConfig``, no
network-shape fields — see ``core/runconfig.py``'s ``TrainingConfig``) and
``model_state_dict``, but no explicit ``NetworkConfig``: a real training run
always builds via ``NetworkConfig.from_game(game)`` and never persists the
trunk width/depth it chose. Re-deriving via ``from_game`` here would silently
assume every checkpoint used the D5 default trunk (8 blocks × 128 channels) —
correct for production checkpoints, but exactly the "config the checkpoint
did *not* actually train with" for any other trunk size (small test
checkpoints included), and `` load_state_dict(strict=True)`` would then raise
on the first shape mismatch instead of loading. So the trunk width/depth
(and whether an aux head exists) are read directly off the persisted
``model_state_dict`` tensors — the one artifact that cannot drift from what
was actually trained — while the game-shape fields (``input_planes``,
``input_shape``, ``policy_shape``) come from ``game``, already pinned equal to
the checkpoint's by :func:`~core.checkpoint.load_checkpoint`'s fingerprint
compare. This is the literal reading of "the checkpoint is the authority on
the architecture it trained," not a re-derivation of D5 defaults.
"""

from __future__ import annotations

from pathlib import Path

from core.agents import Agent
from core.checkpoint import CheckpointBundle, load_checkpoint
from core.game import Action, Game, State
from core.mcts import Evaluator
from core.network import Network, NetworkConfig, make_network_evaluator
from core.runner import AgentFactory

_STEM_CONV_WEIGHT = "stem.0.weight"
_AUX_FC_WEIGHT = "aux_fc.weight"
_BLOCK_KEY_PREFIX = "blocks."


def _trunk_shape_from_state_dict(state_dict: dict) -> tuple[int, int]:
    """Read ``(trunk_blocks, trunk_channels)`` off a saved ``model_state_dict``.

    The stem conv's output channel count is the trunk width directly
    (``core.network.Network.__init__``'s ``stem`` is
    ``Conv2d(input_planes, trunk_channels, ...)``); the trunk depth is the
    count of distinct residual-block indices present in the flat state-dict
    keys (``blocks.<i>.conv1.weight`` etc., one index per
    ``core.network.ResidualBlock``).

    Args:
        state_dict: A ``Network.state_dict()``-shaped mapping (bundle
            ``model_state_dict``).

    Returns:
        ``(trunk_blocks, trunk_channels)``.
    """
    trunk_channels = int(state_dict[_STEM_CONV_WEIGHT].shape[0])
    block_indices = {key.split(".")[1] for key in state_dict if key.startswith(_BLOCK_KEY_PREFIX)}
    trunk_blocks = len(block_indices)
    return trunk_blocks, trunk_channels


def _num_aux_from_state_dict(state_dict: dict) -> int:
    """Read the declared aux-head width off a saved ``model_state_dict``.

    Args:
        state_dict: A ``Network.state_dict()``-shaped mapping.

    Returns:
        ``aux_fc.weight``'s output width if the key is present (an aux head
        was built), else ``0`` (``core.network.Network`` builds no aux
        parameters at all when ``num_aux == 0`` — the pinned "absent"
        convention, mirrored here).
    """
    if _AUX_FC_WEIGHT not in state_dict:
        return 0
    return int(state_dict[_AUX_FC_WEIGHT].shape[0])


def _network_config_from_bundle(bundle: CheckpointBundle, game: Game) -> NetworkConfig:
    """Reconstruct the exact trained :class:`~core.network.NetworkConfig`.

    See the module docstring for why this reads the trunk shape off the
    persisted weights rather than calling ``NetworkConfig.from_game(game)``.

    Args:
        bundle: A fingerprint-validated bundle
            (:func:`~core.checkpoint.load_checkpoint`'s return).
        game: The adapter the bundle was validated against — its declared
            ``input_planes``/``input_shape``/``policy_shape`` are already
            pinned equal to the checkpoint's by that validation.

    Returns:
        The config that reproduces the checkpoint's exact tensor shapes.
    """
    trunk_blocks, trunk_channels = _trunk_shape_from_state_dict(bundle.model_state_dict)
    return NetworkConfig(
        input_planes=game.input_planes,
        input_shape=tuple(game.input_shape),
        policy_shape=tuple(game.policy_shape),
        trunk_blocks=trunk_blocks,
        trunk_channels=trunk_channels,
        num_aux=_num_aux_from_state_dict(bundle.model_state_dict),
    )


def load_eval_network(path: Path | str, game: Game, device: str = "cpu") -> tuple[Evaluator, int]:
    """Load a published checkpoint into a ready-to-search MCTS evaluator.

    The full load contract (review P1), no step implicit:

    1. :func:`core.checkpoint.load_checkpoint` — the loader recomputes and
       compares the full artifact fingerprint (orientation hash included)
       against ``game``'s live one and fails loudly on any disagreement.
       This function never re-does that validation.
    2. Rebuild the network architecture from the bundle's persisted weights
       (:func:`_network_config_from_bundle`) — never ``NetworkConfig.from_game``
       defaults: the checkpoint is the authority on the architecture it
       trained.
    3. ``net.load_state_dict(bundle.model_state_dict, strict=True)`` — strict,
       so any key or shape drift between the rebuilt architecture and the
       stored weights raises immediately instead of silently dropping or
       padding tensors.
    4. Hand ``net`` to :func:`core.network.make_network_evaluator` — it moves
       the net to ``device``, switches it to ``eval()`` mode, and installs the
       encoding-surface cross-wiring guard.
    5. Return the evaluator plus the bundle's ``version`` — the model-version
       ordinal every rung-5/6/7 identity string is parameterized by.

    Inference stays the evaluator's existing batch-1 ``torch.inference_mode()``
    path; batched inference is M5, out of scope here.

    Args:
        path: The checkpoint file to load (a published ``ckpt-<version>.pt``
            or the rolling ``resume.pt`` — see ``core.checkpoint``).
        game: The adapter this checkpoint was trained against — validated by
            ``load_checkpoint`` before anything else here runs, and reused as
            the evaluator's factory-validated pairing.
        device: Torch device for inference (default ``"cpu"``).

    Returns:
        ``(evaluator, model_version)``: an ``Evaluator`` ready for
        ``MCTS(game, evaluate=evaluator)`` or a rung-5 agent, and the
        checkpoint's model-version ordinal for identity strings.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        core.checkpoint.CheckpointFormatError: If the payload is malformed or
            carries an unsupported schema version.
        core.artifact_fingerprint.FingerprintMismatchError: If the stored
            fingerprint disagrees with ``game``'s live one on any field.
        RuntimeError: If the reconstructed architecture's keys/shapes
            disagree with ``bundle.model_state_dict``
            (``load_state_dict(strict=True)``).
    """
    bundle = load_checkpoint(path, game)  # step 1: fingerprint validated or raised
    config = _network_config_from_bundle(bundle, game)  # step 2: from the bundle, not from_game
    net = Network(config)
    net.load_state_dict(bundle.model_state_dict, strict=True)  # step 3: strict restore
    evaluator = make_network_evaluator(net, game, device)  # step 4: device/eval/guard
    return evaluator, bundle.version  # step 5


class NetworkPolicyAgent(Agent):
    """Ladder rung 5: the checkpoint's raw policy, no search (§9).

    One evaluator call per move; argmax over the returned legal-action logit
    dict. The evaluator returns raw logits gathered over legal ids only
    (``core.network.make_network_evaluator``'s contract); softmax is
    monotonic, so argmax over legal logits *is* the masked-softmax argmax —
    no softmax computation is needed to select the move. Ties break to the
    lowest action id (the ``MCTS.best_action`` / ``core.selfplay`` convention).
    No search, no temperature, no noise, no RNG: deterministic by
    construction, per the §9 protocol.

    Args:
        evaluator: An ``Evaluator`` from :func:`load_eval_network` (or any
            evaluator matching that contract) — shared across every agent
            built from the same checkpoint (see :func:`rung5_agent_factory`).
        model_version: The checkpoint's model-version ordinal, for
            :attr:`name`.
    """

    def __init__(self, evaluator: Evaluator, model_version: int):
        self._evaluator = evaluator
        self._name = f"rung5-v1-{model_version}"

    @property
    def name(self) -> str:
        return self._name

    def select_action(self, game: Game, state: State) -> Action:
        _, priors = self._evaluator(game, state)
        # Iterate ascending ids so the first (i.e. lowest-id) maximal logit
        # wins ties -- max() keeps the first item achieving its running
        # maximum, never a later tied one.
        return max(sorted(priors), key=priors.get)


def rung5_agent_factory(path: Path | str, game: Game, device: str = "cpu") -> AgentFactory:
    """Build a ``core.runner.AgentFactory`` sharing one loaded checkpoint.

    The intended factory shape (agents are rebuilt per game by
    ``core.runner.play_pairs`` via ``AgentFactory``): loading a checkpoint per
    game would be wasteful, so :func:`load_eval_network` runs exactly once,
    here, outside the returned closure; the closure itself only constructs
    the lightweight :class:`NetworkPolicyAgent` wrapper per call, and every
    call shares the one loaded net/evaluator.

    Args:
        path: The checkpoint file to load once.
        game: The adapter to load against — also the game the returned
            factory should be used with (e.g. passed to
            ``core.runner.play_pairs`` alongside this factory).
        device: Torch device for inference.

    Returns:
        A ``seed -> NetworkPolicyAgent`` factory matching
        ``core.runner.AgentFactory``'s shape. Rung 5 is deterministic and
        consumes no RNG, so the seed argument is accepted (for the shared
        factory signature) and otherwise unused; every call returns an agent
        sharing the one evaluator loaded above.
    """
    evaluator, model_version = load_eval_network(path, game, device)

    def factory(seed: int) -> NetworkPolicyAgent:
        del seed  # rung 5 has no per-agent RNG state
        return NetworkPolicyAgent(evaluator, model_version)

    return factory
