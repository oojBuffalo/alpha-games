"""Inference mixed precision as a run policy: ``core/amp.py`` (§7 D5 AMP, §12 M5).

The policy itself (:class:`~core.amp.AmpMode`, :func:`~core.amp.amp_evaluator`)
was hoisted out of ``scripts/run_micro.py``; ``tests/test_micro_throughput.py``
keeps pinning the behaviour the micro scripts rely on. This file pins what the
hoist adds: the scripts share the hoisted objects, the device-string form of
``resolve``, the exact CPU no-op on a real network evaluator, and the
first-call observation hook the production actor records from.

The falsifier is the same one ``tests/test_micro_throughput.py`` leans on: CPU
autocast is *real* (bfloat16), so a mode forced on while running on CPU
separates "what the forward actually ran under" from "what device it ran on".
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

import core.amp
from core.amp import AmpMode, amp_evaluator
from core.network import Network, NetworkConfig, make_network_evaluator
from games.tictactoe import TicTacToe

ROOT = Path(__file__).resolve().parent.parent

TTT = TicTacToe()
TTT_NET_CONFIG = NetworkConfig(2, (3, 3), (9,), trunk_blocks=1, trunk_channels=4)


def _load_script(name: str):
    """Import ``scripts/<name>.py`` as a module (``scripts/`` is not a package).

    Args:
        name: The script's module name.

    Returns:
        The loaded module.
    """
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)  # bench_micro_throughput imports its sibling
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", ["run_micro", "bench_micro_throughput"])
def test_the_micro_scripts_use_the_hoisted_policy_not_a_copy(script):
    module = _load_script(script)
    assert module.AmpMode is core.amp.AmpMode
    assert module.amp_evaluator is core.amp.amp_evaluator


def test_resolve_accepts_a_device_or_a_device_string():
    assert AmpMode.resolve("cpu") == AmpMode.resolve(torch.device("cpu")) == AmpMode("cpu", False)
    # No GPU is needed to resolve: the rule reads the device *type* only.
    assert (
        AmpMode.resolve("cuda:0") == AmpMode.resolve(torch.device("cuda")) == AmpMode("cuda", True)
    )


def test_on_cpu_the_wrapper_is_an_exact_no_op_on_a_real_evaluator():
    torch.manual_seed(0)
    bare = make_network_evaluator(Network(TTT_NET_CONFIG), TTT)
    wrapped = amp_evaluator(bare, AmpMode.resolve("cpu"))
    state = TTT.initial_state()
    for action in (4, 0, 8):
        assert wrapped(TTT, state) == bare(TTT, state)  # bit-identical, not merely close
        state = TTT.apply(state, action)


def test_observe_reports_the_live_state_once_from_inside_the_context():
    seen = []
    inside = []

    def fake(game, state):
        """A stand-in evaluator recording whether autocast was live for its 'forward'."""
        inside.append(torch.is_autocast_enabled("cpu"))
        return 0.5, {1: 0.25}

    forced = AmpMode(device_type="cpu", enabled=True)
    wrapped = amp_evaluator(fake, forced, observe=seen.append)
    assert seen == []  # nothing is observed until a forward actually runs
    assert wrapped(None, "s0") == (0.5, {1: 0.25})
    assert wrapped(None, "s1") == (0.5, {1: 0.25})
    assert seen == [True]  # once, and it is the state the forwards ran under
    assert inside == [True, True]

    seen.clear()
    off = amp_evaluator(fake, AmpMode.resolve("cpu"), observe=seen.append)
    off(None, "s")
    assert seen == [False]  # a disabled context is observed as not live


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a host without CUDA")
@pytest.mark.filterwarnings("ignore:CUDA is not available")
def test_observe_reports_torchs_state_not_the_modes_own_flag():
    """A CUDA mode on a CUDA-less host: the flag says on, torch disables the context."""
    seen = []
    claimed = AmpMode(device_type="cuda", enabled=True)
    amp_evaluator(lambda game, state: (0.0, None), claimed, observe=seen.append)(None, "s")
    assert claimed.enabled is True
    assert seen == [False]  # what happened, not what was asked for


def test_the_wrapper_passes_a_batch_of_states_through_untouched():
    """One wrapper serves both engine seams: the second argument is opaque to it."""
    calls = []

    def fake_batch(game, states):
        """A stand-in batched evaluator returning one result per state."""
        calls.append(states)
        return [(0.0, None)] * len(states)

    states = ["a", "b", "a"]  # equal states are legal within one batch
    wrapped = amp_evaluator(fake_batch, AmpMode.resolve("cpu"))
    assert wrapped(None, states) == [(0.0, None)] * 3
    assert calls == [states] and calls[0] is states


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_on_cuda_autocast_is_observed_live_inside_the_context():
    amp = AmpMode.resolve("cuda")
    assert amp.enabled is True
    assert amp.observed() is False  # outside the context
    seen = []
    evaluator = amp_evaluator(
        make_network_evaluator(Network(TTT_NET_CONFIG), TTT, device="cuda"),
        amp,
        observe=seen.append,
    )
    value, priors = evaluator(TTT, TTT.initial_state())
    assert seen == [True]
    assert -1.0 <= value <= 1.0 and len(priors) == 9
