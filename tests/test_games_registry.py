"""The games/-side game registry: ``games/registry.py`` (§Repo layout, issue #63).

``core.runconfig`` resolves a name pair to the adapter's own declared
instance-config *object*, without importing an adapter; this registry is the
next step, turning that object into an actual ``Game`` (or a picklable
factory for it). The battery: every adapter package under ``games/`` has
exactly one registry entry (no drift, either direction), an unknown name is
rejected loudly, the built factory produces a correctly-configured adapter
and survives a real pickle round trip (the exact contract
``core.ipc.launch_run``'s ``"spawn"`` context needs), and ``core/`` never
imports ``games/`` anywhere (the repo-wide, zero-``core/``-diff rule).
"""

from __future__ import annotations

import ast
import pickle
from pathlib import Path

import pytest

from core.runconfig import available_games, load_run_config
from games.blokus_duo import BlokusDuo
from games.blokus_duo.config import FULL_CONFIG, MICRO_CONFIG
from games.connect4 import Connect4
from games.othello import Othello
from games.registry import GAME_FACTORIES, build_game, build_game_factory, registered_games
from games.tictactoe import TicTacToe

ROOT = Path(__file__).resolve().parent.parent


def test_registry_covers_exactly_the_discovered_games():
    """No adapter package is missing a registry entry, and none is a stray."""
    assert set(registered_games()) == set(available_games())


def test_build_game_blokus_duo_uses_the_resolved_config():
    game = build_game("blokus_duo", MICRO_CONFIG)
    assert isinstance(game, BlokusDuo)
    assert game.config == MICRO_CONFIG

    game_full = build_game("blokus_duo", FULL_CONFIG)
    assert game_full.config == FULL_CONFIG


def test_build_game_reference_adapters_ignore_the_config_argument():
    assert isinstance(build_game("tictactoe", None), TicTacToe)
    assert isinstance(build_game("connect4", None), Connect4)
    assert isinstance(build_game("othello", None), Othello)


def test_build_game_unknown_name_raises():
    with pytest.raises(ValueError, match="no registry entry for game"):
        build_game("nonesuch", None)


def test_build_game_factory_resolves_the_micro_run_config():
    run_config = load_run_config()  # names blokus_duo / MICRO_CONFIG
    factory = build_game_factory(run_config)
    game = factory()
    assert isinstance(game, BlokusDuo)
    assert game.config == MICRO_CONFIG
    # Calling again builds a fresh instance, never a cached singleton --
    # exactly what core.ipc.launch_run needs per spawned process.
    assert factory() is not game


def test_build_game_factory_unknown_game_raises():
    """``core.runconfig.RunConfig`` can never itself carry an unresolvable game
    name (its own ``__post_init__`` already rejects one), so this exercises
    ``build_game_factory``'s own guard directly against a duck-typed
    stand-in -- the same technique ``tests/test_learner.py``/``tests/test_ipc.py``
    use for a run-config shape ``core.learner.LearnerDriver`` doesn't fully need.
    """
    from types import SimpleNamespace

    bogus = SimpleNamespace(game="nonesuch", resolve_game_config=lambda: None)
    with pytest.raises(ValueError, match="no registry entry for game"):
        build_game_factory(bogus)


def test_game_factories_are_module_level_and_picklable():
    """The exact contract ``core.ipc``'s spawn context needs (module docstring)."""
    run_config = load_run_config()
    factory = build_game_factory(run_config)
    restored = pickle.loads(pickle.dumps(factory))
    game = restored()
    assert isinstance(game, BlokusDuo)
    assert game.config == MICRO_CONFIG


def test_every_registered_builder_is_a_plain_module_level_function():
    for name, fn in GAME_FACTORIES.items():
        assert fn.__module__ == "games.registry", name
        assert fn.__qualname__.count(".") == 0, f"{name}: {fn.__qualname__} is not module-level"


# --- core/ must never import games/ -------------------------------------------


def _imports_games(path: Path) -> bool:
    """Return whether a Python file has a top-level ``import games`` or ``from games``.

    Args:
        path: The file to scan.

    Returns:
        ``True`` iff an AST ``Import``/``ImportFrom`` node names ``games`` (or
        a ``games.*`` submodule).
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "games" or alias.name.startswith("games.") for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and (
                node.module == "games" or node.module.startswith("games.")
            ):
                return True
    return False


def test_core_package_never_imports_games():
    offenders = [
        str(p.relative_to(ROOT)) for p in sorted((ROOT / "core").rglob("*.py")) if _imports_games(p)
    ]
    assert offenders == []
