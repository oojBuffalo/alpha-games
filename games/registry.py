"""Game-name -> constructor registry (design doc §Repo layout; issue #63).

``core.runconfig`` resolves a run config's ``(game, game_config)`` name pair to
the adapter's own declared instance-config *object* without ever importing an
adapter package itself -- see that module's docstring. This module is the
next, adapter-owning step: turning a resolved instance-config object into an
actual :class:`core.game.Game`, or, for :func:`core.ipc.launch_run`'s process
boundary, a *picklable factory* that builds one on demand. It is a thin
``games/``-side lookup, never imported by anything under ``core/`` -- the CLI
script layer (``scripts/run_selfplay.py``) is the only caller that needs both
``core.runconfig`` (name -> config object) and this module (config object ->
``Game``) at once.

Every entry in :data:`GAME_FACTORIES` is a **module-level function**, never a
lambda or closure: :func:`core.ipc.launch_run` spawns actor/learner processes
under the ``"spawn"`` multiprocessing context, and a spawned child resolves a
``functools.partial(fn, config)`` target by ``fn``'s qualified module path --
a lambda has none and is not picklable (``core.ipc``'s module docstring).
``functools.partial`` around a module-level function plus a picklable config
object (a frozen dataclass, or ``None``) satisfies that contract directly.

Adding a game means adding one adapter package under ``games/`` plus one entry
here -- still a ``games/``-only diff, never a ``core/`` one.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from core.game import Game
from core.runconfig import RunConfig, available_games
from games.blokus_duo import BlokusDuo
from games.blokus_duo.config import BlokusConfig
from games.connect4 import Connect4
from games.othello import Othello
from games.tictactoe import TicTacToe

# A zero-argument, picklable callable that builds a fresh adapter instance --
# exactly ``core.ipc.GameFactory``, repeated here rather than imported so this
# module never needs to import ``core.ipc`` just for a type alias.
GameFactory = Callable[[], Game]


def _build_blokus_duo(config: BlokusConfig) -> Game:
    """Build a :class:`~games.blokus_duo.game.BlokusDuo` over a resolved instance.

    Args:
        config: The resolved :class:`~games.blokus_duo.config.BlokusConfig`
            (``core.runconfig.resolve_game_config("blokus_duo", ...)``).

    Returns:
        The constructed adapter (the production bitboard engine).
    """
    return BlokusDuo(config=config)


def _build_tictactoe(config: Any) -> Game:
    """Build a :class:`~games.tictactoe.game.TicTacToe`.

    Args:
        config: Unused -- Tic-Tac-Toe declares no named instance configs
            (``core.runconfig``'s ``GAME_CONFIGS`` convention), so this is
            always ``None`` in practice; accepted rather than rejected so the
            registry's per-game call shape stays uniform.

    Returns:
        The constructed adapter.
    """
    del config
    return TicTacToe()


def _build_connect4(config: Any) -> Game:
    """Build a :class:`~games.connect4.game.Connect4` (standard 6x7, connect 4).

    Args:
        config: Unused -- Connect Four declares no named instance configs.

    Returns:
        The constructed adapter.
    """
    del config
    return Connect4()


def _build_othello(config: Any) -> Game:
    """Build a :class:`~games.othello.game.Othello` (standard 8x8).

    Args:
        config: Unused -- Othello declares no named instance configs.

    Returns:
        The constructed adapter.
    """
    del config
    return Othello()


# name -> (resolved instance-config object -> Game). Every adapter package
# under ``games/`` gets exactly one entry; ``tests/test_games_registry.py``
# asserts this set equals ``core.runconfig.available_games()`` so a new
# adapter package that forgets to register here fails a test, not silently.
GAME_FACTORIES: dict[str, Callable[[Any], Game]] = {
    "blokus_duo": _build_blokus_duo,
    "connect4": _build_connect4,
    "othello": _build_othello,
    "tictactoe": _build_tictactoe,
}


def registered_games() -> tuple[str, ...]:
    """Return every game name this registry can build, sorted.

    Returns:
        The sorted keys of :data:`GAME_FACTORIES`.
    """
    return tuple(sorted(GAME_FACTORIES))


def build_game(game: str, game_config_obj: Any) -> Game:
    """Build one adapter instance directly from a name and a resolved config.

    Args:
        game: An adapter package name, e.g. ``"blokus_duo"``.
        game_config_obj: The resolved instance-config object
            (``core.runconfig.resolve_game_config(game, game_config)``), or
            ``None`` for a game that declares no named configs.

    Returns:
        The constructed adapter.

    Raises:
        ValueError: If ``game`` has no registry entry.
    """
    try:
        build = GAME_FACTORIES[game]
    except KeyError:
        raise ValueError(
            f"no registry entry for game {game!r}; known games are {list(registered_games())}"
        ) from None
    return build(game_config_obj)


def build_game_factory(run_config: RunConfig) -> GameFactory:
    """Build the picklable, zero-argument factory a run config names.

    Resolves ``run_config.game_config`` through ``run_config.resolve_game_config()``
    (adapter-declared, per ``core.runconfig``) and pairs it with this
    registry's builder as a ``functools.partial`` -- picklable under the
    ``"spawn"`` multiprocessing context (module docstring), so this is safe to
    pass straight into :func:`core.ipc.launch_run`'s ``game_factory`` argument.

    Args:
        run_config: The run's full protocol.

    Returns:
        A zero-argument callable that builds a fresh adapter instance.

    Raises:
        ValueError: If ``run_config.game`` has no registry entry, or names an
            adapter/config pair ``core.runconfig`` cannot resolve
            (unreachable for a ``run_config`` that already passed its own
            ``__post_init__``).
    """
    if run_config.game not in GAME_FACTORIES:
        raise ValueError(
            f"no registry entry for game {run_config.game!r}; "
            f"known games are {list(registered_games())}"
        )
    build = GAME_FACTORIES[run_config.game]
    resolved = run_config.resolve_game_config()
    return functools.partial(build, resolved)


# A loud, import-time consistency check rather than a runtime surprise: every
# package ``core.runconfig.available_games()`` discovers under ``games/`` must
# have a registry entry here (games/-only diffs must include this file), and
# this module names no game the filesystem doesn't have.
_discovered = set(available_games())
_registered = set(GAME_FACTORIES)
if _discovered != _registered:
    raise RuntimeError(
        "games.registry is out of sync with games/: "
        f"discovered={sorted(_discovered)} registered={sorted(_registered)}"
    )
