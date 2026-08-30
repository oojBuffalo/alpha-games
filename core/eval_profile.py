"""A game's declared network-free eval ladder (design doc §9, §12 M1.6/M4).

The M4 orchestrator (``core.eval_run``) schedules and plays cells against the
frozen network-free ladder (rungs 1-4 in v1) and, where the game has one, an
opening-balance hook for the mirrored-pair runner (``core.runner.play_pairs``'s
``opening_balancer`` argument) -- but the orchestrator itself must never name a
game-specific agent class or balancer function (``core/`` never imports
``games.*``, design doc §Repo layout). :class:`EvalProfile` is the seam that
closes that gap: each adapter package builds exactly one frozen instance (e.g.
``games/blokus_duo``'s, registering ``core.agents.RandomAgent``,
``games.blokus_duo.baselines.LargestPieceAgent``, ``core.agents.MobilityAgent``,
and ``core.uct.UCTAgent`` at rungs 1-4, plus its ``start_square_balancer``) and
exposes it through ``games/registry.py``, exactly the way ``games.registry``
already exposes a picklable ``Game`` factory per adapter for
``core.ipc.launch_run``. Adding (or reducing) a game's eval ladder is therefore
a ``games/``-only diff -- the second-game fixture in
``tests/test_eval_orchestrator.py`` proves this by building a whole second
:class:`EvalProfile`, for a tiny stub game defined entirely inside the test,
with zero edits to this module or to ``core/eval_run.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from core.runner import AgentFactory, OpeningBalancer


@dataclass(frozen=True)
class EvalProfile:
    """One game's frozen network-free eval ladder + opening-balancer hook.

    **The one contract a registered rung must satisfy, and why (review-grade**
    **load-bearing detail):** the agent a rung's factory builds must report a
    ``.name`` that does not depend on the seed the factory was called with.
    Every rung-1..4 baseline already satisfies this by construction
    (``core.agents.RandomAgent``/``MobilityAgent``, a game's own baselines,
    ``core.uct.UCTAgent`` all return a fixed string from ``name`` regardless of
    their constructor's ``seed`` argument) -- documented here as a requirement
    rather than assumed silently, because :meth:`rung_identity` reads it
    *before* any per-cell seed exists: a cell's id
    (``core.eval_store.build_cell_id``) is built from the opponent's identity
    string, and the eval orchestrator in turn derives that per-cell seed from
    the finished cell id (``derive_seed(eval_seed, PURPOSE_EVAL, cell_id)``) --
    a rung whose identity depended on the seed would make the cell id and the
    seed used to reach it mutually circular.

    Attributes:
        network_free_rungs: Mapping of frozen ladder rung id (1-4 in v1, per
            design doc §12 M1.6's convention pins) to the
            ``core.runner.AgentFactory`` that builds that rung's agent. Never
            empty -- a game with no network-free ladder at all has nothing for
            forms 5/6/7 to play and cannot use this harness.
        opening_balancer: The game's opening-balance hook
            (``core.runner.play_pairs``'s ``opening_balancer`` argument), or
            ``None`` for a game with no start-square (or equivalent) asymmetry
            to balance -- the mirrored pair then plays unbalanced, which is
            correct rather than a missing feature for such a game.
    """

    network_free_rungs: Mapping[int, AgentFactory]
    opening_balancer: OpeningBalancer | None = None

    def __post_init__(self) -> None:
        """Validate the declared rung ids.

        Raises:
            ValueError: If ``network_free_rungs`` is empty, or any key is not
                a positive ``int`` (``bool`` rejected: it is an ``int``
                subclass and reads as a flag, never as a rung id).
        """
        if not self.network_free_rungs:
            raise ValueError("network_free_rungs must be non-empty")
        bad = [
            rung
            for rung in self.network_free_rungs
            if isinstance(rung, bool) or not isinstance(rung, int) or rung < 1
        ]
        if bad:
            raise ValueError(f"network_free_rungs keys must be positive ints, got {bad!r}")

    def rungs(self) -> tuple[int, ...]:
        """Return the declared rung ids, ascending.

        Returns:
            The sorted keys of :attr:`network_free_rungs`.
        """
        return tuple(sorted(self.network_free_rungs))

    def rung_identity(self, rung: int) -> str:
        """Return one declared rung's fixed agent identity string.

        Builds the rung's agent via its factory (seed ``0`` -- inert, per the
        class docstring's seed-independence contract) purely to read
        ``.name``; the returned agent itself is discarded, never reused for
        actual play (a real game call site builds its own agent through the
        same factory, seeded per :func:`core.eval_run.cell_seed`).

        Args:
            rung: A rung id declared in :attr:`network_free_rungs`.

        Returns:
            The rung's agent identity string (e.g. ``"random"``).

        Raises:
            ValueError: If ``rung`` is not declared by this profile.
        """
        try:
            factory = self.network_free_rungs[rung]
        except KeyError:
            raise ValueError(
                f"profile declares no network-free rung {rung}; declared rungs are "
                f"{list(self.rungs())}"
            ) from None
        return factory(0).name
