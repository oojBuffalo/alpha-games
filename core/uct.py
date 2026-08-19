"""Ladder rung 4: UCT (UCB1) with uniform-random rollouts (design doc §9, §12 M3).

A standalone tree-search opponent — *not* a configuration of the engine's PUCT
search (``core/mcts.py``). Uniform priors do not turn PUCT into UCB1's selection
rule, and rung 4 gets its value signal from a rollout rather than the network
seam PUCT reserves for M2; the M0 ``uniform_prior`` flag on ``MCTS`` is a
different rung (6) entirely. This module is the whole algorithm: UCB1 selection,
uniform-random rollouts to a terminal, and the same player-aware backup
convention as the main engine — ``edge_value = value if edge.parent_player ==
ref_player else -value``, generalized through ``terminal_utility`` so
non-alternating pass games (a player moving twice in a row) are handled
correctly, exactly as in ``core/mcts.py``.

:class:`UCTSearcher` is the parameterizable, testable algorithm — simulation
count, ``c``, and the RNG are constructor arguments so the algorithm can be
exercised at reduced cost in tests. :class:`UCTAgent` freezes it as ladder rung
4: 1,000 simulations/move, ``c = sqrt(2)``, the untried-first/lowest-action-id
UCB1 tie-break, and the uniform-random rollout policy, all pinned under a
versioned identity string (:attr:`UCTAgent.RUNG_ID`) so any future change to
those constants is a new rung version, never a silent edit of a frozen
baseline — a matching :class:`~core.agents.Agent` wrapper so the existing
paired runner and Elo scaffolding (``core/runner.py``, ``core/elo.py``)
schedule it exactly like the earlier ladder rungs.
"""

from __future__ import annotations

import math
import random

from core.agents import Agent
from core.game import Action, Game, PlayerId, State, assert_v1_envelope

#: UCT's canonical exploration constant (Kocsis & Szepesvári, 2006).
DEFAULT_C = math.sqrt(2.0)


class _UCTNode:
    """One search-tree node: a state plus sparse per-legal-action edge statistics.

    Mirrors ``core/mcts.py``'s ``_Node`` shape — sparse ``{N, W, Q}`` over legal
    actions, lazily materialized children — minus the PUCT-specific fields
    (priors, virtual loss) this single-threaded algorithm has no use for.
    """

    __slots__ = (
        "state",
        "is_terminal",
        "to_play",
        "actions",
        "N",
        "W",
        "Q",
        "children",
        "is_expanded",
    )

    def __init__(self, state: State, is_terminal: bool, to_play: PlayerId | None):
        self.state = state
        self.is_terminal = is_terminal
        # to_play is the mover at this node; None only at terminal nodes (which are
        # always leaves and never provide a parent-mover on any backup path).
        self.to_play = to_play
        self.actions: list[Action] = []
        self.N: list[int] = []
        self.W: list[float] = []
        self.Q: list[float] = []
        self.children: list[_UCTNode | None] = []
        self.is_expanded = False


class UCTSearcher:
    """Standalone UCB1 tree search with uniform-random rollouts (design doc §9).

    One instance searches a fresh tree per :meth:`search` call — no subtree
    reuse between moves (unlike ``core/mcts.py``'s ``MCTS``): rung 4 is a much
    cheaper baseline than the network-backed rungs, and a fresh tree keeps its
    behavior simple to reason about and to freeze.

    Args:
        game: The adapter to search. Validated against the v1 envelope on
            construction.
        num_simulations: Number of simulations (one rollout each) to run per
            :meth:`search` call.
        rng: Source of all randomness (rollout move choice) — determinism is
            entirely a function of this stream, so callers own seeding.
        c: UCB1 exploration constant (default ``sqrt(2)``, the canonical value).

    Raises:
        EnvelopeError: If ``game`` declares capabilities outside the v1 envelope.
    """

    def __init__(
        self,
        game: Game,
        *,
        num_simulations: int,
        rng: random.Random,
        c: float = DEFAULT_C,
    ):
        assert_v1_envelope(game)
        self.game = game
        self.num_simulations = num_simulations
        self.rng = rng
        self.c = c
        self.root: _UCTNode | None = None

    # --- public API --------------------------------------------------------

    def search(self, state: State) -> Action:
        """Run ``num_simulations`` simulations from a fresh tree rooted at ``state``.

        Args:
            state: The nonterminal state to search from.

        Returns:
            The most-visited root action, ties broken by lowest action id.

        Raises:
            ValueError: If ``state`` is terminal.
        """
        self.root = self._make_node(state)
        if self.root.is_terminal:
            raise ValueError("cannot search from a terminal state")
        for _ in range(self.num_simulations):
            path, leaf = self._descend()
            ref_player, value = self._evaluate_leaf(path, leaf)
            self._backup(path, ref_player, value)
        return self.best_action()

    def action_visit_counts(self, node: _UCTNode | None = None) -> dict[Action, int]:
        """Return ``{action_id: visit_count}`` for the node's edges (default: root)."""
        node = node or self.root
        if node is None:
            raise ValueError("no node")
        return dict(zip(node.actions, node.N, strict=True))

    def action_values(self, node: _UCTNode | None = None) -> dict[Action, float]:
        """Return ``{action_id: Q}`` in the mover's perspective (default: root)."""
        node = node or self.root
        if node is None:
            raise ValueError("no node")
        return dict(zip(node.actions, node.Q, strict=True))

    def best_action(self, node: _UCTNode | None = None) -> Action:
        """Return the most-visited action, ties broken by lowest action id."""
        node = node or self.root
        if node is None or not node.actions:
            raise ValueError("no expanded node to choose from")
        best_i = 0
        for i in range(1, len(node.actions)):
            # Most visits wins; ties go to the lowest action id (not adapter/index order) —
            # mirrors core/mcts.py's MCTS.best_action rule exactly.
            if node.N[i] > node.N[best_i] or (
                node.N[i] == node.N[best_i] and node.actions[i] < node.actions[best_i]
            ):
                best_i = i
        return node.actions[best_i]

    # --- internals -----------------------------------------------------------

    def _make_node(self, state: State) -> _UCTNode:
        terminal = self.game.is_terminal(state)
        to_play = None if terminal else self.game.current_player(state)
        return _UCTNode(state, terminal, to_play)

    def _expand(self, node: _UCTNode) -> None:
        """Attach sparse edges to ``node`` (no priors — UCB1 needs none)."""
        actions = list(self.game.legal_moves(node.state))
        n = len(actions)
        node.actions = actions
        node.N = [0] * n
        node.W = [0.0] * n
        node.Q = [0.0] * n
        node.children = [None] * n
        node.is_expanded = True

    def _select_edge(self, node: _UCTNode) -> int:
        """Return the local edge index to descend: UCB1's selection rule (pinned exactly).

        Untried children (``N == 0``) are visited before any UCB1 comparison, in
        lowest-action-id order. Once every child has ``N >= 1``, maximize
        ``Q(a) + c * sqrt(ln(sum(N)) / N(a))`` over the parent's total child
        visits, ties broken by lowest action id.
        """
        order = sorted(range(len(node.actions)), key=lambda i: node.actions[i])
        for i in order:
            if node.N[i] == 0:
                return i
        log_total = math.log(sum(node.N))
        best_i = order[0]
        best_score = node.Q[best_i] + self.c * math.sqrt(log_total / node.N[best_i])
        for i in order[1:]:
            score = node.Q[i] + self.c * math.sqrt(log_total / node.N[i])
            if score > best_score:  # strict: first (lowest-id, by `order`) max wins ties
                best_i, best_score = i, score
        return best_i

    def _descend(self) -> tuple[list[tuple[_UCTNode, int]], _UCTNode]:
        """Select from the root to this simulation's new leaf.

        Exactly one node is added to the tree per simulation: descent follows
        already-visited edges until it reaches either a pre-existing terminal, or
        the first untried edge, whose child is materialized (and returned as the
        leaf) without being visited any further this simulation.

        Returns:
            ``(path, leaf)``: ``path`` is the list of ``(node, edge_index)``
            traversed; ``leaf`` is the node to evaluate (rollout or
            ``terminal_utility``).
        """
        path: list[tuple[_UCTNode, int]] = []
        node = self.root
        assert node is not None
        while True:
            if node.is_terminal:
                return path, node
            if not node.is_expanded:
                self._expand(node)
            i = self._select_edge(node)
            path.append((node, i))
            child = node.children[i]
            if child is None:
                child = self._make_node(self.game.apply(node.state, node.actions[i]))
                node.children[i] = child
            if node.N[i] == 0:
                return path, child
            node = child

    def _evaluate_leaf(
        self, path: list[tuple[_UCTNode, int]], leaf: _UCTNode
    ) -> tuple[PlayerId | None, float]:
        """Return ``(ref_player, value)`` for backup — rollout, or ``terminal_utility``.

        For a terminal leaf, the value is measured from the perspective of the
        player who just moved into it (the parent node's mover) — a terminal
        state's own mover is contractually meaningless. For a freshly-added
        nonterminal leaf, the value is the rollout's terminal utility from the
        leaf's own mover's perspective.
        """
        if leaf.is_terminal:
            if not path:
                return None, 0.0  # degenerate: root itself terminal, contributes nothing
            parent, _ = path[-1]
            ref = parent.to_play
            return ref, self.game.terminal_utility(leaf.state, ref)
        ref = leaf.to_play
        return ref, self._rollout(leaf.state, ref)

    def _rollout(self, state: State, ref_player: PlayerId) -> float:
        """Play uniform-random legal moves from ``state`` to a terminal.

        Args:
            state: The nonterminal state to roll out from.
            ref_player: The player whose terminal utility to return.

        Returns:
            ``terminal_utility(terminal_state, ref_player)`` — read once at the
            rollout's terminal, never negated blindly through its plies.
        """
        s = state
        while not self.game.is_terminal(s):
            moves = list(self.game.legal_moves(s))
            a = self.rng.choice(moves)
            s = self.game.apply(s, a)
        return self.game.terminal_utility(s, ref_player)

    def _backup(
        self, path: list[tuple[_UCTNode, int]], ref_player: PlayerId | None, value: float
    ) -> None:
        """Propagate ``value`` up ``path`` (design doc §6.2's edge-relative sign rule).

        Edge-relative sign rule: ``edge_value = value`` if the edge's node moved
        as ``ref_player`` else ``-value``. Because each edge reads its own node's
        mover, this is correct across consecutive same-player moves.
        """
        for node, i in path:
            node.N[i] += 1
            edge_value = value if node.to_play == ref_player else -value
            node.W[i] += edge_value
            node.Q[i] = node.W[i] / node.N[i]


class UCTAgent(Agent):
    """Ladder rung 4: UCT + uniform-random rollouts (design doc §9, §12 M3).

    Frozen constants — 1,000 simulations/move, ``c = sqrt(2)``, the untried-
    first/lowest-action-id UCB1 tie-break (:meth:`UCTSearcher._select_edge`),
    and the uniform-random rollout policy — are pinned here under a versioned
    identity (:attr:`RUNG_ID`), never exposed as configuration: a frozen rung's
    strength must mean one fixed algorithm forever. Any future change to these
    constants is a new rung version (bump ``RUNG_ID``), never an edit of this
    class.

    Args:
        seed: Seed for the agent's private RNG stream (search + rollout
            randomness; the runner reseeds per pair for reproducibility).
    """

    #: Frozen ladder rung-4 identity; a golden test pins this exact string so a
    #: silent change to the constants below fails a test instead of silently
    #: redefining the baseline.
    RUNG_ID = "uct-rollout-v1"
    NUM_SIMULATIONS = 1000
    C = DEFAULT_C

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return self.RUNG_ID

    def select_action(self, game: Game, state: State) -> Action:
        searcher = UCTSearcher(game, num_simulations=self.NUM_SIMULATIONS, rng=self._rng, c=self.C)
        return searcher.search(state)
