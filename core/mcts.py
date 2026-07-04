"""Sparse, player-aware PUCT MCTS (design doc §6.2).

The engine is game-generic: it drives any ``Game`` through selection / expansion /
backup. There is **no leaf-evaluator abstraction** (design doc §12, M0): a leaf's
value is ``terminal_utility`` at terminals and, from M2, a network — supplied here as
an optional ``evaluate`` callable, not a swappable class hierarchy. In M0 no evaluator
is supplied: non-terminal leaves take value ``0.0`` (the "draw" prior, consistent with
first-play-urgency ``Q = 0``) and uniform priors. Ladder rung 6 ("uniform-prior MCTS
with network value") is the ``uniform_prior`` flag on this same engine; rung 4
(UCT + rollouts) is a separate standalone opponent built at M3, not a mode of this class.

Key invariants (never violated):
  * Sparse ``{N, W, Q, P}`` over legal actions only; children allocated lazily.
  * Player-aware backup is *edge-relative* — each edge reads its own node's mover, so
    the sign is correct even when one player moves consecutively (a blocked opponent).
    ``Q`` is therefore stored in the parent-mover's perspective and PUCT needs no flip.
  * Virtual loss is applied on descent and removed on backup, so a completed search
    leaves zero residue; it exists so batched selection (M5) never re-picks an
    in-flight leaf.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from core.game import Action, Game, PlayerId, State, assert_v1_envelope

# evaluate(game, state) -> (value_from_movers_perspective, priors_by_action_id | None)
Evaluator = Callable[[Game, State], tuple[float, "dict[Action, float] | None"]]


class _Node:
    """One search-tree node: a state plus sparse per-legal-action edge statistics."""

    __slots__ = (
        "state",
        "is_terminal",
        "to_play",
        "actions",
        "P",
        "N",
        "W",
        "Q",
        "vloss",
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
        self.P: list[float] = []
        self.N: list[int] = []
        self.W: list[float] = []
        self.Q: list[float] = []
        self.vloss: list[int] = []
        self.children: list[_Node | None] = []
        self.is_expanded = False


class MCTS:
    """Sparse player-aware PUCT search with subtree reuse (design doc §6.2, §11 D11).

    A single instance owns one search tree and is reused across the moves of a game:
    :meth:`run` accumulates simulations from the current root, :meth:`advance` reparents
    the tree onto a chosen child (subtree reuse), retaining its accumulated statistics.

    Args:
        game: The adapter to search. Validated against the v1 envelope on construction.
        c_init: PUCT constant ``c_init`` (D11; default 1.25).
        c_base: PUCT growth constant ``c_base`` (D11; default 19,652).
        evaluate: Optional leaf evaluator ``(game, state) -> (value, priors)`` used at
            non-terminal leaves. ``None`` (M0) means value ``0.0`` and uniform priors.
            The network plugs in here at M2 — the same seam, no new abstraction.
        uniform_prior: If True, ignore any evaluator-provided priors and use uniform
            priors over legal actions (ladder rung 6). The evaluator's *value* is kept.
        virtual_loss: Virtual-loss magnitude applied per in-flight edge (default 1).

    Raises:
        EnvelopeError: If ``game`` declares capabilities outside the v1 envelope.
    """

    def __init__(
        self,
        game: Game,
        *,
        c_init: float = 1.25,
        c_base: float = 19_652.0,
        evaluate: Evaluator | None = None,
        uniform_prior: bool = False,
        virtual_loss: int = 1,
    ):
        assert_v1_envelope(game)
        self.game = game
        self.c_init = c_init
        self.c_base = c_base
        self.evaluate = evaluate
        self.uniform_prior = uniform_prior
        self.virtual_loss = virtual_loss
        self.root: _Node | None = None

    # --- public API ------------------------------------------------------------

    def set_root(self, state: State) -> None:
        """Start a fresh search tree rooted at ``state`` (discards any prior tree)."""
        self.root = self._make_node(state)

    def run(self, num_simulations: int, root_state: State | None = None) -> _Node:
        """Run ``num_simulations`` simulations from the root and return the root node.

        Args:
            num_simulations: Number of leaf evaluations to perform.
            root_state: If given, (re)sets the root to this state first; otherwise the
                current root is used (allowing incremental search after :meth:`advance`).

        Returns:
            The root ``_Node`` (an opaque tree handle; read stats via the helpers below).
        """
        if root_state is not None:
            self.set_root(root_state)
        if self.root is None:
            raise ValueError("no root set; pass root_state or call set_root first")
        for _ in range(num_simulations):
            path, leaf, leaf_value = self._descend(apply_vloss=True)
            ref_player, value = self._leaf_value(leaf, path, leaf_value)
            self._backup(path, ref_player, value)
        return self.root

    def advance(self, action: Action) -> None:
        """Reuse the subtree under ``action`` as the new root (subtree reuse, §6.2).

        The retained child keeps its accumulated statistics; the rest of the tree is
        dropped. If the child was never materialized, a fresh node is created for the
        resulting state (no statistics to reuse).

        Args:
            action: The action id played from the current root.

        Raises:
            ValueError: If there is no root or ``action`` is not legal at the root.
        """
        if self.root is None:
            raise ValueError("no root set")
        if not self.root.is_expanded:
            # Nothing was searched; just move the root forward.
            self.root = self._make_node(self.game.apply(self.root.state, action))
            return
        try:
            i = self.root.actions.index(action)
        except ValueError as exc:
            raise ValueError(f"action {action} is not legal at the current root") from exc
        child = self.root.children[i]
        if child is None:
            child = self._make_node(self.game.apply(self.root.state, action))
        self.root = child

    def action_visit_counts(self, node: _Node | None = None) -> dict[Action, int]:
        """Return ``{action_id: visit_count}`` for the node's edges (default: root)."""
        node = node or self.root
        if node is None:
            raise ValueError("no node")
        return dict(zip(node.actions, node.N, strict=True))

    def action_values(self, node: _Node | None = None) -> dict[Action, float]:
        """Return ``{action_id: Q}`` in the mover's perspective (default: root)."""
        node = node or self.root
        if node is None:
            raise ValueError("no node")
        return dict(zip(node.actions, node.Q, strict=True))

    def best_action(self, node: _Node | None = None) -> Action:
        """Return the most-visited action, ties broken by lowest action id."""
        node = node or self.root
        if node is None or not node.actions:
            raise ValueError("no expanded node to choose from")
        best_i, best_n = 0, node.N[0]
        for i in range(1, len(node.actions)):
            if node.N[i] > best_n:
                best_i, best_n = i, node.N[i]
        return node.actions[best_i]

    def select_action(self, temperature: float, rng, node: _Node | None = None) -> Action:
        """Select an action from visit counts (design doc D10).

        Args:
            temperature: ``0`` selects the most-visited action (argmax N). ``1`` samples
                proportionally to visit counts (``π ∝ N``, no exponentiation). Other
                positive values sample ``∝ N**(1/temperature)``.
            rng: A ``random.Random`` used when sampling (``temperature > 0``).
            node: Node to select from (default: root).

        Returns:
            The chosen action id.
        """
        node = node or self.root
        if node is None or not node.actions:
            raise ValueError("no expanded node to choose from")
        if temperature == 0:
            return self.best_action(node)
        if temperature == 1.0:
            weights = list(node.N)
        else:
            inv = 1.0 / temperature
            weights = [float(n) ** inv for n in node.N]
        total = sum(weights)
        if total <= 0:  # no visits recorded (degenerate); fall back to uniform
            return rng.choice(node.actions)
        return rng.choices(node.actions, weights=weights, k=1)[0]

    # --- internals -------------------------------------------------------------

    def _make_node(self, state: State) -> _Node:
        terminal = self.game.is_terminal(state)
        to_play = None if terminal else self.game.current_player(state)
        return _Node(state, terminal, to_play)

    def _expand(self, node: _Node) -> float:
        """Attach sparse edges + priors to ``node``; return its value (mover's perspective).

        In M0 the value is ``0.0``; with an evaluator it is the network's value estimate
        for ``node.state`` from ``node.to_play``'s perspective.
        """
        actions = list(self.game.legal_moves(node.state))
        node.actions = actions
        n = len(actions)
        if self.evaluate is None:
            value, raw_priors = 0.0, None
        else:
            value, raw_priors = self.evaluate(self.game, node.state)
        node.P = self._priors(actions, raw_priors)
        node.N = [0] * n
        node.W = [0.0] * n
        node.Q = [0.0] * n
        node.vloss = [0] * n
        node.children = [None] * n
        node.is_expanded = True
        return value

    def _priors(self, actions: Sequence[Action], raw: dict[Action, float] | None) -> list[float]:
        """Return normalized priors over ``actions`` (uniform if none / flag set)."""
        n = len(actions)
        if raw is None or self.uniform_prior:
            return [1.0 / n] * n
        # Softmax over the legal-action logits only (sparse everywhere).
        logits = [raw.get(a, 0.0) for a in actions]
        m = max(logits)
        exps = [math.exp(z - m) for z in logits]
        s = sum(exps)
        return [e / s for e in exps]

    def _select_edge(self, node: _Node) -> int:
        """Return the local edge index maximizing ``Q + U`` (ties: lowest index).

        Effective statistics include virtual loss so concurrent in-flight descents do
        not re-select the same edge; a virtual visit counts as a loss in the mover's
        perspective, temporarily depressing that edge's value.
        """
        total = 0
        for k in range(len(node.actions)):
            total += node.N[k] + node.vloss[k]
        sqrt_total = math.sqrt(total)
        c = self.c_init + math.log((total + self.c_base + 1.0) / self.c_base)
        best_i, best_score = -1, -math.inf
        for k in range(len(node.actions)):
            n_k = node.N[k] + node.vloss[k]
            # First-play urgency: unvisited edges use Q = 0 (D11).
            q = (node.W[k] - node.vloss[k]) / n_k if n_k > 0 else 0.0
            u = c * node.P[k] * sqrt_total / (1.0 + n_k)
            score = q + u
            if score > best_score:
                best_i, best_score = k, score
        return best_i

    def _descend(self, apply_vloss: bool = True):
        """Select from the root to a leaf, expanding it if it is a fresh non-terminal.

        Applies virtual loss to each traversed edge (removed on backup). Returns
        ``(path, leaf, leaf_value)`` where ``path`` is a list of ``(node, edge_index)``
        and ``leaf_value`` is the freshly-expanded leaf's value (mover's perspective),
        or ``None`` if the leaf is terminal (value comes from ``terminal_utility``).
        """
        path: list[tuple[_Node, int]] = []
        node = self.root
        assert node is not None
        while True:
            if node.is_terminal:
                return path, node, None
            if not node.is_expanded:
                value = self._expand(node)
                return path, node, value
            i = self._select_edge(node)
            if apply_vloss:
                node.vloss[i] += self.virtual_loss
            path.append((node, i))
            child = node.children[i]
            if child is None:
                child = self._make_node(self.game.apply(node.state, node.actions[i]))
                node.children[i] = child
            node = child

    def _leaf_value(self, leaf: _Node, path, expanded_value):
        """Return ``(ref_player, leaf_value)`` for backup.

        For a terminal leaf, the value is measured from the perspective of the player
        who just moved (the parent node's mover); the root being terminal is degenerate
        and contributes nothing. For a freshly-expanded leaf, the value is measured from
        the leaf mover's perspective (``0.0`` in M0).
        """
        if leaf.is_terminal:
            if not path:
                return None, 0.0
            parent, _ = path[-1]
            ref = parent.to_play
            return ref, self.game.terminal_utility(leaf.state, ref)
        return leaf.to_play, expanded_value

    def _backup(self, path, ref_player: PlayerId | None, leaf_value: float) -> None:
        """Propagate ``leaf_value`` up ``path``, removing virtual loss (design doc §6.2).

        Edge-relative sign rule: ``edge_value = leaf_value`` if the edge's node moved as
        ``ref_player`` else ``-leaf_value``. Because each edge reads its own node's mover,
        this is correct across consecutive same-player moves.
        """
        for node, i in path:
            node.vloss[i] -= self.virtual_loss
            node.N[i] += 1
            edge_value = leaf_value if node.to_play == ref_player else -leaf_value
            node.W[i] += edge_value
            node.Q[i] = node.W[i] / node.N[i]
