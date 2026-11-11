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
    leaves zero residue; it steers concurrent in-flight descents apart.
  * Root Dirichlet noise (D7) is an *option*, off by default: it is exploration for
    self-play only, so every evaluation path runs the undisturbed priors.

Batched parallel selection (design doc §6.2, §12 M5): ``run(n, batch_size=K)`` collects
up to ``K`` descents before evaluating anything, hands the non-terminal leaves to one
evaluation (``evaluate_batch`` if supplied, else ``evaluate`` mapped over them), then
fills and backs up every path. Expansion is therefore split in two — a *defer* step that
reserves the leaf during descent and a *fill* step that installs its evaluation — and
terminal leaves take ``terminal_utility`` inside the loop without reaching the evaluator.

  * Two mechanisms keep a batch's leaves distinct, and both are required. Virtual loss
    diverts the next descent where there is a choice; the per-node *in-flight marker*
    detects the cases it cannot divert — a forced move, or an unexpanded root, where no
    edge exists for virtual loss to act on.
  * The first collision closes the batch. A descent that reaches a leaf already in
    flight has its virtual loss rolled back in full and is not a simulation; there is no
    retry and no collision budget. ``K`` is thus an upper bound: a batch is also capped
    by the remaining budget, and it need not fill even when the tree holds ``K``
    unevaluated leaves. A batch's first descent cannot collide, so every batch completes
    at least one simulation.
  * ``completed_simulations`` counts backed-up simulations, as distinct from descents
    attempted; the budget is spent against it, exactly.
  * ``batch_size = 1`` is the serial search, bit for bit. Batching changes selection only
    through the in-flight virtual-loss term: PUCT (D11), root noise (D7) and subtree
    reuse are as in the serial engine.
"""

from __future__ import annotations

import math
import numbers
import random
from collections.abc import Callable, Sequence

from core.game import Action, Game, PlayerId, State, assert_v1_envelope

# evaluate(game, state) -> (value_from_movers_perspective, priors_by_action_id | None)
Evaluator = Callable[[Game, State], tuple[float, "dict[Action, float] | None"]]

# evaluate_batch(game, states) -> one (value, priors) result per state, in order.
BatchEvaluator = Callable[
    [Game, Sequence[State]], Sequence[tuple[float, "dict[Action, float] | None"]]
]

# root_noise = (eps, alpha_numerator, rng); alpha = alpha_numerator / #legal (D7).
RootNoise = tuple[float, float, random.Random]


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
        "in_flight",
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
        # True while this node is the leaf of a descent that has not been backed up yet
        # (design doc §6.2: the in-flight-leaf marker; set on terminal leaves too).
        self.in_flight = False


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
        evaluate_batch: Optional batched leaf evaluator ``(game, states) -> results``,
            returning one ``(value, priors)`` per state in order. When supplied it
            evaluates every batch (of any size) in one call and ``evaluate`` is not
            used; when ``None``, a batch maps ``evaluate`` over its leaves — correct,
            unoptimized. The states of one call belong to distinct leaves but need not
            be distinct values: there is no transposition table, so two leaves reached
            by different move orders can hold equal states, and an evaluator must not
            assume otherwise.
        uniform_prior: If True, ignore any evaluator-provided priors and use uniform
            priors over legal actions (ladder rung 6). The evaluator's *value* is kept.
        virtual_loss: Virtual-loss magnitude applied per in-flight edge (default 1).
        root_noise: Optional ``(eps, alpha_numerator, rng)`` enabling D7 root Dirichlet
            noise — ``P' = (1-eps)·P + eps·Dir(alpha)`` with ``alpha = alpha_numerator /
            #legal`` over the root's legal actions only. Self-play sets ``(0.25, 10.8,
            rng)``; ``None`` (default) leaves the search bit-identical to the noiseless
            engine, which is what every evaluation path wants.

    Raises:
        EnvelopeError: If ``game`` declares capabilities outside the v1 envelope.
        ValueError: If ``root_noise`` carries an ``eps`` outside ``[0, 1]`` or a
            non-positive ``alpha_numerator``.

    Attributes:
        completed_simulations: Simulations backed up over this engine's lifetime. A
            descent rejected at an in-flight leaf is not a simulation, so this — not the
            number of descents attempted — is what a search budget is spent against.
    """

    def __init__(
        self,
        game: Game,
        *,
        c_init: float = 1.25,
        c_base: float = 19_652.0,
        evaluate: Evaluator | None = None,
        evaluate_batch: BatchEvaluator | None = None,
        uniform_prior: bool = False,
        virtual_loss: int = 1,
        root_noise: RootNoise | None = None,
    ):
        assert_v1_envelope(game)
        if root_noise is not None:
            eps, alpha_numerator, _ = root_noise
            if not 0.0 <= eps <= 1.0:
                raise ValueError(f"root-noise eps must lie in [0, 1]; got {eps}")
            if alpha_numerator <= 0.0:
                raise ValueError(
                    f"root-noise alpha_numerator must be positive; got {alpha_numerator}"
                )
        self.game = game
        self.c_init = c_init
        self.c_base = c_base
        self.evaluate = evaluate
        self.evaluate_batch = evaluate_batch
        self.uniform_prior = uniform_prior
        self.virtual_loss = virtual_loss
        self.root_noise = root_noise
        self.root: _Node | None = None
        # A root "incarnation" counter: bumped every time a node *becomes* the root, so
        # noise is re-drawn per incarnation (D7) — once, not on every run() call.
        self._root_epoch = 0
        self._noised_epoch = -1
        # Simulations backed up over this engine's lifetime — never reset, not by
        # set_root() and not by advance(); callers read deltas around a run().
        self.completed_simulations = 0

    # --- public API ------------------------------------------------------------

    def set_root(self, state: State) -> None:
        """Start a fresh search tree rooted at ``state`` (discards any prior tree)."""
        self.root = self._make_node(state)
        self._root_epoch += 1

    def run(
        self, num_simulations: int, root_state: State | None = None, *, batch_size: int = 1
    ) -> _Node:
        """Run exactly ``num_simulations`` simulations from the root; return the root node.

        Simulations run in batches: up to ``batch_size`` descents are collected with
        virtual loss held on every in-flight path, the non-terminal leaves are evaluated
        in one call (terminal leaves take ``terminal_utility`` and never reach the
        evaluator), and every path is then filled and backed up. ``batch_size`` is only an
        upper bound on leaves per evaluation — a batch is capped by the remaining budget
        and closed early by the first descent that reaches a leaf already in flight. That
        descent's virtual loss is rolled back in full and it spends none of the budget;
        there is no retry. A batch's first descent cannot collide, so every batch
        completes at least one simulation. With ``batch_size = 1`` this is the serial
        search, bit for bit.

        A batch whose evaluation fails is abandoned whole. If the evaluator raises, or
        returns anything the engine cannot use (see Raises), every descent of that batch
        is rolled back before the error propagates: no virtual loss or in-flight mark is
        left, no leaf of the batch is expanded, and none of it is counted. Every result is
        checked before the first leaf is filled, so one bad result discards the batch's
        well-formed ones too. This covers the evaluator only: an exception raised by the
        game adapter itself (``apply`` / ``legal_moves`` during a descent,
        ``terminal_utility`` during backup) can still leave virtual loss behind.

        Args:
            num_simulations: Number of simulations (backed-up leaf visits) to perform.
            root_state: If given, (re)sets the root to this state first; otherwise the
                current root is used (allowing incremental search after :meth:`advance`).
            batch_size: Maximum number of leaves per evaluation; any value ``>= 1`` is
                valid, whatever the simulation budget.

        Returns:
            The root ``_Node`` (an opaque tree handle; read stats via the helpers below).

        Raises:
            ValueError: If ``batch_size < 1``, if no root is set, if ``evaluate_batch``
                returns a different number of results than it was given states, or if a
                result is not a ``(value, priors)`` pair.
            TypeError: If an evaluated value is not a real number. Malformed priors raise
                whatever the softmax over them raises (typically also ``TypeError``).
            RuntimeError: If a batch opens onto a leaf that is already in flight — a
                stale mark, which no completed run and no failed evaluation leaves behind.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size}")
        if root_state is not None:
            self.set_root(root_state)
        if self.root is None:
            raise ValueError("no root set; pass root_state or call set_root first")
        remaining = num_simulations
        while remaining > 0:
            # Leaves are filled only once the batch has closed, so root priors cannot
            # appear mid-batch: one noise check per batch is one check per descent.
            self._maybe_noise_root()
            batch: list[tuple[list[tuple[_Node, int]], _Node]] = []
            try:
                for _ in range(min(batch_size, remaining)):
                    path, leaf, collided = self._descend(apply_vloss=True)
                    if collided:
                        self._rollback_vloss(path)
                        break  # the first collision closes the batch (no retry)
                    batch.append((path, leaf))
                if not batch:
                    raise RuntimeError("a batch opened onto a leaf that is already in flight")
                results = iter(
                    self._evaluate_leaves([leaf for _, leaf in batch if not leaf.is_terminal])
                )
            except BaseException:
                # Nothing is filled yet, so abandoning the batch is exact: no virtual
                # loss, no in-flight mark, no simulation counted.
                for path, leaf in batch:
                    self._rollback_vloss(path)
                    leaf.in_flight = False
                    leaf.actions = []  # undo the deferral: a fresh node again
                raise
            # Every result is already checked, so nothing below fails on the evaluator.
            for path, leaf in batch:
                expanded_value = (
                    None if leaf.is_terminal else self._fill_expand(leaf, *next(results))
                )
                ref_player, value = self._leaf_value(leaf, path, expanded_value)
                self._backup(path, ref_player, value)
                leaf.in_flight = False
                self.completed_simulations += 1
            remaining -= len(batch)
        return self.root

    def advance(self, action: Action) -> None:
        """Reuse the subtree under ``action`` as the new root (subtree reuse, §6.2).

        The retained child keeps its accumulated statistics; the rest of the tree is
        dropped. If the child was never materialized, a fresh node is created for the
        resulting state (no statistics to reuse). The promoted node is a *new* root
        incarnation, so the next :meth:`run` re-draws root noise onto its priors (D7) —
        the retained priors carry the previous root's noise, if any, only until then.

        Args:
            action: The action id played from the current root.

        Raises:
            ValueError: If there is no root or ``action`` is not legal at the root.
        """
        if self.root is None:
            raise ValueError("no root set")
        if not self.root.is_expanded:
            # Nothing was searched; move the root forward — but validate legality first,
            # since an unexpanded root has no edge list to reject the action for us.
            if action not in self.game.legal_moves(self.root.state):
                raise ValueError(f"action {action} is not legal at the current root")
            self.root = self._make_node(self.game.apply(self.root.state, action))
            self._root_epoch += 1
            return
        try:
            i = self.root.actions.index(action)
        except ValueError as exc:
            raise ValueError(f"action {action} is not legal at the current root") from exc
        child = self.root.children[i]
        if child is None:
            child = self._make_node(self.game.apply(self.root.state, action))
        self.root = child
        self._root_epoch += 1

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
        best_i = 0
        for i in range(1, len(node.actions)):
            # Most visits wins; ties go to the lowest action id (not adapter/index order).
            if node.N[i] > node.N[best_i] or (
                node.N[i] == node.N[best_i] and node.actions[i] < node.actions[best_i]
            ):
                best_i = i
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

    def _evaluate_leaves(self, leaves: Sequence[_Node]) -> list[tuple[float, list[float]]]:
        """Evaluate one batch's non-terminal leaves and check every result.

        The evaluation comes from ``evaluate_batch`` in a single call if supplied, else
        from ``evaluate`` mapped over the leaves, else it is the M0 default of value
        ``0.0`` and uniform priors. Everything that can fail on what the evaluator
        returned fails here, before any leaf is touched — which is what lets
        :meth:`run` abandon a batch whole. Values are checked, never coerced.

        Args:
            leaves: The batch's deferred (non-terminal) leaves, in collection order.

        Returns:
            One ``(value, priors)`` per leaf, in order, with ``priors`` already
            normalized over that leaf's legal actions (:meth:`_priors`).

        Raises:
            ValueError: If ``evaluate_batch`` returns the wrong number of results, or a
                result is not a ``(value, priors)`` pair.
            TypeError: If a value is not a real number.
        """
        if self.evaluate_batch is None:
            if self.evaluate is None:
                results = [(0.0, None)] * len(leaves)
            else:
                results = [self.evaluate(self.game, leaf.state) for leaf in leaves]
        else:
            # An all-terminal batch has nothing to evaluate: make no call at all.
            results = (
                self.evaluate_batch(self.game, [leaf.state for leaf in leaves]) if leaves else []
            )
            if len(results) != len(leaves):
                raise ValueError(
                    f"evaluate_batch returned {len(results)} results for {len(leaves)} states"
                )
        checked = []
        for leaf, (value, raw_priors) in zip(leaves, results, strict=True):
            if not isinstance(value, numbers.Real):
                raise TypeError(f"evaluated value must be a real number; got {value!r}")
            checked.append((value, self._priors(leaf.actions, raw_priors)))
        return checked

    def _defer_expand(self, node: _Node) -> None:
        """Reserve a fresh non-terminal leaf for evaluation (the defer half of expansion).

        Stores the legal actions and marks the node in flight, with priors and value still
        pending. The node stays unexpanded, so a descent that reaches it again before
        :meth:`_fill_expand` stops there and reports a collision instead of re-expanding it.

        Args:
            node: A non-terminal node that has not been expanded.
        """
        node.actions = list(self.game.legal_moves(node.state))
        node.in_flight = True

    def _fill_expand(self, node: _Node, value: float, priors: list[float]) -> float:
        """Install an evaluation on a deferred leaf (the fill half of expansion).

        Completes the sparse ``{N, W, Q, P}`` expansion over the legal actions stored by
        :meth:`_defer_expand`. It only assigns, so it cannot fail part-way.

        Args:
            node: A node previously passed to :meth:`_defer_expand`.
            value: The evaluation of ``node.state`` from ``node.to_play``'s perspective
                (``0.0`` in M0, the network's value estimate with an evaluator).
            priors: Normalized priors over ``node.actions``, as :meth:`_priors` returns
                them for this node.

        Returns:
            ``value``, unchanged — the leaf value to back up.
        """
        n = len(node.actions)
        node.P = priors
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

    def _maybe_noise_root(self) -> None:
        """Mix D7 Dirichlet noise into the current root's priors, once per root incarnation.

        A no-op unless ``root_noise`` is configured. Root priors only exist once the root is
        expanded, and a fresh root is expanded by its own first simulation — so this runs
        before every batch and fires on the first one that finds expanded priors (a fresh
        root's first batch is that one simulation, whatever the batch size: its descent
        stops at the root, before any prior is read). Because the guard
        is the root *epoch*, a subtree promoted by :meth:`advance` gets fresh noise on the
        next search, while repeated :meth:`run` calls on one root never re-mix.
        """
        if self.root_noise is None or self._noised_epoch == self._root_epoch:
            return
        root = self.root
        if root is None or not root.is_expanded or not root.actions:
            return
        eps, alpha_numerator, rng = self.root_noise
        noise = self._dirichlet(len(root.actions), alpha_numerator / len(root.actions), rng)
        root.P = [(1.0 - eps) * p + eps * d for p, d in zip(root.P, noise, strict=True)]
        self._noised_epoch = self._root_epoch

    @staticmethod
    def _dirichlet(n: int, alpha: float, rng: random.Random) -> list[float]:
        """Return one symmetric ``Dir(alpha)`` sample of length ``n`` drawn from ``rng``.

        Stdlib-only construction (``core/`` stays NumPy-free here): draw ``Gamma(alpha, 1)``
        per component and normalize.

        Args:
            n: Number of components (the root's legal-action count).
            alpha: The symmetric concentration parameter (D7: ``10.8 / n``).
            rng: The Dirichlet stream — its own purpose-keyed ``random.Random``.

        Returns:
            ``n`` non-negative weights summing to 1.
        """
        draws = [rng.gammavariate(alpha, 1.0) for _ in range(n)]
        total = sum(draws)
        if total <= 0.0:
            # alpha ≈ 0.013 at an 828-wide Blokus root, so every gamma draw underflowing to
            # 0.0 is representable-in-principle; fall back to the Dirichlet mean, never a
            # division by zero.
            return [1.0 / n] * n
        return [d / total for d in draws]

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
        """Select from the root to a leaf: a terminal node or an unexpanded one.

        Applies virtual loss to each traversed edge (removed on backup). A leaf that is
        not already in flight is marked so — a non-terminal one through
        :meth:`_defer_expand`, which leaves its evaluation pending — and :meth:`run` clears
        the mark once the path is backed up.

        Args:
            apply_vloss: Whether to add virtual loss to the traversed edges.

        Returns:
            ``(path, leaf, collided)``: ``path`` is the list of ``(node, edge_index)``
            traversed, and ``collided`` is True iff ``leaf`` was already in flight when
            this descent reached it (the descent then left the leaf untouched).
        """
        path: list[tuple[_Node, int]] = []
        node = self.root
        assert node is not None
        while True:
            if node.is_terminal or not node.is_expanded:
                collided = node.in_flight
                if node.is_terminal:
                    node.in_flight = True
                elif not collided:
                    self._defer_expand(node)
                return path, node, collided
            i = self._select_edge(node)
            if apply_vloss:
                node.vloss[i] += self.virtual_loss
            path.append((node, i))
            child = node.children[i]
            if child is None:
                child = self._make_node(self.game.apply(node.state, node.actions[i]))
                node.children[i] = child
            node = child

    def _rollback_vloss(self, path: Sequence[tuple[_Node, int]]) -> None:
        """Remove one descent's virtual loss from ``path`` without backing anything up.

        Args:
            path: The ``(node, edge_index)`` pairs of a descent made with virtual loss.
        """
        for node, i in path:
            node.vloss[i] -= self.virtual_loss

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
