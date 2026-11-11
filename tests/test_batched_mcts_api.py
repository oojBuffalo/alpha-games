"""Batched parallel selection: API-level smoke tests (design doc §6.2, §12 M5).

The serial engine is pinned first: a seeded golden of every node's ``{N, W, Q}``,
captured from the engine *before* batching touched it, so "``batch_size = 1`` reproduces
the serial engine bit for bit" is checked against that engine and not against itself.
"""

from __future__ import annotations

import hashlib
import random

import pytest

from core import MCTS
from games.connect4 import Connect4
from games.tictactoe import TicTacToe
from tests.fixtures.pass_game import (
    PassGame,
    Scenario,
    consecutive_trap_game,
    consecutive_win_game,
)

EPS = 0.25  # D7
ALPHA_NUMERATOR = 10.8  # D7


def _all_nodes(root):
    """Return every materialized node under ``root`` in deterministic preorder."""
    nodes, stack = [], [root]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        nodes.append(node)
        stack.extend(reversed(node.children))
    return nodes


def _digest(root) -> str:
    """Return a hash of every node's exact ``{N, W, Q}`` (floats compared bit for bit)."""
    rows = [
        (list(node.N), [w.hex() for w in node.W], [q.hex() for q in node.Q])
        for node in _all_nodes(root)
    ]
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def _seeded_evaluator(seed: int):
    """Return a deterministic evaluator: value and logits are a pure function of the state."""

    def evaluate(game, state):
        rng = random.Random(f"{seed}:{state!r}")
        value = rng.uniform(-1.0, 1.0)
        return value, {a: rng.uniform(-2.0, 2.0) for a in game.legal_moves(state)}

    return evaluate


def _mapped(evaluate):
    """Return a batched evaluator that maps the single-leaf ``evaluate`` over the states."""
    return lambda game, states: [evaluate(game, state) for state in states]


def _seeded_search(
    game, seed: int | None, *, batched_evaluator: bool = False, **run_kwargs
) -> tuple[str, int, str]:
    """Search, reuse the chosen subtree, search again; return what the golden pins.

    Args:
        game: The adapter to search from its initial state.
        seed: Seeds the evaluator and the D7 noise stream; ``None`` runs the M0 default
            (no evaluator, no noise).
        batched_evaluator: Supply the seeded evaluator through ``evaluate_batch`` only,
            instead of through ``evaluate`` only.
        **run_kwargs: Forwarded to both :meth:`MCTS.run` calls.

    Returns:
        ``(first-search digest, best action, post-advance digest)``.
    """
    if seed is None:
        m = MCTS(game)
    else:
        noise = (EPS, ALPHA_NUMERATOR, random.Random(seed))
        evaluate = _seeded_evaluator(seed)
        if batched_evaluator:
            m = MCTS(game, evaluate_batch=_mapped(evaluate), root_noise=noise)
        else:
            m = MCTS(game, evaluate=evaluate, root_noise=noise)
    first = _digest(m.run(300, root_state=game.initial_state(), **run_kwargs))
    action = m.best_action()
    m.advance(action)
    return first, action, _digest(m.run(150, **run_kwargs))


# Captured from the serial engine as it stood before batched selection was added.
SERIAL_GOLDEN = {
    "ttt-m0": (
        "e2b444ea265559ac670e00bc57738ccd8bbda1a0d0c351e0536299795b9570d9",
        0,
        "e6912681331065cee7f3420c6dcc67e81ae204d4f5e624fe2dacd054bff9bac5",
    ),
    "ttt-seeded": (
        "1c2a3f211913aa05256d78ad6eab16611ef4b4afdfde63d9e3f05a496b83b1ad",
        2,
        "f1eef20b33ab4bee7c7ec9f0faa4542b6a74b2d3cb784ee8ec8c23b78e8b679e",
    ),
    "c4-m0": (
        "aa08986251a0120ee017e478dd8dd599f9ef1da543bfb80e9e4279abc758a0e0",
        0,
        "05722434f77fd65d6ab1062d9f4a601fd3c2f8c06aa37ee91d7130dfc3ec7be6",
    ),
    "c4-seeded": (
        "f9750cbfe351749f8c3f05b25f51c5711540a1abfc78ce50494b18e9d0a032af",
        4,
        "a0e377bafd2fe94257ef73edbca08e68787bf3b88dcdfb76cc91deda1d9dc7d3",
    ),
}

GOLDEN_CASES = {
    "ttt-m0": (TicTacToe, None),
    "ttt-seeded": (TicTacToe, 20261110),
    "c4-m0": (Connect4, None),
    "c4-seeded": (Connect4, 20261110),
}


@pytest.mark.parametrize("case", sorted(GOLDEN_CASES))
def test_serial_search_matches_the_pre_batching_golden(case):
    make_game, seed = GOLDEN_CASES[case]
    assert _seeded_search(make_game(), seed) == SERIAL_GOLDEN[case]


@pytest.mark.parametrize("case", sorted(GOLDEN_CASES))
def test_batch_size_one_reproduces_the_serial_engine_bit_for_bit(case):
    make_game, seed = GOLDEN_CASES[case]
    assert _seeded_search(make_game(), seed, batch_size=1) == SERIAL_GOLDEN[case]


@pytest.mark.parametrize("case", ["ttt-seeded", "c4-seeded"])
def test_batch_size_one_through_the_batched_evaluator_is_also_bit_identical(case):
    make_game, seed = GOLDEN_CASES[case]
    found = _seeded_search(make_game(), seed, batched_evaluator=True, batch_size=1)
    assert found == SERIAL_GOLDEN[case]


# --- bookkeeping primitives: completed-simulations counter + in-flight marker ----------


@pytest.mark.parametrize("make_game", [TicTacToe, Connect4])
def test_run_counts_completed_simulations_and_leaves_nothing_in_flight(make_game):
    game = make_game()
    m = MCTS(game)
    assert m.completed_simulations == 0

    root = m.run(120, root_state=game.initial_state())
    assert m.completed_simulations == 120
    assert not any(node.in_flight for node in _all_nodes(root))

    # The counter is engine-lifetime: subtree reuse and incremental runs only add to it.
    m.advance(m.best_action())
    m.run(30)
    assert m.completed_simulations == 150
    assert not any(node.in_flight for node in _all_nodes(m.root))


# --- expansion split: defer (reserve the leaf) then fill (install the evaluation) ------


def test_deferred_leaf_is_a_collision_target_until_it_is_filled():
    game = TicTacToe()
    m = MCTS(game)
    m.set_root(game.initial_state())
    node = m.root

    m._defer_expand(node)
    # Reserved, not expanded: a second descent reaching it must see a collision, never
    # re-expand it or select through edges that have no priors yet.
    assert node.in_flight and not node.is_expanded
    assert node.actions == list(game.legal_moves(node.state))
    assert node.P == []

    assert m._fill_expand(node, 0.25, m._priors(node.actions, {4: 3.0})) == 0.25
    n = len(node.actions)
    assert node.is_expanded
    assert abs(sum(node.P) - 1.0) < 1e-12
    assert max(node.P) == node.P[node.actions.index(4)]  # the one raised logit
    assert node.N == [0] * n and node.W == [0.0] * n and node.Q == [0.0] * n
    assert node.vloss == [0] * n and node.children == [None] * n


def test_descent_reports_a_collision_at_a_leaf_that_is_already_in_flight():
    game = TicTacToe()
    m = MCTS(game)
    m.set_root(game.initial_state())

    path, leaf, collided = m._descend(apply_vloss=True)
    assert (path, leaf, collided) == ([], m.root, False)
    assert leaf.in_flight and not leaf.is_expanded  # deferred: no evaluation yet

    assert m._descend(apply_vloss=True) == ([], m.root, True)


# --- the batched run loop --------------------------------------------------------------


class _Recorder:
    """A batched evaluator that logs every call and observes the tree mid-batch.

    Attributes:
        calls: The list of states handed over, one entry per ``evaluate_batch`` call.
        root_vloss: The root's virtual-loss vector as seen inside each call — i.e. while
            the batch's descents are still in flight.
        in_flight: The states of every in-flight node, as seen inside each call.
        pending: The states of the in-flight *non-terminal* nodes, as seen inside each
            call. The tree walk yields each node once, so this lists one state per node.
    """

    def __init__(self, value: float = 0.0):
        self.value = value
        self.mcts: MCTS | None = None
        self.calls: list[list] = []
        self.root_vloss: list[list[int]] = []
        self.in_flight: list[list] = []
        self.pending: list[list] = []

    def __call__(self, game, states):
        nodes = [n for n in _all_nodes(self.mcts.root) if n.in_flight]
        self.calls.append(list(states))
        self.root_vloss.append(list(self.mcts.root.vloss))
        self.in_flight.append([n.state for n in nodes])
        self.pending.append([n.state for n in nodes if not n.is_terminal])
        return [(self.value, None)] * len(states)


def _recorded(game, value: float = 0.0) -> tuple[MCTS, _Recorder]:
    """Return an engine whose only evaluator is a fresh :class:`_Recorder`."""
    recorder = _Recorder(value)
    recorder.mcts = MCTS(game, evaluate_batch=recorder)
    return recorder.mcts, recorder


def _assert_no_residue(root) -> None:
    """Assert a finished search left no virtual loss and no in-flight mark anywhere."""
    for node in _all_nodes(root):
        assert not node.in_flight
        assert all(v == 0 for v in node.vloss)
        assert len(node.N) == len(node.actions)  # no half-expanded node
        for i in range(len(node.actions)):
            assert node.Q[i] == (node.W[i] / node.N[i] if node.N[i] else 0.0)


@pytest.mark.parametrize("make_game", [TicTacToe, Connect4])
@pytest.mark.parametrize("batch_size", [2, 8, 64, 1000])
def test_batched_run_spends_the_budget_exactly_and_leaves_no_residue(make_game, batch_size):
    game = make_game()
    m = MCTS(game)
    n_sims = 101  # not a multiple of any batch size above

    root = m.run(n_sims, root_state=game.initial_state(), batch_size=batch_size)

    # The pinned convention: sim 1 expands the root without traversing an edge.
    assert sum(root.N) == n_sims - 1
    assert m.completed_simulations == n_sims
    _assert_no_residue(root)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_below_one_is_rejected(batch_size):
    game = TicTacToe()
    with pytest.raises(ValueError):
        MCTS(game).run(10, root_state=game.initial_state(), batch_size=batch_size)


def test_one_batch_collects_distinct_unevaluated_leaves():
    game = Connect4(4, 4, 3)  # branching factor 4
    m, rec = _recorded(game)
    start = game.initial_state()

    m.run(5, root_state=start, batch_size=4)

    # The first batch on an unexpanded root is structurally size 1: only the root exists,
    # and virtual loss has no edge to act on — the in-flight marker closes the batch.
    assert rec.calls[0] == [start]
    # The next batch fills: virtual loss steers four descents onto four distinct children,
    # all held in flight together while the one evaluation runs.
    assert rec.calls[1] == [game.apply(start, a) for a in game.legal_moves(start)]
    assert len(set(rec.calls[1])) == 4
    assert rec.root_vloss[1] == [1, 1, 1, 1]
    assert rec.in_flight[1] == rec.calls[1]
    assert len(rec.calls) == 2
    assert m.completed_simulations == 5
    _assert_no_residue(m.root)


def test_no_batch_ever_holds_one_leaf_twice_though_equal_states_may_share_a_batch():
    game = TicTacToe()
    m, rec = _recorded(game)

    m.run(100, root_state=game.initial_state(), batch_size=2)

    assert all(0 < len(call) <= 2 for call in rec.calls)
    assert any(len(call) > 1 for call in rec.calls)
    # Distinct leaves means distinct *nodes*: every state handed over belongs to its own
    # in-flight node, so a leaf collected twice would leave a state without one.
    for call, pending in zip(rec.calls, rec.pending, strict=True):
        assert sorted(pending) == sorted(call)
    # It does not mean distinct states. With no transposition table two leaves reached by
    # different move orders hold equal states, and one batch can carry both.
    assert any(len(set(call)) < len(call) for call in rec.calls)
    assert m.completed_simulations == 100


def test_each_result_of_a_batch_is_installed_on_its_own_leaf():
    game = Connect4(4, 4, 3)
    evaluate = _seeded_evaluator(3)  # value and logits depend on the state
    m = MCTS(game, evaluate_batch=_mapped(evaluate))
    start = game.initial_state()

    root = m.run(5, root_state=start, batch_size=4)

    assert root.N == [1, 1, 1, 1]  # one batch of four, one visit down each root edge
    for i, child in enumerate(root.children):
        value, logits = evaluate(game, game.apply(start, root.actions[i]))
        assert root.W[i] == -value  # the child's own value, from its mover's perspective
        assert child.P == m._priors(child.actions, logits)  # and its own priors


def test_a_duplicate_at_an_in_flight_terminal_leaf_closes_the_batch():
    # One forced move straight into a terminal: every descent after the first in a batch
    # lands on the same terminal leaf, which needs no evaluation but is in flight.
    game = PassGame(
        Scenario(start=0, to_play={0: 0}, edges={0: [(0, 1)]}, terminal={1: (1.0, -1.0)})
    )
    m = MCTS(game)
    collided = []
    descend = m._descend

    def spy(**kwargs):
        result = descend(**kwargs)
        collided.append(result[2])
        return result

    m._descend = spy
    root = m.run(4, root_state=0, batch_size=4)

    assert collided == [False, True, False, True, False, True, False]
    assert m.completed_simulations == 4
    assert root.N == [3] and root.W == [3.0]
    _assert_no_residue(root)


def test_forced_duplicate_closes_the_batch_and_the_rejected_descent_costs_nothing():
    # A forced line: the root has one legal action, so virtual loss cannot divert the
    # second descent — it lands on the same unevaluated leaf (state 1).
    game = consecutive_trap_game()
    m, rec = _recorded(game)
    descents = []
    descend = m._descend
    m._descend = lambda **kw: descents.append(1) or descend(**kw)

    root = m.run(3, root_state=game.initial_state(), batch_size=4)

    # Batch 1: the root, closed by the collision at the root itself. Batch 2: state 1,
    # closed by the forced duplicate. Batch 3: one terminal leaf (no evaluation at all).
    assert rec.calls == [[0], [1]]
    # While state 1 was being evaluated only the accepted descent's virtual loss was on
    # the forced edge: the rejected descent's was rolled back in full.
    assert rec.root_vloss[1] == [1]
    assert rec.in_flight[1] == [1]
    # Five descents were attempted, two were rejected, and exactly three simulations were
    # spent: a rejected descent consumes none of the budget.
    assert len(descents) == 5
    assert m.completed_simulations == 3
    assert sum(root.N) == 2
    _assert_no_residue(root)


def test_terminal_leaves_in_a_mixed_batch_back_up_terminal_utility():
    game = Connect4()
    state = game.from_moves([0, 1, 0, 1, 0, 1])  # player 0 wins by playing column 0
    m, rec = _recorded(game, value=0.5)

    root = m.run(8, root_state=state, batch_size=7)

    # Batch 2 is seven descents down the seven root edges: one terminal leaf (the win)
    # and six non-terminal ones. Only the six reach the evaluator.
    assert [len(call) for call in rec.calls] == [1, 6]
    assert not any(game.is_terminal(s) for s in rec.calls[1])
    assert rec.in_flight[1][0] == game.apply(state, 0)  # the terminal leaf is in flight too
    assert root.actions == [0, 1, 2, 3, 4, 5, 6]
    assert root.N == [1] * 7
    # The win backs up terminal_utility for the player who just moved; each evaluated leaf
    # backs up its value from the opponent's (the leaf mover's) perspective, negated.
    assert root.W == [1.0] + [-0.5] * 6
    assert m.completed_simulations == 8
    _assert_no_residue(root)


def test_batched_backup_is_player_aware_across_a_consecutive_mover():
    game = consecutive_win_game()  # player 0 moves at the root and again at state 1
    m, rec = _recorded(game, value=0.5)

    root = m.run(3, root_state=game.initial_state(), batch_size=2)

    assert rec.calls == [[0], [1, 2]]
    # State 1 is player 0's own move (same sign as the root mover); state 2 is player 1's.
    assert root.W == [0.5, -0.5]

    m.run(400, batch_size=4)
    assert m.best_action() == 0
    assert m.best_action(root.children[0]) == 0
    _assert_no_residue(root)


def test_terminal_root_spends_the_budget_without_spinning():
    game = consecutive_trap_game()
    m = MCTS(game)

    m.run(5, root_state=2, batch_size=4)  # state 2 is terminal

    assert m.completed_simulations == 5
    assert not m.root.in_flight


@pytest.mark.parametrize("case", ["ttt-seeded", "c4-seeded"])
def test_same_seed_and_same_batch_size_give_an_identical_search(case):
    make_game, seed = GOLDEN_CASES[case]
    first = _seeded_search(make_game(), seed, batch_size=8)
    assert first == _seeded_search(make_game(), seed, batch_size=8)
    assert first != SERIAL_GOLDEN[case]  # batching really did change the visit order


@pytest.mark.parametrize("case", ["ttt-seeded", "c4-seeded"])
def test_single_leaf_evaluator_is_mapped_over_the_batch(case):
    make_game, seed = GOLDEN_CASES[case]
    mapped = _seeded_search(make_game(), seed, batch_size=8)
    batched = _seeded_search(make_game(), seed, batched_evaluator=True, batch_size=8)
    assert mapped == batched


def test_batched_evaluator_takes_precedence_over_the_single_leaf_one():
    game = TicTacToe()
    rec = _Recorder()

    def single(game, state):
        raise AssertionError("the single-leaf evaluator must not be called")

    rec.mcts = MCTS(game, evaluate=single, evaluate_batch=rec)
    rec.mcts.run(20, root_state=game.initial_state(), batch_size=4)
    assert sum(len(call) for call in rec.calls) == 20


# What the third evaluate_batch call does, and the error (type, message) run() must
# surface. The last four spoil only the *final* result, behind a well-formed prefix.
COUNT = r"evaluate_batch returned \d+ results for \d+ states"
EVALUATION_FAILURES = {
    "raises": (None, RuntimeError, "evaluator down"),
    "short": (lambda good: good[:-1], ValueError, COUNT),
    "long": (lambda good: [*good, (0.0, None)], ValueError, COUNT),
    "value-none": (lambda good: [*good[:-1], (None, None)], TypeError, "real number"),
    "value-not-a-number": (lambda good: [*good[:-1], ("0.5", None)], TypeError, "real number"),
    "priors-not-logits": (lambda good: [*good[:-1], (0.0, {0: "high"})], TypeError, None),
    "result-not-a-pair": (lambda good: [*good[:-1], (0.0,)], ValueError, "unpack"),
}


@pytest.mark.parametrize("failure", sorted(EVALUATION_FAILURES))
def test_a_failed_evaluation_abandons_the_batch_and_leaves_the_tree_clean(failure):
    game = Connect4(4, 4, 3)
    spoil, error, message = EVALUATION_FAILURES[failure]
    calls = []

    def flaky(game, states):
        calls.append(len(states))
        good = [(0.0, None)] * len(states)
        if len(calls) != 3:
            return good
        if spoil is None:
            raise RuntimeError("evaluator down")
        return spoil(good)

    m = MCTS(game, evaluate_batch=flaky)
    with pytest.raises(error, match=message):
        m.run(40, root_state=game.initial_state(), batch_size=4)

    # All or nothing: only the two batches that finished count. Every descent of the
    # failed batch is rolled back, including those whose own result was well formed.
    assert calls[2] > 1
    assert m.completed_simulations == calls[0] + calls[1] == sum(m.root.N) + 1
    _assert_no_residue(m.root)

    m.run(40, batch_size=4)  # the tree is still searchable
    assert m.completed_simulations == calls[0] + calls[1] + 40
    _assert_no_residue(m.root)


def test_a_stale_in_flight_mark_fails_loudly_instead_of_spinning():
    game = TicTacToe()
    m = MCTS(game)
    m.set_root(game.initial_state())
    m._descend(apply_vloss=True)  # a hand-driven descent that is never backed up

    with pytest.raises(RuntimeError):
        m.run(1)


def test_batched_search_draws_root_noise_once_per_root_incarnation():
    game = Connect4(4, 4, 3)
    rng = random.Random(4)
    rec = _Recorder()
    m = rec.mcts = MCTS(game, evaluate_batch=rec, root_noise=(EPS, ALPHA_NUMERATOR, rng))
    start = game.initial_state()

    m.run(60, root_state=start, batch_size=8)
    mixed = list(m.root.P)
    n = len(mixed)
    assert mixed != [1.0 / n] * n

    # The noise is already in the priors when the first batch that selects from the root
    # descends. That batch sends one descent down each root edge: edge 0 first (no visits
    # yet, so U = 0 everywhere), then the rest in order of decreasing *noised* prior.
    by_noised_prior = [0, *sorted(range(1, n), key=lambda k: (-mixed[k], k))]
    assert by_noised_prior != list(range(n))  # the clean priors would give index order
    assert rec.calls[1] == [game.apply(start, m.root.actions[k]) for k in by_noised_prior]

    # Exactly one Dirichlet draw was consumed, however many batches ran, and a further
    # run on the same root never re-mixes.
    replay = random.Random(4)
    MCTS._dirichlet(n, ALPHA_NUMERATOR / n, replay)
    assert rng.getstate() == replay.getstate()
    m.run(60, batch_size=8)
    assert m.root.P == mixed

    m.advance(m.best_action())  # a new incarnation draws again
    clean = list(m.root.P)
    m.run(20, batch_size=8)
    assert m.root.P != clean
