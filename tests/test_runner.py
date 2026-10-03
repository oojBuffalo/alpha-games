"""Evaluation game runner (M1.6): single games, then mirrored pairs.

The runner drives seeded agents through the public ``Game`` interface and
reports exact terminal utilities; the §9 protocol details (seat swap, per-pair
seeds, draws 0.5, opening balancing) are layered on top of single games.
"""

from __future__ import annotations

import pytest

from core import RandomAgent
from core.agents import Agent
from core.game import Game
from core.runner import _OpeningRestricted, play_game, play_pairs
from core.seeding import derive_seed
from games.blokus_duo import BlokusDuo
from games.blokus_duo.config import MICRO_CONFIG
from games.tictactoe import TicTacToe

GAME = TicTacToe()


class CenterMinAgent(Agent):
    """Deterministic seat-agnostic rule: take the center if free, else the
    lowest legal cell. Self-play with this rule draws TTT from either seat."""

    @property
    def name(self) -> str:
        return "center-min"

    def select_action(self, game, state):
        moves = list(game.legal_moves(state))
        return 4 if 4 in moves else min(moves)


def test_play_game_reaches_a_zero_sum_terminal():
    rec = play_game(GAME, (RandomAgent(seed=1), RandomAgent(seed=2)))
    assert sum(rec.utilities) == 0.0
    assert all(u in (-1.0, 0.0, 1.0) for u in rec.utilities)
    assert 5 <= rec.plies <= 9


def test_play_game_is_deterministic_given_seeded_agents():
    rec_a = play_game(GAME, (RandomAgent(seed=3), RandomAgent(seed=4)))
    rec_b = play_game(GAME, (RandomAgent(seed=3), RandomAgent(seed=4)))
    assert rec_a == rec_b


def test_play_game_seats_map_to_player_ids():
    # Agent at index p moves exactly when current_player == p: an always-first
    # scripted agent on seat 0 must produce a game whose first move is its pick.
    class Scripted(RandomAgent):
        def __init__(self):
            super().__init__(seed=0)
            self.seen_players = []

        def select_action(self, game, state):
            self.seen_players.append(game.current_player(state))
            return super().select_action(game, state)

    a0, a1 = Scripted(), Scripted()
    play_game(GAME, (a0, a1))
    assert set(a0.seen_players) == {0}
    assert set(a1.seen_players) == {1}


# --- mirrored pairs (§9 protocol) ---------------------------------------------


def test_play_pairs_scores_are_complementary_and_bounded():
    results = play_pairs(
        GAME,
        lambda seed: RandomAgent(seed),
        lambda seed: RandomAgent(seed),
        n_pairs=8,
        seed=123,
    )
    assert len(results) == 8
    for i, pair in enumerate(results):
        assert pair.pair_index == i
        # score_a + score_b == 2 per pair (each game contributes 1 total).
        assert pair.score_a + pair.score_b == 2.0
        assert 0.0 <= pair.score_a <= 2.0
        assert len(pair.games) == 2


def test_play_pairs_is_deterministic_given_master_seed():
    args = (GAME, lambda s: RandomAgent(s), lambda s: RandomAgent(s))
    assert play_pairs(*args, n_pairs=5, seed=9) == play_pairs(*args, n_pairs=5, seed=9)
    assert play_pairs(*args, n_pairs=5, seed=9) != play_pairs(*args, n_pairs=5, seed=10)


def test_play_pairs_swaps_seats_and_reuses_the_pair_seed():
    calls_a, calls_b = [], []

    def factory_a(seed):
        calls_a.append(seed)
        return RandomAgent(seed)

    def factory_b(seed):
        calls_b.append(seed)
        return RandomAgent(seed)

    play_pairs(GAME, factory_a, factory_b, n_pairs=3, seed=0)
    # Each factory is invoked once per game (two per pair), with the same
    # per-pair seed in both games of a pair, and different seeds across pairs.
    assert len(calls_a) == len(calls_b) == 6
    for i in range(3):
        assert calls_a[2 * i] == calls_a[2 * i + 1]
        assert calls_b[2 * i] == calls_b[2 * i + 1]
    assert len(set(calls_a)) == 3


def test_game_records_carry_the_opening_action():
    rec = play_game(GAME, (RandomAgent(seed=1), RandomAgent(seed=2)))
    assert rec.opening in range(9)


def test_opening_balancer_constrains_the_second_game_of_a_pair():
    # Generic hook (§12 M1.6 pin): the balancer sees game 1's opening and
    # returns a predicate restricting game 2's opener. Forcing equality makes
    # both games of every pair open identically.
    def same_opening(game, opening):
        del game
        return lambda a: a == opening

    results = play_pairs(
        GAME,
        lambda s: RandomAgent(s),
        lambda s: RandomAgent(s),
        n_pairs=6,
        seed=77,
        opening_balancer=same_opening,
    )
    for pair in results:
        assert pair.games[0].opening == pair.games[1].opening


def test_draws_score_half_per_game():
    # Center-then-min self-play draws TTT from either seat: both games of the
    # pair end 0/0 and each must contribute exactly 0.5.
    results = play_pairs(
        GAME, lambda s: CenterMinAgent(), lambda s: CenterMinAgent(), n_pairs=1, seed=1
    )
    (pair,) = results
    assert all(rec.utilities == (0.0, 0.0) for rec in pair.games)
    assert pair.score_a == pair.score_b == 1.0  # 0.5 + 0.5 each


def test_play_pairs_rejects_a_zero_game_match():
    # A ladder that runs no games must fail loudly, not emit an empty result
    # that downstream fabricates into a fake "tied random" rating.
    def rand(s):
        return RandomAgent(s)

    for bad in (0, -1):
        with pytest.raises(ValueError):
            play_pairs(GAME, rand, rand, n_pairs=bad, seed=0)


# --- index-keyed pair seeds and resumable windows (M4 §5.1) -------------------


def test_pair_seed_is_the_index_keyed_derivation():
    # pair_seed is a pure function of (seed, pair_index) -- recomputing it via
    # core.seeding.derive_seed must match the value the runner recorded.
    results = play_pairs(
        GAME, lambda s: RandomAgent(s), lambda s: RandomAgent(s), n_pairs=5, seed=42
    )
    for pair in results:
        assert pair.pair_seed == derive_seed(42, "pair", pair.pair_index)
    # Distinct per pair -- a stronger form of the mirrored/distinct-seed
    # property test_play_pairs_swaps_seats_and_reuses_the_pair_seed checks at
    # the factory-call level.
    assert len({pair.pair_seed for pair in results}) == 5


def test_play_pairs_resumption_reproduces_an_uninterrupted_run():
    # Index-keyed seeds are pure functions of (seed, pair_index), independent
    # of stream position, so a resumed window must reproduce byte-for-byte
    # (here: field-for-field) what an uninterrupted run would have played --
    # this is the exact-resumption contract the eval store leans on.
    args = (GAME, lambda s: RandomAgent(s), lambda s: RandomAgent(s))
    seed = 2024
    full = play_pairs(*args, n_pairs=10, seed=seed)
    head = play_pairs(*args, n_pairs=4, seed=seed)
    tail = play_pairs(*args, n_pairs=6, seed=seed, start_pair_index=4)
    assert head + tail == full
    assert [pair.pair_index for pair in tail] == list(range(4, 10))
    assert [pair.pair_index for pair in head] == list(range(0, 4))


def test_play_pairs_rejects_a_negative_start_pair_index():
    # There is no pair before index 0 to resume from.
    def rand(s):
        return RandomAgent(s)

    with pytest.raises(ValueError):
        play_pairs(GAME, rand, rand, n_pairs=1, seed=0, start_pair_index=-1)


# --- reflective delegation audit: every Game ABC member -------------------------------


def test_opening_restricted_delegates_every_game_abc_member():
    """Every abstract *and* concrete ``Game`` member must delegate to the
    wrapped game unchanged (outside the deliberate initial-state opening
    filter) -- so a future ABC addition that this test isn't updated for
    fails loudly here (the ``declared == set(checks)`` guard below) instead
    of silently shipping an undelegated member (as ``orientation_table_hash``/
    ``encoding_conventions`` were before this task, since both are concrete
    on the ABC and Python happily inherits a default for an unoverridden
    concrete method -- no ``TypeError`` the way a missed abstract member
    would raise)."""
    inner = BlokusDuo(config=MICRO_CONFIG)
    wrapper = _OpeningRestricted(inner, lambda a: True)  # accept-all: no filtering effect

    state0 = inner.initial_state()
    a0 = min(inner.legal_moves(state0))
    state1 = inner.apply(state0, a0)  # non-initial, nonterminal: bypasses the filter path
    a1 = min(inner.legal_moves(state1))
    move1 = inner.decode_action(a1)

    terminal = state0
    while not inner.is_terminal(terminal):
        terminal = inner.apply(terminal, min(inner.legal_moves(terminal)))

    def _symmetry_groups_match(wrapped_group, inner_group):
        # (transform, permutation) pairs: the transform is a freshly built
        # closure on every property access (not cached), so two calls never
        # produce `==`-equal callables even when they behave identically --
        # compare permutations directly and transforms by their output on a
        # real encoded state instead of by object identity.
        sample_planes = inner.encode_state(state1)
        if len(wrapped_group) != len(inner_group):
            return False
        for (t_w, perm_w), (t_i, perm_i) in zip(wrapped_group, inner_group, strict=True):
            if tuple(perm_w) != tuple(perm_i):
                return False
            if t_w(sample_planes) != t_i(sample_planes):
                return False
        return True

    checks = {
        # declared capabilities
        "num_players": (lambda g: g.num_players, None),
        "is_stochastic": (lambda g: g.is_stochastic, None),
        "is_perfect_information": (lambda g: g.is_perfect_information, None),
        "symmetry_group": (lambda g: g.symmetry_group, _symmetry_groups_match),
        "value_targets": (lambda g: g.value_targets, None),
        # fingerprint surface
        "orientation_table_hash": (lambda g: g.orientation_table_hash, None),
        "encoding_conventions": (lambda g: g.encoding_conventions, None),
        # core contract
        "initial_state": (lambda g: g.initial_state(), None),
        "current_player": (lambda g: g.current_player(state1), None),
        "legal_moves": (lambda g: list(g.legal_moves(state1)), None),
        "apply": (lambda g: g.apply(state1, a1), None),
        "is_terminal": (lambda g: g.is_terminal(state1), None),
        "terminal_utility": (lambda g: g.terminal_utility(terminal, 0), None),
        "training_targets": (lambda g: g.training_targets(terminal, 0), None),
        # encoding surface
        "encode_state": (lambda g: g.encode_state(state1), None),
        "encode_action": (lambda g: g.encode_action(move1), None),
        "decode_action": (lambda g: g.decode_action(a1), None),
        "policy_shape": (lambda g: g.policy_shape, None),
        "input_planes": (lambda g: g.input_planes, None),
        "input_shape": (lambda g: g.input_shape, None),
    }

    declared = {name for name in vars(Game) if not name.startswith("_")}
    assert declared == set(checks), (
        f"Game ABC members missing from this delegation audit: {declared - set(checks)}; "
        f"stale entries no longer on the ABC: {set(checks) - declared}"
    )

    for name, (call, compare) in checks.items():
        got, want = call(wrapper), call(inner)
        if compare is None:
            assert got == want, name
        else:
            assert compare(got, want), name
