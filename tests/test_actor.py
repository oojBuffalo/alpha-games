"""The M3 self-play actor driver: ``core/actor.py`` (§12 M3, issue #59).

``core.selfplay.play_game`` is exercised elsewhere (``tests/test_micro_loop.py``);
this battery covers the two things layered on top of it here: the D6
config gate (:func:`validate_actor_self_play_config`), and
:class:`~core.actor.ActorDriver` — durable per-actor game identity sourced
only from ``ShardWriter``'s persisted state, crash/restart safety, the
refresh-between-games and pacing discipline, and end-to-end determinism.

Kept CPU-cheap throughout: Tic-Tac-Toe with the M0 uniform-prior evaluator
(``evaluator=None``) at the pinned 128 sims — TTT's tree is tiny enough that
128 sims/move costs nothing, so every test can legitimately construct
:class:`~core.actor.ActorDriver`, whose D6 gate requires exactly that sim
count, rather than working around it.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from core.actor import (
    VALIDATE_TIER_SIMS,
    ActorDriver,
    validate_actor_self_play_config,
)
from core.replay_shard import read_shard, shard_filename
from core.runconfig import load_run_config
from games.tictactoe import TicTacToe

TTT = TicTacToe()


def self_play_cfg(**overrides):
    """Build a :class:`~core.runconfig.SelfPlayConfig` off the pinned micro one.

    Args:
        **overrides: Fields to replace (e.g. ``sims``).

    Returns:
        The overridden config.
    """
    return dataclasses.replace(load_run_config().self_play, **overrides)


def refresh_sequence(versions):
    """Build a ``refresh`` callable that yields ``(None, version)`` in order.

    Args:
        versions: The model versions to hand out, one per call.

    Returns:
        A zero-argument callable; raises ``StopIteration`` if called more
        times than ``len(versions)`` (a test bug, not a driver one).
    """
    it = iter(versions)
    return lambda: (None, next(it))


def make_driver(
    tmp_path, *, run_id="run", actor_id=0, run_seed=1, sims=VALIDATE_TIER_SIMS, **kwargs
):
    """Build an :class:`~core.actor.ActorDriver` over TTT with sensible defaults.

    Args:
        tmp_path: Destination directory for shards (typically pytest's
            ``tmp_path``, or a subdirectory of it).
        run_id: Run identity.
        actor_id: Actor identity within the run.
        run_seed: The run's root seed.
        sims: Self-play sim count (default: the D6 validate tier).
        **kwargs: Forwarded to :class:`~core.actor.ActorDriver` (``refresh``,
            ``pacing``, ``wait``, ``max_games``, ``should_stop``).

    Returns:
        The constructed driver.
    """
    kwargs.setdefault("refresh", refresh_sequence([1] * 10))
    return ActorDriver(
        game=TTT,
        self_play=self_play_cfg(sims=sims),
        run_id=run_id,
        actor_id=actor_id,
        out_dir=tmp_path,
        run_seed=run_seed,
        **kwargs,
    )


# --- D6 config validation ---------------------------------------------------


def test_validate_accepts_exactly_128_sims():
    validate_actor_self_play_config(self_play_cfg(sims=128))


@pytest.mark.parametrize("bad_sims", [1, 64, 127, 129, 256, 512])
def test_validate_rejects_any_other_sim_count(bad_sims):
    with pytest.raises(ValueError, match="128"):
        validate_actor_self_play_config(self_play_cfg(sims=bad_sims))


def test_validate_rejects_playout_cap_randomization_when_present_and_true():
    """Forward-compat: a future SelfPlayConfig.playout_cap_randomization field."""
    stub = SimpleNamespace(sims=128, playout_cap_randomization=True)
    with pytest.raises(ValueError, match="playout-cap"):
        validate_actor_self_play_config(stub)


def test_validate_accepts_playout_cap_randomization_when_present_and_false():
    stub = SimpleNamespace(sims=128, playout_cap_randomization=False)
    validate_actor_self_play_config(stub)


def test_validate_accepts_its_absence_today():
    """SelfPlayConfig carries no PCR field yet (D8/M5) -- absence must pass."""
    cfg = self_play_cfg(sims=128)
    assert not hasattr(cfg, "playout_cap_randomization")
    validate_actor_self_play_config(cfg)


def test_actor_driver_construction_enforces_the_same_gate(tmp_path):
    with pytest.raises(ValueError, match="128"):
        make_driver(tmp_path, sims=64)
    driver = make_driver(tmp_path, sims=128)
    assert driver.writer.state.next_game_index == 0


def test_actor_driver_rejects_a_non_positive_max_games(tmp_path):
    with pytest.raises(ValueError, match="max_games"):
        make_driver(tmp_path, max_games=0)
    with pytest.raises(ValueError, match="max_games"):
        make_driver(tmp_path, max_games=-1)


# --- full sample record ------------------------------------------------------


def test_samples_carry_mover_pinned_version_and_durable_game_id(tmp_path):
    """Mover-relative backfill wiring, one pinned version/game, durable ids."""
    driver = make_driver(
        tmp_path, run_id="run-a", actor_id=3, refresh=refresh_sequence([5, 5, 9]), max_games=3
    )
    paths = driver.run()
    assert len(paths) == 3

    expected_versions = [5, 5, 9]
    for game_index, path in enumerate(paths):
        data = read_shard(path, TTT)
        assert data.records, "an empty game must never publish an empty shard"
        for record in data.records:
            assert record.mover in (0, 1)
            assert record.model_version == expected_versions[game_index]
            assert record.game_id == ("run-a", "3", game_index)

    # The pinned version changed between game 1 and game 2, never within one.
    assert {r.model_version for r in read_shard(paths[0], TTT).records} == {5}
    assert {r.model_version for r in read_shard(paths[2], TTT).records} == {9}


def test_every_ply_is_stored_no_d12_dropping(tmp_path):
    """M3 stores every position (M5 owns the fast-tier drop policy)."""
    driver = make_driver(tmp_path, max_games=4)
    for path in driver.run():
        plies = [r.ply for r in read_shard(path, TTT).records]
        assert plies == list(range(len(plies)))
        assert len(plies) >= 5  # a finished TTT game is never shorter than 5 plies


# --- restart safety -----------------------------------------------------------


def test_restart_continues_game_indices_without_reissue(tmp_path):
    first = make_driver(tmp_path, run_id="run-b", actor_id=1, max_games=3)
    first_paths = first.run()
    first_indices = [read_shard(p, TTT).records[0].game_id[2] for p in first_paths]
    assert first_indices == [0, 1, 2]

    # "Restart": a brand-new driver instance over the same durable coordinates.
    second = make_driver(tmp_path, run_id="run-b", actor_id=1, max_games=2)
    second_paths = second.run()
    second_indices = [read_shard(p, TTT).records[0].game_id[2] for p in second_paths]
    assert second_indices == [3, 4]

    all_paths = first_paths + second_paths
    assert [p.name for p in all_paths] == [shard_filename("run-b", "1", seq) for seq in range(5)]


def test_crash_window_burns_an_index_without_reissue(tmp_path):
    driver = make_driver(tmp_path, run_id="run-c", actor_id=2, max_games=1)
    published = driver.run()
    assert read_shard(published[0], TTT).records[0].game_id[2] == 0

    # Simulate a crash mid ``write_shard``: state durably advances, the shard
    # is never published -- exactly ShardWriter's own crash window.
    burned_seq, burned_index = driver.writer._reserve(1)
    assert burned_index == 1
    burned_path = tmp_path / shard_filename("run-c", "2", burned_seq)
    assert not burned_path.exists()

    # A fresh driver ("restart") reloads the already-advanced persisted state.
    restarted = make_driver(tmp_path, run_id="run-c", actor_id=2, max_games=1)
    restarted_paths = restarted.run()
    next_index = read_shard(restarted_paths[0], TTT).records[0].game_id[2]
    assert next_index == burned_index + 1 == 2
    assert not burned_path.exists()  # the burned index was never reissued


# --- refresh discipline -------------------------------------------------------


def test_refresh_is_called_exactly_once_between_each_pair_of_games(tmp_path):
    refresh_calls = []

    class _TrackingEvaluator:
        """Wraps the M0 uniform-prior path while flagging reentrancy."""

        def __init__(self):
            self.in_flight = 0

        def __call__(self, game, state):
            self.in_flight += 1
            try:
                return (0.0, None)
            finally:
                self.in_flight -= 1

    evaluator = _TrackingEvaluator()
    versions = iter([1, 1, 2, 2, 2])

    def refresh():
        # The trip wire: a refresh invoked while a leaf evaluation is on the
        # call stack (e.g. a future refactor that called it from inside the
        # search) would fail this before ever reaching a game boundary.
        assert evaluator.in_flight == 0, "refresh invoked while a search was in flight"
        version = next(versions)
        refresh_calls.append(version)
        return evaluator, version

    driver = make_driver(tmp_path, refresh=refresh, max_games=5)
    paths = driver.run()

    assert refresh_calls == [1, 1, 2, 2, 2]  # called once per game, before it
    assert [read_shard(p, TTT).records[0].model_version for p in paths] == refresh_calls


def test_pinned_model_version_is_missing_raises_at_publish(tmp_path):
    """A refresh that forgets to pin a version must fail loudly, not silently."""
    driver = make_driver(tmp_path, refresh=lambda: (None, None), max_games=1)
    with pytest.raises(ValueError, match="model_version"):
        driver.run()


# --- pacing --------------------------------------------------------------------


def test_no_pacing_hook_produces_games_immediately(tmp_path):
    driver = make_driver(tmp_path, pacing=None, max_games=2)
    assert len(driver.run()) == 2


def test_pacing_false_never_waits(tmp_path):
    waits = []
    driver = make_driver(tmp_path, pacing=lambda: False, wait=lambda: waits.append(1), max_games=2)
    assert len(driver.run()) == 2
    assert waits == []


def test_pacing_true_holds_until_the_wait_strategy_flips_it(tmp_path):
    hold = {"value": True}
    events = []

    def pacing():
        return hold["value"]

    def wait():
        events.append("wait")
        if len(events) == 3:
            hold["value"] = False

    def refresh():
        events.append("refresh")
        return None, 1

    driver = make_driver(tmp_path, pacing=pacing, wait=wait, refresh=refresh, max_games=1)
    paths = driver.run()

    assert len(paths) == 1
    assert events == ["wait", "wait", "wait", "refresh"]  # no game starts before the flip


def test_pacing_is_polled_between_every_game_not_only_the_first(tmp_path):
    """The hold fires once, positioned strictly between game 0 and game 1."""
    state = {"games_done": 0, "held_once": False}

    def pacing():
        return state["games_done"] == 1 and not state["held_once"]

    def wait():
        state["held_once"] = True

    def refresh():
        state["games_done"] += 1
        return None, 1

    driver = make_driver(tmp_path, pacing=pacing, wait=wait, refresh=refresh, max_games=3)
    paths = driver.run()
    assert len(paths) == 3
    assert state["held_once"] is True


# --- stop conditions ------------------------------------------------------------


def test_should_stop_callable_stops_the_loop(tmp_path):
    checks = {"n": 0}

    def should_stop():
        checks["n"] += 1
        return checks["n"] > 2

    driver = make_driver(tmp_path, max_games=None, should_stop=should_stop)
    paths = driver.run()
    assert len(paths) == 2
    assert checks["n"] == 3


def test_should_stop_and_max_games_combine_as_whichever_fires_first(tmp_path):
    driver = make_driver(tmp_path, max_games=5, should_stop=lambda: True)
    assert driver.run() == []


# --- determinism -----------------------------------------------------------------


def _flatten_records(paths):
    """Read every shard path in order and concatenate their records.

    Args:
        paths: Shard paths, in play order.

    Returns:
        The concatenated records, in play order.
    """
    records = []
    for path in paths:
        records.extend(read_shard(path, TTT).records)
    return records


def test_restart_vs_continuous_are_bit_identical_per_game_index(tmp_path):
    continuous = make_driver(
        tmp_path / "continuous",
        run_id="run-d",
        actor_id=4,
        run_seed=555,
        refresh=refresh_sequence([1, 1, 2]),
        max_games=3,
    )
    continuous_records = _flatten_records(continuous.run())

    restart_dir = tmp_path / "restart"
    first = make_driver(
        restart_dir,
        run_id="run-d",
        actor_id=4,
        run_seed=555,
        refresh=refresh_sequence([1]),
        max_games=1,
    )
    first_paths = first.run()
    second = make_driver(
        restart_dir,
        run_id="run-d",
        actor_id=4,
        run_seed=555,
        refresh=refresh_sequence([1, 2]),
        max_games=2,
    )
    second_paths = second.run()
    restart_records = _flatten_records(first_paths + second_paths)

    assert len(continuous_records) == len(restart_records)
    for c, r in zip(continuous_records, restart_records, strict=True):
        assert np.array_equal(np.asarray(c.planes), np.asarray(r.planes))
        assert c.sparse_pi == r.sparse_pi
        assert c.z == r.z
        assert c.aux == r.aux
        assert c.mover == r.mover
        assert c.model_version == r.model_version
        assert c.ply == r.ply
        assert c.game_id[2] == r.game_id[2]  # same durable index, different run_id


def test_different_actor_id_decorrelates_the_same_game_index(tmp_path):
    a = make_driver(tmp_path / "a", run_id="run-e", actor_id=0, run_seed=42, max_games=1)
    b = make_driver(tmp_path / "b", run_id="run-e", actor_id=1, run_seed=42, max_games=1)
    records_a = _flatten_records(a.run())
    records_b = _flatten_records(b.run())
    moves_a = [(r.mover, r.ply, r.sparse_pi) for r in records_a]
    moves_b = [(r.mover, r.ply, r.sparse_pi) for r in records_b]
    assert moves_a != moves_b


# --- signal-shutdown wiring (issue #61): should_stop honored during a pacing hold ---


def test_should_stop_firing_during_a_pacing_hold_exits_without_a_new_game(tmp_path):
    """A hold has no other exit condition -- should_stop must be polled inside it."""
    events = []

    def pacing():
        return True  # holds forever unless should_stop cuts it short

    def wait():
        events.append("wait")

    def should_stop():
        return len(events) >= 2  # fires partway through the hold

    driver = make_driver(tmp_path, pacing=pacing, wait=wait, should_stop=should_stop)
    paths = driver.run()

    assert paths == []  # no game was ever in flight, so none gets played
    assert events == ["wait", "wait"]


def test_should_stop_firing_during_a_pacing_hold_still_finishes_an_in_flight_game(tmp_path):
    """should_stop stops *production*, never an already-in-flight game."""
    events = []
    state = {"held": True, "games_started": 0}

    def pacing():
        return state["held"]

    def wait():
        events.append("wait")
        if len(events) == 2:
            state["held"] = False  # release the hold on the next poll

    def refresh():
        state["games_started"] += 1
        return None, 1

    def should_stop():
        # True only once a game has actually started -- proves the hold's
        # should_stop check never pre-empts a game already under way, while
        # still stopping production immediately after it.
        return state["games_started"] >= 1

    driver = make_driver(
        tmp_path, pacing=pacing, wait=wait, refresh=refresh, should_stop=should_stop
    )
    paths = driver.run()

    assert len(paths) == 1
    assert events == ["wait", "wait"]


def test_should_stop_call_count_is_unchanged_when_no_pacing_hook_is_installed(tmp_path):
    """The run()-level re-check only fires when a pacing hook exists (no extra polls)."""
    checks = {"n": 0}

    def should_stop():
        checks["n"] += 1
        return checks["n"] > 2

    driver = make_driver(tmp_path, pacing=None, max_games=None, should_stop=should_stop)
    paths = driver.run()
    assert len(paths) == 2
    assert checks["n"] == 3
