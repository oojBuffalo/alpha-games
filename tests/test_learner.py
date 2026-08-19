"""The M3 learner driver: ``core/learner.py`` (§12 M3, issue #60).

Three layers:

1. **The metrics primitive** ``core/learner.py`` depends on
   (``core.metrics``): per-process epoch files, restart-bumps-epoch, and the
   cross-epoch record iterator the exactly-once marker check needs.
2. **Ratio enforcement + pacing** (Tic-Tac-Toe, synthetic shards -- fast,
   game-mechanics-irrelevant): warm-up defers both the ceiling and the
   floor; the ceiling blocks on ``wait`` until ingestion; the floor writes
   ``hold``/``go`` and the exact ratio arithmetic.
3. **Publication, the marker, augmentation, and the resume golden** (real
   micro-Blokus shards, a tiny net -- exercises the real D9 augmentation
   group and the real checkpoint/replay stack): version-0-at-fresh-start,
   the publish/step sequence, immutability, resume never re-publishing,
   the exact stop condition, seeded per-step determinism, and the
   bit-for-bit resume-equivalence golden -- including the straddle case
   where a crash lands between a publish and its following resume snapshot.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from core.checkpoint import (
    list_published_versions,
    load_checkpoint,
    read_latest_pointer,
    resume_path,
)
from core.learner import (
    CHECKPOINT_PUBLISHED_KIND,
    PACING_GO,
    PACING_HOLD,
    LearnerDriver,
    read_pacing_file,
)
from core.metrics import EpochMetricsWriter, epoch_metrics_path, iter_epoch_records, next_epoch
from core.network import NetworkConfig
from core.observability import reduce_run
from core.replay_shard import SampleRecord, shard_filename, write_shard
from core.runconfig import TrainingConfig, load_run_config
from games.blokus_duo import BlokusDuo
from games.blokus_duo.config import MICRO_CONFIG
from games.tictactoe import TicTacToe

TTT = TicTacToe()
TTT_NET_CONFIG = NetworkConfig(2, (3, 3), (9,), trunk_blocks=1, trunk_channels=4)

MICRO = BlokusDuo(config=MICRO_CONFIG)
MICRO_NUM_AUX = len(MICRO.value_targets.aux_names)
MICRO_GROUP_SIZE = len(MICRO.symmetry_group)


def _micro_net_config():
    """A tiny net over the micro-Blokus encoding surface -- speed, not throughput."""
    base = NetworkConfig.from_game(MICRO)
    return dataclasses.replace(base, trunk_blocks=1, trunk_channels=4)


MICRO_NET_CONFIG = _micro_net_config()


# --- TTT synthetic shard fixtures (mirrors tests/test_replay_window.py) ------


def _ttt_records(run_id, actor_id, game_index, num_positions):
    """Build ``num_positions`` trivially-valid TTT sample records.

    Real board reachability does not matter -- only the array invariants
    ``write_shard`` checks (mirrors ``tests/test_replay_window.py``).
    """
    state = TTT.initial_state()
    planes = TTT.encode_state(state)
    legal_action = next(iter(TTT.legal_moves(state)))
    return tuple(
        SampleRecord(
            planes=planes,
            sparse_pi=((legal_action, 1),),
            z=0.0,
            aux=(),
            mover=0,
            model_version=0,
            ply=ply,
            game_id=(run_id, actor_id, game_index),
        )
        for ply in range(num_positions)
    )


def _write_ttt_shard(shard_dir, run_id, actor_id, seq, num_positions):
    """Write one synthetic TTT shard of exactly ``num_positions`` records."""
    records = _ttt_records(run_id, actor_id, seq, num_positions)
    shard_id = shard_filename(run_id, actor_id, seq)
    write_shard(shard_dir / shard_id, TTT, records, run_id=run_id, actor_id=actor_id, seq=seq)
    return shard_id


def ttt_training_config(**overrides):
    """A ``TrainingConfig`` sized for fast TTT-based learner tests."""
    base = dict(
        games=1,
        learner_steps=1,
        steps_per_game=1,
        batch_size=4,
        replay_window=10_000,
        learning_rate=1e-2,
        warmup_steps=0,
        cosine_total_steps=100,
        aux_loss_weight=0.0,
        checkpoint_selection="final",
        publish_interval=1000,
        checkpoint_count=1,
        replay_warmup_positions=1,
    )
    base.update(overrides)
    return TrainingConfig(**base)


def stub_run_config(training, run_seed=1):
    """A minimal duck-typed run-config stand-in for TTT tests.

    ``core.runconfig.RunConfig`` cannot itself name Tic-Tac-Toe (only
    ``blokus_duo`` declares ``GAME_CONFIGS`` today), so TTT-only tests build
    the small subset ``LearnerDriver`` actually reads (``training``,
    ``run_seed``, ``to_dict``) directly -- the same duck-typing
    ``tests/test_actor.py`` uses for ``SelfPlayConfig`` stand-ins.
    """
    return SimpleNamespace(
        training=training,
        run_seed=run_seed,
        to_dict=lambda: {"training": dataclasses.asdict(training), "run_seed": run_seed},
    )


def make_ttt_driver(tmp_path, *, training=None, run_seed=1, subdir="run", **kwargs):
    """Build a :class:`~core.learner.LearnerDriver` over TTT with a tiny net."""
    training = training if training is not None else ttt_training_config()
    root = tmp_path / subdir
    return LearnerDriver(
        game=TTT,
        run_config=stub_run_config(training, run_seed=run_seed),
        shard_dir=kwargs.pop("shard_dir", root / "shards"),
        ckpt_dir=root / "ckpt",
        run_dir=root / "run",
        network_config=TTT_NET_CONFIG,
        **kwargs,
    )


# --- real micro-Blokus shard fixtures (mirrors tests/test_checkpoint.py) ----


def _real_micro_records(run_id, actor_id, game_index, n, seed):
    """``n`` structurally real samples from an actual micro-Blokus rollout."""
    import random

    rng = random.Random(seed)
    state = MICRO.initial_state()
    records = []
    for ply in range(n):
        if MICRO.is_terminal(state):
            break
        legal = list(MICRO.legal_moves(state))
        ids = rng.sample(legal, min(len(legal), rng.randint(1, min(4, len(legal)))))
        counts = [rng.randint(1, 5) for _ in ids]
        records.append(
            SampleRecord(
                planes=MICRO.encode_state(state),
                sparse_pi=tuple(zip(ids, counts, strict=True)),
                z=rng.choice([-1.0, 0.0, 1.0]),
                aux=(rng.uniform(-1.0, 1.0),) if MICRO_NUM_AUX else (),
                mover=MICRO.current_player(state),
                model_version=0,
                ply=ply,
                game_id=(run_id, actor_id, game_index),
            )
        )
        state = MICRO.apply(state, min(legal))
    return tuple(records)


def write_real_micro_shards(shard_dir, *, n_shards=8, positions_per_shard=7, seed=1000):
    """Populate ``shard_dir`` with real, on-disk, fingerprint-valid micro shards.

    Returns:
        Total positions written (may be less than ``n_shards *
        positions_per_shard`` if a rollout terminates early).
    """
    total = 0
    for i in range(n_shards):
        actor = f"actor-{i}"
        records = _real_micro_records("golden-run", actor, 0, positions_per_shard, seed + i)
        write_shard(
            shard_dir / f"shard-golden-run-{actor}-0.npz",
            MICRO,
            records,
            run_id="golden-run",
            actor_id=actor,
            seq=0,
        )
        total += len(records)
    return total


def tiny_micro_run_config(**training_overrides):
    """A tiny but coherent, real ``RunConfig`` for micro-Blokus learner tests."""
    cfg = load_run_config()
    training = dict(
        batch_size=4,
        replay_window=1000,
        learning_rate=1e-3,
        warmup_steps=0,
        cosine_total_steps=100,
        publish_interval=2,
        checkpoint_count=2,
        replay_warmup_positions=1,
    )
    training.update(training_overrides)
    return dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, **training))


def make_micro_driver(shard_dir, ckpt_dir, run_dir, *, run_config=None, run_seed=None, **kwargs):
    """Build a :class:`~core.learner.LearnerDriver` over micro-Blokus with a tiny net."""
    cfg = run_config if run_config is not None else tiny_micro_run_config()
    if run_seed is not None:
        cfg = dataclasses.replace(cfg, run_seed=run_seed)
    return LearnerDriver(
        game=MICRO,
        run_config=cfg,
        shard_dir=shard_dir,
        ckpt_dir=ckpt_dir,
        run_dir=run_dir,
        network_config=MICRO_NET_CONFIG,
        **kwargs,
    )


# ==============================================================================
# 1. core.metrics: the epoch-file primitive
# ==============================================================================


def test_max_steps_must_be_positive(tmp_path):
    shard_dir = tmp_path / "shards"
    _write_ttt_shard(shard_dir, "run", "a", 0, 5)
    with pytest.raises(ValueError, match="max_steps"):
        make_ttt_driver(tmp_path, shard_dir=shard_dir, max_steps=0)
    with pytest.raises(ValueError, match="max_steps"):
        make_ttt_driver(tmp_path, shard_dir=shard_dir, max_steps=-1)


def test_epoch_writer_starts_at_epoch_zero_and_records_round_trip(tmp_path):
    writer = EpochMetricsWriter(tmp_path, "learner")
    assert writer.epoch == 0
    assert writer.path == epoch_metrics_path(tmp_path, "learner", 0)
    writer.append({"kind": "x", "value": 1})
    writer.append({"kind": "x", "value": 2})
    records = list(iter_epoch_records(tmp_path, "learner"))
    assert records == [{"kind": "x", "value": 1}, {"kind": "x", "value": 2}]


def test_a_restart_opens_a_new_epoch_file(tmp_path):
    first = EpochMetricsWriter(tmp_path, "learner")
    first.append({"kind": "x", "value": 1})
    second = EpochMetricsWriter(tmp_path, "learner")
    assert second.epoch == 1
    assert second.path != first.path
    second.append({"kind": "x", "value": 2})

    assert next_epoch(tmp_path, "learner") == 2
    records = list(iter_epoch_records(tmp_path, "learner"))
    assert records == [{"kind": "x", "value": 1}, {"kind": "x", "value": 2}]  # epoch order


def test_two_writers_never_share_a_file(tmp_path):
    learner = EpochMetricsWriter(tmp_path, "learner")
    actor = EpochMetricsWriter(tmp_path, "actor-0")
    assert learner.path != actor.path
    learner.append({"kind": "x"})
    assert list(iter_epoch_records(tmp_path, "actor-0")) == []


def test_iter_epoch_records_on_an_untouched_directory_is_empty(tmp_path):
    assert list(iter_epoch_records(tmp_path, "learner")) == []
    assert next_epoch(tmp_path, "learner") == 0


# ==============================================================================
# 2. Pacing file
# ==============================================================================


def test_read_pacing_file_before_any_write_is_none(tmp_path):
    assert read_pacing_file(tmp_path / "pacing.json") is None


def test_write_pacing_file_round_trips_every_field(tmp_path):
    from core.learner import pacing_file_path, write_pacing_file

    path = pacing_file_path(tmp_path)
    write_pacing_file(
        path, state=PACING_HOLD, ratio=0.25, positions_stored=40, samples_drawn=10, learner_step=5
    )
    payload = read_pacing_file(path)
    assert payload == {
        "state": PACING_HOLD,
        "ratio": 0.25,
        "positions_stored": 40,
        "samples_drawn": 10,
        "learner_step": 5,
    }
    # Overwriting refreshes the same file atomically -- not append-only.
    write_pacing_file(
        path, state=PACING_GO, ratio=3.0, positions_stored=40, samples_drawn=120, learner_step=30
    )
    assert read_pacing_file(path)["state"] == PACING_GO


# ==============================================================================
# 3. Ratio enforcement (TTT, synthetic shards)
# ==============================================================================


def test_warmup_defers_both_the_ceiling_and_the_floor(tmp_path):
    shard_dir = tmp_path / "shards"
    _write_ttt_shard(shard_dir, "run", "a", 0, 1)  # positions_stored = 1

    def wait():
        raise AssertionError("the ceiling's wait must never fire during warm-up")

    training = ttt_training_config(batch_size=10, replay_warmup_positions=50)
    driver = make_ttt_driver(tmp_path, training=training, shard_dir=shard_dir, wait=wait)
    driver._run_step()

    assert driver.step == 1
    assert read_pacing_file(driver.pacing_path) is None  # the floor never wrote anything


def test_ceiling_blocks_on_wait_and_rescans_until_ingestion_clears_it(tmp_path):
    shard_dir = tmp_path / "shards"
    _write_ttt_shard(shard_dir, "run", "a", 0, 2)  # positions_stored = 2
    calls = {"n": 0}

    def wait():
        calls["n"] += 1
        if calls["n"] == 3:
            _write_ttt_shard(shard_dir, "run", "b", 0, 8)  # -> positions_stored = 10

    training = ttt_training_config(batch_size=10, replay_warmup_positions=1)
    driver = make_ttt_driver(tmp_path, training=training, shard_dir=shard_dir, wait=wait)
    driver._run_step()

    assert calls["n"] == 3  # blocked exactly until ingestion caught it up
    assert driver.step == 1
    assert driver.window.positions_stored == 10


def test_ceiling_never_blocks_when_already_within_band(tmp_path):
    shard_dir = tmp_path / "shards"
    _write_ttt_shard(shard_dir, "run", "a", 0, 100)

    def wait():
        raise AssertionError("must not block: prospective ratio is well under the ceiling")

    training = ttt_training_config(batch_size=4, replay_warmup_positions=1)
    driver = make_ttt_driver(tmp_path, training=training, shard_dir=shard_dir, wait=wait)
    driver._run_step()
    assert driver.step == 1


def test_ceiling_should_stop_escapes_the_wait_loop_and_the_step_still_completes(tmp_path):
    """A learner shutting down while permanently ceiling-blocked must not hang forever.

    Issue #61's signal-shutdown wiring needs ``should_stop`` polled from
    inside the ceiling's wait loop -- otherwise a learner blocked on actors
    that have genuinely stalled could never observe a shutdown signal. The
    in-flight step must still run to completion once the escape fires (the
    module docstring's "Stop condition" contract: ``run`` only stops
    *between* steps).
    """
    shard_dir = tmp_path / "shards"
    _write_ttt_shard(shard_dir, "run", "a", 0, 2)  # positions_stored = 2, never grows
    calls = {"n": 0}

    def wait():
        calls["n"] += 1

    def should_stop():
        return calls["n"] >= 3

    training = ttt_training_config(batch_size=10, replay_warmup_positions=1)
    driver = make_ttt_driver(
        tmp_path, training=training, shard_dir=shard_dir, wait=wait, should_stop=should_stop
    )
    driver._run_step()  # would block forever pre-#61 without the should_stop escape

    assert calls["n"] == 3  # stopped waiting the instant should_stop fired
    assert driver.step == 1  # the in-flight step still ran to completion


def test_run_blocks_until_the_first_shard_lands_then_trains(tmp_path):
    """Warm-up waives ratio *enforcement*, never whether training can start.

    A freshly started learner racing a freshly started actor (issue #61) has
    no other guarantee any shard exists yet when ``run`` starts -- this is
    the wait that resolves it.
    """
    shard_dir = tmp_path / "shards"
    calls = {"n": 0}

    def wait():
        calls["n"] += 1
        if calls["n"] == 2:
            _write_ttt_shard(shard_dir, "run", "a", 0, 3)

    training = ttt_training_config(batch_size=4, replay_warmup_positions=1)
    driver = make_ttt_driver(
        tmp_path, training=training, shard_dir=shard_dir, wait=wait, max_steps=1
    )
    results = driver.run()

    assert calls["n"] == 2  # blocked exactly until the first shard was ingested
    assert len(results) == 1
    assert driver.step == 1


def test_run_with_should_stop_before_any_data_ever_arrives_trains_nothing(tmp_path):
    """A shutdown signal that beats every actor to the punch must not crash."""
    shard_dir = tmp_path / "shards"  # never populated

    def wait():
        pass  # nothing ever appears; should_stop is what ends the wait

    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] >= 2

    training = ttt_training_config(batch_size=4, replay_warmup_positions=1)
    driver = make_ttt_driver(
        tmp_path, training=training, shard_dir=shard_dir, wait=wait, should_stop=should_stop
    )
    results = driver.run()

    assert results == []  # never trained a step against an empty window
    assert driver.step == 0


def test_floor_writes_hold_below_two_and_clears_to_go_at_two(tmp_path):
    shard_dir = tmp_path / "shards"
    _write_ttt_shard(shard_dir, "run", "a", 0, 100)  # positions_stored = 100, static

    training = ttt_training_config(batch_size=50, replay_warmup_positions=1)
    driver = make_ttt_driver(tmp_path, training=training, shard_dir=shard_dir)

    expected_states = [PACING_HOLD, PACING_HOLD, PACING_HOLD, PACING_GO]
    expected_ratios = [0.5, 1.0, 1.5, 2.0]
    for i in range(4):
        driver._run_step()
        payload = read_pacing_file(driver.pacing_path)
        assert payload["state"] == expected_states[i]
        assert payload["ratio"] == pytest.approx(expected_ratios[i])
        assert payload["positions_stored"] == 100
        assert payload["samples_drawn"] == (i + 1) * 50  # the derived samples_drawn arithmetic
        assert payload["learner_step"] == i + 1


# ==============================================================================
# 4. Augmentation
# ==============================================================================


def test_empty_symmetry_group_is_skipped_not_drawn_and_discarded(tmp_path):
    """TTT declares no symmetry group; a naive randrange(0) would raise."""
    assert TTT.symmetry_group == ()
    shard_dir = tmp_path / "shards"
    _write_ttt_shard(shard_dir, "run", "a", 0, 20)

    training = ttt_training_config(batch_size=6, replay_warmup_positions=1)
    driver = make_ttt_driver(tmp_path, training=training, shard_dir=shard_dir)
    results = [driver._run_step() for _ in range(3)]
    assert all(torch.isfinite(r.total) for r in results)


def test_ttt_same_seed_reproduces_step_for_step_different_seed_diverges(tmp_path):
    shard_dir = tmp_path / "shards"
    _write_ttt_shard(shard_dir, "run", "a", 0, 20)
    training = ttt_training_config(batch_size=4, replay_warmup_positions=1)

    def losses(run_seed, subdir):
        driver = make_ttt_driver(
            tmp_path, training=training, run_seed=run_seed, subdir=subdir, shard_dir=shard_dir
        )
        return [driver._run_step().total.item() for _ in range(3)]

    losses_a = losses(7, "a")
    losses_b = losses(7, "b")
    assert losses_a == losses_b

    losses_c = losses(8, "c")
    assert losses_a != losses_c


def test_micro_same_seed_reproduces_step_for_step_different_seed_diverges(tmp_path):
    """The real D9 group (Klein-4): augmentation draws are actually exercised."""
    assert MICRO_GROUP_SIZE > 0
    shard_dir = tmp_path / "shards"
    write_real_micro_shards(shard_dir, n_shards=6, positions_per_shard=6)
    cfg = tiny_micro_run_config(replay_warmup_positions=1)

    def losses(run_seed, subdir):
        driver = make_micro_driver(
            shard_dir,
            tmp_path / subdir / "ckpt",
            tmp_path / subdir / "run",
            run_config=cfg,
            run_seed=run_seed,
        )
        return [driver._run_step().total.item() for _ in range(3)]

    losses_a = losses(4242, "a")
    losses_b = losses(4242, "b")
    assert losses_a == losses_b

    losses_c = losses(4243, "c")
    assert losses_a != losses_c


# ==============================================================================
# 5. Publication
# ==============================================================================


def test_version_zero_publishes_at_fresh_startup(tmp_path):
    shard_dir = tmp_path / "shards"
    write_real_micro_shards(shard_dir)
    ckpt_dir, run_dir = tmp_path / "ckpt", tmp_path / "run"
    driver = make_micro_driver(shard_dir, ckpt_dir, run_dir)

    assert list_published_versions(ckpt_dir) == (0,)
    assert read_latest_pointer(ckpt_dir) == 0
    loaded = load_checkpoint(ckpt_dir / "ckpt-0.pt", MICRO)
    assert loaded.version == 0
    assert loaded.learner_step == 0

    markers = [
        r for r in iter_epoch_records(run_dir, "learner") if r["kind"] == CHECKPOINT_PUBLISHED_KIND
    ]
    assert len(markers) == 1
    assert markers[0]["model_version"] == 0
    assert markers[0]["learner_step"] == 0
    assert set(markers[0]) == {"kind", "model_version", "learner_step", "timestamp"}
    assert driver.step == 0  # no training happened yet


def test_publish_sequence_and_stop_at_total_steps_exactly(tmp_path):
    shard_dir = tmp_path / "shards"
    write_real_micro_shards(shard_dir)
    ckpt_dir, run_dir = tmp_path / "ckpt", tmp_path / "run"
    cfg = tiny_micro_run_config(publish_interval=2, checkpoint_count=3, replay_warmup_positions=1)
    driver = make_micro_driver(shard_dir, ckpt_dir, run_dir, run_config=cfg)

    results = driver.run()

    assert driver.step == 6 == cfg.training.checkpoint_count * cfg.training.publish_interval
    assert len(results) == 6
    assert list_published_versions(ckpt_dir) == (0, 1, 2, 3)
    assert read_latest_pointer(ckpt_dir) == 3
    for version in (0, 1, 2, 3):
        loaded = load_checkpoint(ckpt_dir / f"ckpt-{version}.pt", MICRO)
        assert loaded.version == version
        assert loaded.learner_step == version * 2

    # No partial final interval: a further run() call is a pure no-op.
    assert driver.run() == []
    assert driver.step == 6


def test_immutability_a_redundant_publish_call_is_a_silent_no_op(tmp_path):
    shard_dir = tmp_path / "shards"
    write_real_micro_shards(shard_dir)
    ckpt_dir, run_dir = tmp_path / "ckpt", tmp_path / "run"
    cfg = tiny_micro_run_config(publish_interval=2, checkpoint_count=1, replay_warmup_positions=1)
    driver = make_micro_driver(shard_dir, ckpt_dir, run_dir, run_config=cfg)
    driver.run()

    published_path = ckpt_dir / "ckpt-1.pt"
    before = published_path.read_bytes()
    driver._maybe_publish()  # self.step is already the version-1 boundary
    assert published_path.read_bytes() == before
    markers = [
        r
        for r in iter_epoch_records(run_dir, "learner")
        if r.get("kind") == CHECKPOINT_PUBLISHED_KIND and r["model_version"] == 1
    ]
    assert len(markers) == 1  # not duplicated by the redundant call


def test_resume_never_republishes_an_existing_version(tmp_path):
    shard_dir = tmp_path / "shards"
    write_real_micro_shards(shard_dir)
    ckpt_dir, run_dir = tmp_path / "ckpt", tmp_path / "run"
    cfg = tiny_micro_run_config(publish_interval=2, checkpoint_count=2, replay_warmup_positions=1)
    driver = make_micro_driver(shard_dir, ckpt_dir, run_dir, run_config=cfg)
    driver.run()

    versions_before = list_published_versions(ckpt_dir)
    bytes_before = {v: (ckpt_dir / f"ckpt-{v}.pt").read_bytes() for v in versions_before}

    # A brand-new driver instance over the same directories: a "restart"
    # after the run had already fully completed.
    resumed = make_micro_driver(shard_dir, ckpt_dir, run_dir, run_config=cfg)
    assert resumed.step == 4
    assert resumed.run() == []  # nothing left to do

    assert list_published_versions(ckpt_dir) == versions_before
    for v in versions_before:
        assert (ckpt_dir / f"ckpt-{v}.pt").read_bytes() == bytes_before[v]
    # A new epoch was opened, but no version's marker count changed.
    assert resumed.epoch_writer.epoch == 1
    markers = [
        r for r in iter_epoch_records(run_dir, "learner") if r["kind"] == CHECKPOINT_PUBLISHED_KIND
    ]
    counts = Counter(r["model_version"] for r in markers)
    assert counts == {0: 1, 1: 1, 2: 1}


# ==============================================================================
# 6. The resume-equivalence golden (bit-for-bit), including the marker straddle
# ==============================================================================


def _weights_equal(net_a, net_c):
    for k, v in net_a.state_dict().items():
        assert torch.equal(v, net_c.state_dict()[k]), f"weights differ at {k}"


def test_resume_equivalence_bit_for_bit_kill_mid_interval(tmp_path):
    shard_dir = tmp_path / "shards"
    write_real_micro_shards(shard_dir, n_shards=8, positions_per_shard=7)
    cfg = tiny_micro_run_config(publish_interval=2, checkpoint_count=2, replay_warmup_positions=1)

    # --- branch A: fully uninterrupted -----------------------------------
    driver_a = make_micro_driver(
        shard_dir, tmp_path / "a" / "ckpt", tmp_path / "a" / "run", run_config=cfg
    )
    results_a = driver_a.run()

    # --- branch B: one step, then "kill" ----------------------------------
    driver_b = make_micro_driver(
        shard_dir, tmp_path / "b" / "ckpt", tmp_path / "b" / "run", run_config=cfg, max_steps=1
    )
    results_b = driver_b.run()
    assert driver_b.step == 1

    # --- resume into a fresh driver, run to completion --------------------
    driver_c = make_micro_driver(
        shard_dir, tmp_path / "b" / "ckpt", tmp_path / "b" / "run", run_config=cfg
    )
    assert driver_c.step == 1  # recovered from the rolling resume snapshot
    results_c = driver_c.run()

    results_resumed = results_b + results_c
    assert len(results_resumed) == len(results_a) == 4
    for step, (a, r) in enumerate(zip(results_a, results_resumed, strict=True)):
        assert torch.equal(a.total, r.total), f"step {step}: total loss differs"
        assert torch.equal(a.policy, r.policy), f"step {step}: policy loss differs"
        assert torch.equal(a.value, r.value), f"step {step}: value loss differs"
        if a.aux is None:
            assert r.aux is None
        else:
            assert torch.equal(a.aux, r.aux), f"step {step}: aux loss differs"

    _weights_equal(driver_a.net, driver_c.net)
    assert list_published_versions(tmp_path / "a" / "ckpt") == list_published_versions(
        tmp_path / "b" / "ckpt"
    )


def test_resume_equivalence_straddles_a_publish_and_its_snapshot(tmp_path):
    """The kill lands between a publish and the *next* resume snapshot.

    ``_advance_one_step`` runs everything ``_run_step`` does except the
    rolling resume-snapshot write, so calling it directly for the step that
    crosses a publish boundary -- and then simply never calling
    ``_write_resume_snapshot`` for that step -- reproduces exactly the
    scenario a crash landing in that gap would leave behind: the checkpoint
    and its marker are already durable, but ``resume.pt`` still names the
    *previous* step.
    """
    shard_dir = tmp_path / "shards"
    write_real_micro_shards(shard_dir, n_shards=8, positions_per_shard=7)
    cfg = tiny_micro_run_config(publish_interval=2, checkpoint_count=2, replay_warmup_positions=1)
    ckpt_dir, run_dir = tmp_path / "b" / "ckpt", tmp_path / "b" / "run"

    # --- branch A: fully uninterrupted (the reference trajectory) ---------
    driver_a = make_micro_driver(
        shard_dir, tmp_path / "a" / "ckpt", tmp_path / "a" / "run", run_config=cfg
    )
    results_a = driver_a.run()

    # --- branch B: step 0 normally, then straddle step 1's publish --------
    driver_b1 = make_micro_driver(shard_dir, ckpt_dir, run_dir, run_config=cfg)
    result_step0 = driver_b1._run_step()  # step: 0 -> 1, snapshot written (learner_step=1)
    assert driver_b1.step == 1
    result_step1 = driver_b1._advance_one_step()  # step: 1 -> 2, publishes v1, no snapshot
    assert driver_b1.step == 2
    assert list_published_versions(ckpt_dir) == (0, 1)  # v1 is durable...
    # ...but resume.pt still lags at learner_step=1 -- the straddle.
    from core.checkpoint import load_checkpoint as _load_ckpt

    lagging = _load_ckpt(ckpt_dir / "resume.pt", MICRO)
    assert lagging.learner_step == 1

    # --- "kill": stop using driver_b1; resume into a fresh driver ---------
    driver_c = make_micro_driver(shard_dir, ckpt_dir, run_dir, run_config=cfg)
    assert driver_c.epoch_writer.epoch == 1  # a genuinely new epoch
    assert driver_c.step == 2  # recovered from the *publish*, not the stale snapshot
    results_c = driver_c.run()

    results_resumed = [result_step0, result_step1] + results_c
    assert len(results_resumed) == len(results_a) == 4
    for step, (a, r) in enumerate(zip(results_a, results_resumed, strict=True)):
        assert torch.equal(a.total, r.total), f"step {step}: total loss differs"

    _weights_equal(driver_a.net, driver_c.net)
    assert list_published_versions(ckpt_dir) == (0, 1, 2)

    # The load-bearing assertion: the straddled version's marker was
    # neither dropped nor duplicated across the crash.
    markers = [
        r for r in iter_epoch_records(run_dir, "learner") if r["kind"] == CHECKPOINT_PUBLISHED_KIND
    ]
    counts = Counter(r["model_version"] for r in markers)
    assert counts == {0: 1, 1: 1, 2: 1}


# ==============================================================================
# 7. The observability high-water snapshot (issue #62): survives a resume
# ==============================================================================


def test_high_water_snapshot_survives_resume_and_continues_not_resets(tmp_path):
    """The checkpoint ``metrics`` dict is the learner's own high-water
    snapshot (issue #62, the opaque seam issue #56 built for this): a
    resumed driver restores it verbatim rather than starting empty, and
    ``reduce_run``'s cumulative curves only ever grow across the resume.
    """
    shard_dir = tmp_path / "shards"
    write_real_micro_shards(shard_dir, n_shards=8, positions_per_shard=7)
    cfg = tiny_micro_run_config(publish_interval=2, checkpoint_count=2, replay_warmup_positions=1)
    ckpt_dir, run_dir = tmp_path / "ckpt", tmp_path / "run"

    driver_b = make_micro_driver(shard_dir, ckpt_dir, run_dir, run_config=cfg, max_steps=1)
    driver_b.run()
    assert driver_b.step == 1

    snapshot = load_checkpoint(resume_path(ckpt_dir), MICRO)
    assert snapshot.metrics["learner_step"] == 1.0
    assert "loss_total" in snapshot.metrics

    reduced_before = reduce_run(run_dir)
    assert reduced_before.totals["learner_step"] == 1.0
    assert reduced_before.gauges["loss_total"] == snapshot.metrics["loss_total"]

    # --- "kill"; resume into a fresh driver instance ------------------------
    driver_c = make_micro_driver(shard_dir, ckpt_dir, run_dir, run_config=cfg)
    assert driver_c.step == 1  # recovered from the rolling resume snapshot
    # Restored *before* any new step trains: the exact prior snapshot, not
    # an empty reset.
    assert driver_c._high_water == snapshot.metrics

    driver_c.run()  # trains the remaining steps to total_steps == 4
    assert driver_c.step == 4

    reduced_after = reduce_run(run_dir)
    assert reduced_after.totals["learner_step"] == 4.0
    # The reducer's cumulative learner_step total only ever grows across the
    # resume -- never resets to 0, never reverts to the pre-resume value.
    assert reduced_after.totals["learner_step"] > reduced_before.totals["learner_step"]
    # The gauges reflect the *latest* step in run time order, post-resume.
    assert reduced_after.gauges["loss_total"] == driver_c._high_water["loss_total"]
