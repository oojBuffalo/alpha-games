"""The checkpoint bundle: ``core/checkpoint.py`` (§12 M3, issue #56).

Round-trips every bundle field exactly, exercises the two write namespaces'
policy (immutable publish, atomic ``latest`` pointer, rolling resume
snapshot), the four-plus-one resume-selection cases, validate-on-load's
fingerprint gate, and torn-write safety. The load-bearing test at the bottom
is a bit-for-bit resume-equivalence golden: a tiny real micro-Blokus pipeline
(real ``core.replay_window.ReplayWindow`` over real on-disk shards, real
``core.augment.augment_sample`` D9 streams, real ``core.train.train_step``)
run four steps uninterrupted vs. two steps / checkpoint / reload-into-fresh-
objects / two more steps, compared with ``torch.equal`` -- no tolerances.
"""

from __future__ import annotations

import dataclasses
import json
import random
import warnings

import pytest
import torch

from core.artifact_fingerprint import FingerprintMismatchError, build_fingerprint
from core.augment import augment_sample
from core.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointFormatError,
    build_bundle,
    latest_pointer_path,
    list_published_versions,
    load_checkpoint,
    newest_published_version,
    published_checkpoint_path,
    read_latest_pointer,
    resume_path,
    select_resume_bundle,
    write_latest_pointer,
    write_published_checkpoint,
    write_resume_snapshot,
)
from core.network import Network, NetworkConfig
from core.replay_shard import SampleRecord, write_shard
from core.replay_window import ReplayWindow
from core.runconfig import MICRO_RUN_CONFIG_PATH, load_run_config
from core.seeding import LearnerRNGs, net_init_seed
from core.train import collate, make_lr_scheduler, make_optimizer, make_scaler, train_step
from games.blokus_duo import BlokusDuo
from games.blokus_duo.config import MICRO_CONFIG
from games.othello import Othello
from games.tictactoe import TicTacToe

TTT = TicTacToe()
OTHELLO = Othello()
TTT_NET_CONFIG = NetworkConfig(2, (3, 3), (9,), trunk_blocks=1, trunk_channels=4)


def _tiny_ttt_net(seed):
    """A tiny TTT-shaped net, optimizer, and (disabled, CPU) scaler, seeded."""
    torch.manual_seed(seed)
    net = Network(TTT_NET_CONFIG)
    optimizer = make_optimizer(net, lr=1e-2)
    scaler = make_scaler("cpu")
    return net, optimizer, scaler


def _one_ttt_sample():
    """One collate-shaped TTT sample (no aux): a real encoded state + a
    trivial legal sparse pi."""
    state = TTT.initial_state()
    legal = TTT.legal_moves(state)
    return (TTT.encode_state(state), [(a, 1) for a in legal], 0.0)


def _train_ttt_step(net, optimizer, scaler):
    """One real TTT train step, on a single real sample -- populates real
    (nonzero) SGD momentum buffers so bundle round-trip tests exercise
    genuine optimizer tensor state, not a freshly-constructed empty one."""
    batch = collate(TTT, [_one_ttt_sample(), _one_ttt_sample()])
    return train_step(net, optimizer, scaler, batch)


def _assert_equal_recursive(a, b, path="root"):
    """Deep-equality assertion over a state-dict-shaped structure.

    Tensors compare with ``torch.equal`` (exact, no tolerance); dict/list/
    tuple containers recurse; everything else compares with ``==``.
    """
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), f"{path}: expected a tensor, got {type(b)}"
        assert torch.equal(a, b), f"{path}: tensors differ"
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: key sets differ"
        for k in a:
            _assert_equal_recursive(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), f"{path}: shape differs"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_equal_recursive(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


# --- bundle round trip -------------------------------------------------------


def test_bundle_round_trip_every_field_exact(tmp_path):
    net, optimizer, scaler = _tiny_ttt_net(seed=1)
    _train_ttt_step(net, optimizer, scaler)  # real, nonzero momentum buffers

    run_config = {"name": "ttt-golden", "training": {"learning_rate": 1e-2}}
    metrics = {"best_total_loss": 0.42, "checkpoints_written": 3, "note": None}
    bundle = build_bundle(
        version=0,
        learner_step=1,
        game=TTT,
        run_config=run_config,
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics=metrics,
    )
    path = write_published_checkpoint(tmp_path, bundle)
    loaded = load_checkpoint(path, TTT)

    assert loaded.schema_version == CHECKPOINT_SCHEMA_VERSION == bundle.schema_version
    assert loaded.version == 0
    assert loaded.learner_step == 1
    assert loaded.run_config == run_config
    assert loaded.fingerprint == build_fingerprint(TTT)
    assert loaded.metrics == metrics
    _assert_equal_recursive(dict(net.state_dict()), loaded.model_state_dict)
    _assert_equal_recursive(optimizer.state_dict(), loaded.optimizer_state_dict)
    _assert_equal_recursive(scaler.state_dict(), loaded.scaler_state_dict)
    # Disabled on CPU (core.train.make_scaler): the scaler's own state_dict is
    # the empty dict, and it round-trips as exactly that -- never None.
    assert loaded.scaler_state_dict == {}


def test_bundle_round_trip_via_resume_snapshot(tmp_path):
    # The resume namespace shares the exact same serializer -- the
    # module-docstring's "one serializer/deserializer pair" claim, checked.
    net, optimizer, scaler = _tiny_ttt_net(seed=2)
    bundle = build_bundle(
        version=0,
        learner_step=7,
        game=TTT,
        run_config={"a": 1},
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics={},
    )
    path = write_resume_snapshot(tmp_path, bundle)
    assert path == resume_path(tmp_path)
    loaded = load_checkpoint(path, TTT)
    assert loaded.learner_step == 7
    _assert_equal_recursive(dict(net.state_dict()), loaded.model_state_dict)


def test_load_checkpoint_rejects_unsupported_schema_version(tmp_path):
    net, optimizer, scaler = _tiny_ttt_net(seed=3)
    bundle = build_bundle(
        version=0,
        learner_step=0,
        game=TTT,
        run_config={},
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics={},
    )
    path = write_published_checkpoint(tmp_path, bundle)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["schema_version"] = 999
    torch.save(payload, path)

    with pytest.raises(CheckpointFormatError, match="schema_version"):
        load_checkpoint(path, TTT)


# --- immutability: published namespace ---------------------------------------


def test_write_published_checkpoint_immutable_existing_version_raises(tmp_path):
    net, optimizer, scaler = _tiny_ttt_net(seed=4)
    first = build_bundle(
        version=0,
        learner_step=1,
        game=TTT,
        run_config={"tag": "first"},
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics={},
    )
    path = write_published_checkpoint(tmp_path, first)
    original_bytes = path.read_bytes()

    _train_ttt_step(net, optimizer, scaler)
    second = build_bundle(
        version=0,
        learner_step=2,
        game=TTT,
        run_config={"tag": "second"},
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics={},
    )
    with pytest.raises(FileExistsError, match="version 0"):
        write_published_checkpoint(tmp_path, second)

    # The failed attempt touched nothing: byte-identical to before, and the
    # still-readable content is the *first* bundle, never the second's.
    assert path.read_bytes() == original_bytes
    reloaded = load_checkpoint(path, TTT)
    assert reloaded.learner_step == 1
    assert reloaded.run_config == {"tag": "first"}


def test_write_published_checkpoint_different_versions_coexist(tmp_path):
    net, optimizer, scaler = _tiny_ttt_net(seed=5)
    for v in (0, 1, 2):
        bundle = build_bundle(
            version=v,
            learner_step=v * 10,
            game=TTT,
            run_config={},
            net=net,
            optimizer=optimizer,
            scaler=scaler,
            metrics={},
        )
        write_published_checkpoint(tmp_path, bundle)
    assert list_published_versions(tmp_path) == (0, 1, 2)
    assert newest_published_version(tmp_path) == 2
    for v in (0, 1, 2):
        assert load_checkpoint(published_checkpoint_path(tmp_path, v), TTT).learner_step == v * 10


def test_list_published_versions_empty_directory(tmp_path):
    assert list_published_versions(tmp_path) == ()
    assert list_published_versions(tmp_path / "does-not-exist-yet") == ()
    assert newest_published_version(tmp_path) is None


# --- latest pointer ------------------------------------------------------------


def test_latest_pointer_atomic_and_names_existing_version(tmp_path):
    net, optimizer, scaler = _tiny_ttt_net(seed=6)
    for v in (0, 1):
        bundle = build_bundle(
            version=v,
            learner_step=v,
            game=TTT,
            run_config={},
            net=net,
            optimizer=optimizer,
            scaler=scaler,
            metrics={},
        )
        write_published_checkpoint(tmp_path, bundle)

    assert read_latest_pointer(tmp_path) is None  # never written yet
    write_latest_pointer(tmp_path, 0)
    assert read_latest_pointer(tmp_path) == 0
    write_latest_pointer(tmp_path, 1)
    assert read_latest_pointer(tmp_path) == 1

    payload = json.loads(latest_pointer_path(tmp_path).read_text())
    assert payload == {"version": 1}


def test_write_latest_pointer_rejects_nonexistent_version(tmp_path):
    net, optimizer, scaler = _tiny_ttt_net(seed=7)
    bundle = build_bundle(
        version=0,
        learner_step=0,
        game=TTT,
        run_config={},
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics={},
    )
    write_published_checkpoint(tmp_path, bundle)
    with pytest.raises(FileNotFoundError, match="5"):
        write_latest_pointer(tmp_path, 5)
    assert read_latest_pointer(tmp_path) is None


# --- resume selection ----------------------------------------------------------


def _publish_and_snapshot(tmp_path, net, optimizer, scaler, *, publish_step, snapshot_step=None):
    """Publish version 0 at ``publish_step``, optionally also write a resume
    snapshot at ``snapshot_step``."""
    published = build_bundle(
        version=0,
        learner_step=publish_step,
        game=TTT,
        run_config={},
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics={"kind": "publish"},
    )
    write_published_checkpoint(tmp_path, published)
    if snapshot_step is not None:
        snapshot = build_bundle(
            version=0,
            learner_step=snapshot_step,
            game=TTT,
            run_config={},
            net=net,
            optimizer=optimizer,
            scaler=scaler,
            metrics={"kind": "snapshot"},
        )
        write_resume_snapshot(tmp_path, snapshot)


def test_resume_selection_snapshot_newer_than_publish_wins(tmp_path):
    net, optimizer, scaler = _tiny_ttt_net(seed=8)
    _publish_and_snapshot(tmp_path, net, optimizer, scaler, publish_step=5, snapshot_step=9)
    selected = select_resume_bundle(tmp_path, TTT)
    assert selected.learner_step == 9
    assert selected.metrics == {"kind": "snapshot"}


def test_resume_selection_older_snapshot_falls_back_to_publish(tmp_path):
    net, optimizer, scaler = _tiny_ttt_net(seed=9)
    _publish_and_snapshot(tmp_path, net, optimizer, scaler, publish_step=9, snapshot_step=5)
    selected = select_resume_bundle(tmp_path, TTT)
    assert selected.learner_step == 9
    assert selected.metrics == {"kind": "publish"}


def test_resume_selection_equal_step_falls_back_to_publish(tmp_path):
    net, optimizer, scaler = _tiny_ttt_net(seed=10)
    _publish_and_snapshot(tmp_path, net, optimizer, scaler, publish_step=6, snapshot_step=6)
    selected = select_resume_bundle(tmp_path, TTT)
    assert selected.learner_step == 6
    assert selected.metrics == {"kind": "publish"}  # tie goes to the publish, never the snapshot


def test_resume_selection_no_snapshot_uses_newest_publish(tmp_path):
    net, optimizer, scaler = _tiny_ttt_net(seed=11)
    _publish_and_snapshot(tmp_path, net, optimizer, scaler, publish_step=3)
    selected = select_resume_bundle(tmp_path, TTT)
    assert selected.learner_step == 3
    assert selected.metrics == {"kind": "publish"}


def test_resume_selection_snapshot_only_no_publish_yet(tmp_path):
    net, optimizer, scaler = _tiny_ttt_net(seed=12)
    snapshot = build_bundle(
        version=0,
        learner_step=4,
        game=TTT,
        run_config={},
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics={"kind": "snapshot"},
    )
    write_resume_snapshot(tmp_path, snapshot)
    selected = select_resume_bundle(tmp_path, TTT)
    assert selected.learner_step == 4
    assert selected.metrics == {"kind": "snapshot"}


def test_resume_selection_neither_exists_signals_fresh_start(tmp_path):
    assert select_resume_bundle(tmp_path, TTT) is None
    assert select_resume_bundle(tmp_path / "brand-new-run-dir", TTT) is None


# --- fingerprint validate-on-load ---------------------------------------------


def test_load_checkpoint_fingerprint_mismatch_names_fields_and_applies_nothing(tmp_path):
    net, optimizer, scaler = _tiny_ttt_net(seed=13)
    bundle = build_bundle(
        version=0,
        learner_step=0,
        game=TTT,
        run_config={},
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics={},
    )
    path = write_published_checkpoint(tmp_path, bundle)

    fresh_net, _, _ = _tiny_ttt_net(seed=99)
    weights_before = {k: v.clone() for k, v in fresh_net.state_dict().items()}

    with pytest.raises(FingerprintMismatchError, match="game_identity") as excinfo:
        loaded = load_checkpoint(path, OTHELLO)
        fresh_net.load_state_dict(loaded.model_state_dict)  # must never execute
    assert "policy_shape" in str(excinfo.value) or "input_planes" in str(excinfo.value)

    for k, v in fresh_net.state_dict().items():
        assert torch.equal(v, weights_before[k])  # nothing partially applied


def test_select_resume_bundle_fingerprint_mismatch_on_winner(tmp_path):
    net, optimizer, scaler = _tiny_ttt_net(seed=14)
    bundle = build_bundle(
        version=0,
        learner_step=2,
        game=TTT,
        run_config={},
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics={},
    )
    write_published_checkpoint(tmp_path, bundle)
    with pytest.raises(FingerprintMismatchError):
        select_resume_bundle(tmp_path, OTHELLO)


# --- torn-file safety ----------------------------------------------------------


def test_published_write_ignores_a_stray_temp_file_from_a_dead_process(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    stray = tmp_path / "ckpt-3.pt.tmp-deadbeefdeadbeefdeadbeefdeadbeef"
    stray.write_bytes(b"not-a-real-checkpoint")

    assert list_published_versions(tmp_path) == ()  # never matched by the published glob
    with pytest.raises(FileNotFoundError):
        load_checkpoint(published_checkpoint_path(tmp_path, 3), TTT)

    net, optimizer, scaler = _tiny_ttt_net(seed=15)
    bundle = build_bundle(
        version=3,
        learner_step=1,
        game=TTT,
        run_config={},
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics={},
    )
    path = write_published_checkpoint(tmp_path, bundle)  # succeeds despite the stray file

    assert path.exists()
    assert load_checkpoint(path, TTT).learner_step == 1
    assert stray.exists()  # ignored, not cleaned -- an orphan from a different write attempt
    assert list_published_versions(tmp_path) == (3,)


def test_resume_snapshot_torn_write_leaves_the_previous_snapshot_intact(tmp_path):
    net, optimizer, scaler = _tiny_ttt_net(seed=16)
    first = build_bundle(
        version=0,
        learner_step=2,
        game=TTT,
        run_config={},
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics={},
    )
    write_resume_snapshot(tmp_path, first)
    original_bytes = resume_path(tmp_path).read_bytes()

    # Simulate a crash mid-write of a *second* snapshot: a temp file lands
    # under resume.pt's naming convention, but the rename to the final name
    # (resume.pt) never happens -- exactly what a killed process leaves
    # behind under core.replay_shard._atomic_write's scheme.
    crashed_temp = tmp_path / "resume.pt.tmp-cafebabecafebabecafebabecafebabe"
    crashed_temp.write_bytes(b"torn-write-garbage")

    # A reader targeting the real name never sees the torn write.
    assert resume_path(tmp_path).read_bytes() == original_bytes
    reloaded = load_checkpoint(resume_path(tmp_path), TTT)
    assert reloaded.learner_step == 2

    # A subsequent real write succeeds normally, under its own fresh temp
    # name, ignoring the crashed leftover entirely.
    _train_ttt_step(net, optimizer, scaler)
    second = build_bundle(
        version=0,
        learner_step=7,
        game=TTT,
        run_config={},
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics={},
    )
    write_resume_snapshot(tmp_path, second)
    assert load_checkpoint(resume_path(tmp_path), TTT).learner_step == 7
    assert crashed_temp.exists()  # the stray leftover is never cleaned up automatically
    assert crashed_temp.read_bytes() == b"torn-write-garbage"


# ==============================================================================
# THE golden: bit-for-bit resume equivalence over a real tiny pipeline
# ==============================================================================

_MICRO_GAME = BlokusDuo(config=MICRO_CONFIG)
_MICRO_NUM_AUX = len(_MICRO_GAME.value_targets.aux_names)
_MICRO_GROUP_SIZE = len(_MICRO_GAME.symmetry_group)


def _micro_net_config():
    base = NetworkConfig.from_game(_MICRO_GAME)
    # Tiny trunk: correctness/determinism under test, not D5 throughput.
    return dataclasses.replace(base, trunk_blocks=1, trunk_channels=4)


def _real_micro_records(run_id, actor_id, game_index, n, seed):
    """``n`` structurally real samples from an actual micro-Blokus rollout.

    States come from genuinely applying legal moves (the lowest legal action
    id each ply -- move *quality* is irrelevant here, only that ``planes``
    are real ``encode_state`` output over reachable states); sparse pi/z/aux
    are seeded synthetic targets, exactly the pattern
    ``tests/test_replay_window.py`` and ``tests/test_train_step.py`` already
    use for shard/collate fixtures.
    """
    rng = random.Random(seed)
    state = _MICRO_GAME.initial_state()
    records = []
    for ply in range(n):
        if _MICRO_GAME.is_terminal(state):
            break
        legal = list(_MICRO_GAME.legal_moves(state))
        ids = rng.sample(legal, min(len(legal), rng.randint(1, min(4, len(legal)))))
        counts = [rng.randint(1, 5) for _ in ids]
        records.append(
            SampleRecord(
                planes=_MICRO_GAME.encode_state(state),
                sparse_pi=tuple(zip(ids, counts, strict=True)),
                z=rng.choice([-1.0, 0.0, 1.0]),
                aux=(rng.uniform(-1.0, 1.0),) if _MICRO_NUM_AUX else (),
                mover=_MICRO_GAME.current_player(state),
                model_version=0,
                ply=ply,
                game_id=(run_id, actor_id, game_index),
            )
        )
        state = _MICRO_GAME.apply(state, min(legal))
    return tuple(records)


def _write_real_micro_shards(shard_dir):
    """Populate ``shard_dir`` with three real, on-disk, fingerprint-valid shards."""
    for i, (actor, n) in enumerate((("a", 7), ("b", 6), ("c", 5))):
        records = _real_micro_records("golden-run", actor, 0, n, seed=1000 + i)
        write_shard(
            shard_dir / f"shard-golden-run-{actor}-0.npz",
            _MICRO_GAME,
            records,
            run_id="golden-run",
            actor_id=actor,
            seq=0,
        )


def _training_row(sample, num_aux):
    """The ``core.train.collate`` row shape for one ``SampleRecord`` (§12 M2)."""
    if num_aux == 0:
        return (sample.planes, sample.sparse_pi, sample.z)
    return (sample.planes, sample.sparse_pi, sample.z, sample.aux)


def _run_real_learner_step(net, optimizer, scaler, window, run_seed, step, batch_size):
    """One real learner step: real sampler, real D9 augmentation, real train_step.

    Mirrors ``scripts/run_micro.py::_learner_step``'s composition exactly,
    but drawing from the real on-disk-shard ``core.replay_window.ReplayWindow``
    (``sample_batch(run_seed, step, batch_size)``) rather than the in-memory
    M2.5 window -- the M3 replay stack this checkpoint module's resume golden
    is meant to exercise for real.
    """
    rngs = LearnerRNGs.for_step(run_seed, step)
    batch = window.sample_batch(run_seed, step, batch_size)
    rows = []
    for sample in batch:
        if _MICRO_GROUP_SIZE:
            g_index = rngs.augmentation.randrange(_MICRO_GROUP_SIZE)
            planes, sparse_pi = augment_sample(
                _MICRO_GAME, sample.planes, sample.sparse_pi, g_index
            )
            sample = dataclasses.replace(sample, planes=planes, sparse_pi=tuple(sparse_pi))
        rows.append(_training_row(sample, _MICRO_NUM_AUX))
    collated = collate(_MICRO_GAME, rows)
    return train_step(net, optimizer, scaler, collated)


def _fresh_micro_training_objects(run_seed, learning_rate):
    """Build a freshly-seeded net/optimizer/scaler for the golden's micro net."""
    torch.manual_seed(net_init_seed(run_seed))
    net = Network(_micro_net_config())
    optimizer = make_optimizer(net, lr=learning_rate)
    scaler = make_scaler("cpu")
    return net, optimizer, scaler


def test_resume_equivalence_bit_for_bit_over_four_real_learner_steps(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write_real_micro_shards(shard_dir)
    ckpt_dir = tmp_path / "checkpoints"

    run_seed = 20260818
    real_cfg = load_run_config(MICRO_RUN_CONFIG_PATH)
    # Small, distinct-per-step warmup/cosine window so the LR genuinely
    # varies across all four steps -- a constant LR would leave the
    # schedule's resume-from-step-alone claim untested.
    training = dataclasses.replace(
        real_cfg.training, learning_rate=1e-3, warmup_steps=1, cosine_total_steps=6
    )
    cfg = dataclasses.replace(real_cfg, training=training)
    batch_size = 4

    # --- branch A: fully uninterrupted, four real learner steps ---------------
    net_a, opt_a, scaler_a = _fresh_micro_training_objects(run_seed, cfg.training.learning_rate)
    sched_a = make_lr_scheduler(opt_a, cfg.training.warmup_steps, cfg.training.cosine_total_steps)
    window_a = ReplayWindow(shard_dir, _MICRO_GAME, capacity=1000)
    window_a.rescan()
    losses_a = []
    for step in range(4):
        parts = _run_real_learner_step(net_a, opt_a, scaler_a, window_a, run_seed, step, batch_size)
        sched_a.step()
        losses_a.append(parts)

    # --- branch B: two steps, then checkpoint --------------------------------
    net_b, opt_b, scaler_b = _fresh_micro_training_objects(run_seed, cfg.training.learning_rate)
    sched_b = make_lr_scheduler(opt_b, cfg.training.warmup_steps, cfg.training.cosine_total_steps)
    window_b = ReplayWindow(shard_dir, _MICRO_GAME, capacity=1000)
    window_b.rescan()
    losses_b = []
    for step in range(2):
        parts = _run_real_learner_step(net_b, opt_b, scaler_b, window_b, run_seed, step, batch_size)
        sched_b.step()
        losses_b.append(parts)

    bundle = build_bundle(
        version=1,
        learner_step=2,
        game=_MICRO_GAME,
        run_config=cfg.to_dict(),
        net=net_b,
        optimizer=opt_b,
        scaler=scaler_b,
        metrics={"best_total_loss": min(float(p.total) for p in losses_b)},
    )
    write_resume_snapshot(ckpt_dir, bundle)
    write_published_checkpoint(ckpt_dir, bundle)
    write_latest_pointer(ckpt_dir, 1)

    # --- reload into entirely fresh objects, then two more steps -------------
    resumed = select_resume_bundle(ckpt_dir, _MICRO_GAME)
    assert resumed.learner_step == 2

    net_c = Network(_micro_net_config())  # architecture only; overwritten below
    opt_c = make_optimizer(net_c, lr=cfg.training.learning_rate)
    sched_c = make_lr_scheduler(opt_c, cfg.training.warmup_steps, cfg.training.cosine_total_steps)
    # Fast-forwarding a schedule state does not call optimizer.step() in
    # between -- torch warns about that ordering (its warning assumes the
    # scheduler and optimizer are always advanced together in a live
    # training loop), but the LR values produced this way are bit-for-bit
    # identical to a continuously-stepped scheduler at the same step count
    # (the resume rule this golden test exists to prove); the warning is
    # expected and benign here, not a bug.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(resumed.learner_step):
            sched_c.step()  # the schedule's entire resumable state is the step count
    scaler_c = make_scaler("cpu")

    net_c.load_state_dict(resumed.model_state_dict)
    opt_c.load_state_dict(resumed.optimizer_state_dict)
    scaler_c.load_state_dict(resumed.scaler_state_dict)

    window_c = ReplayWindow(shard_dir, _MICRO_GAME, capacity=1000)  # fresh instance, re-scans disk
    window_c.rescan()
    losses_c = []
    for step in range(2, 4):
        parts = _run_real_learner_step(net_c, opt_c, scaler_c, window_c, run_seed, step, batch_size)
        sched_c.step()
        losses_c.append(parts)

    # --- compare: uninterrupted vs. checkpoint-reload, all four steps --------
    resumed_losses = losses_b + losses_c
    assert len(resumed_losses) == len(losses_a) == 4
    for step, (a, r) in enumerate(zip(losses_a, resumed_losses, strict=True)):
        assert torch.equal(a.total, r.total), f"step {step}: total loss differs"
        assert torch.equal(a.policy, r.policy), f"step {step}: policy loss differs"
        assert torch.equal(a.value, r.value), f"step {step}: value loss differs"
        if a.aux is None:
            assert r.aux is None
        else:
            assert torch.equal(a.aux, r.aux), f"step {step}: aux loss differs"

    for k, v in net_a.state_dict().items():
        assert torch.equal(v, net_c.state_dict()[k]), f"final weights differ at {k}"
