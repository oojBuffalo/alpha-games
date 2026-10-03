"""M4 protocol constants registry + fingerprint (design doc §9 amendment, tasks/m4/001).

Pure stdlib, no torch, no ``games.*`` import: this module is the single place every
covered M4 evaluation convention is written down as data, so a change to any of them
changes one hash rather than depending on someone remembering to bump a version by
hand. :func:`protocol_fingerprint` is that hash -- sha256 over :data:`REGISTRY`'s
canonical JSON -- and it is what ``core.eval_store`` stamps into every cell header and
what a resuming writer asserts against before it is allowed to append another line.

``PROTOCOL_VERSION`` (this amendment's own version number, tasks/m4/001 pin 10) and
``protocol_fingerprint()`` are two different tools for two different jobs: the version
is a human-legible label bumped deliberately alongside a doc amendment; the fingerprint
is the mechanical, self-enforcing guard that catches *any* covered-constant drift,
deliberate or not, version bump or none -- code that adds a constant to ``REGISTRY``
without bumping ``PROTOCOL_VERSION`` still changes the fingerprint, and a resuming
writer still refuses to append under it.

Tasks 7 and 8 register their own convention constants (the bootstrap/Mann-Kendall
conventions, the plateau-rule constants) into this same :data:`REGISTRY` later --
additive only. Nothing here is ever edited in place to change a *value*; a genuine
value change is a new ``PROTOCOL_VERSION`` and, per the design doc, a new eval
namespace (the relaunch guard in a later task refuses to mix evidence across one).

The committed design doc §9 and §12 are checked against this registry by tests.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: This amendment's version (tasks/m4/001 pin 10). A human-legible label, bumped
#: deliberately alongside a doc amendment that changes a covered convention's value --
#: distinct from :func:`protocol_fingerprint`, which changes automatically on *any*
#: registry drift regardless of whether this constant was remembered to move too.
PROTOCOL_VERSION = 2

#: Cell-header / pair-record on-disk shape version (independent axis from
#: ``PROTOCOL_VERSION``: the record *shape* a reader must recognize can move
#: separately from the *values* the protocol pins). ``core.eval_store`` rejects any
#: schema_version it does not equal, loudly.
SCHEMA_VERSION = 1

# --- seed-derivation labels (core.seeding.derive_seed literal label parts) --------

#: ``core.runner.play_pairs``'s per-pair label: ``derive_seed(seed, SEED_LABEL_PAIR,
#: pair_index)``.
SEED_LABEL_PAIR = "pair"

#: ``core.runner.play_pairs``'s per-seat labels: ``derive_seed(pair_seed,
#: SEED_LABEL_SEAT_A)`` / ``derive_seed(pair_seed, SEED_LABEL_SEAT_B)``.
SEED_LABEL_SEAT_A = "a"
SEED_LABEL_SEAT_B = "b"

#: This store's own per-cell seeding label (a later task registers
#: ``core.seeding.PURPOSE_EVAL`` under this exact string: ``derive_seed(eval_seed,
#: PURPOSE_EVAL, cell_id)``). Recorded here so the two sides are pinned against one
#: source rather than a literal someone has to keep in sync by memory.
SEED_LABEL_EVAL = "eval"
SEED_LABEL_BOOTSTRAP = "bootstrap"
SEED_LABEL_REPLICATE = "replicate"

# --- pinned eval constants (tasks/m4/001) -----------------------------------------

#: Mirrored pairs per (candidate, rung, opponent) cell -- the §1 bootstrap's
#: resampling unit (tasks/m4/001 pin 1).
PAIRS_PER_CELL = 24

#: Rung 6/7 eval search-form simulation budget S (tasks/m4/001 pin 4), matching D6's
#: 512-sim full tier. Must equal ``core.eval_agents.EVAL_SIMS`` -- verified by a
#: cross-module golden in ``tests/test_eval_store.py`` rather than imported directly:
#: ``core.eval_agents`` pulls in torch (via ``core.network``/``core.checkpoint``), and
#: this module is pure-stdlib by design (mirrors ``core.seeding``'s confinement).
EVAL_SIMS = 512

# Operational liveness thresholds are reported separately from the scoring registry.
LIVENESS_MAX_LAG = 4
LIVENESS_BREACH_CONSECUTIVE = 2
LIVENESS_MAX_THROUGHPUT_DEGRADATION = 0.05

#: Rung-8 historical-opponent selection rule (tasks/m4/001 pin 5): a candidate's
#: opponents are ``{v - 1, v - ceil(K / RUNG8_LAG_DIVISOR), RUNG8_EARLIEST_VERSION}``,
#: intersected with the available member versions -- see
#: ``core.eval_agents.historical_opponents``, the code-side implementation this
#: registers the shape of.
RUNG8_LAG_DIVISOR = 4
RUNG8_EARLIEST_VERSION = 1

# --- bootstrap / Mann-Kendall statistical conventions (tasks/m4/001 pin 7, tasks/m4/007) -----

#: The pinned production bootstrap replicate count (tasks/m4/001 pin 7): ``B = 1,999``,
#: satisfying §1's "B ≈ 2,000" while keeping both order-statistic ranks below integral
#: (``(B+1)*0.025 = 50``, ``(B+1)*0.975 = 1,950``). ``core.eval_stats``'s CI/gate
#: functions take ``B`` as a parameter defaulting to this value; an authoritative
#: verdict (task 7.3) requires ``B == BOOTSTRAP_B_PRODUCTION`` exactly.
BOOTSTRAP_B_PRODUCTION = 1999

#: The admissible-``B`` rank rule (tasks/m4/001 pin 7): both order-statistic ranks
#: ``(B+1)*BOOTSTRAP_CI_LOWER_QUANTILE`` / ``(B+1)*BOOTSTRAP_CI_UPPER_QUANTILE`` are
#: integral exactly when ``(B + 1)`` is a multiple of this modulus -- equivalently
#: ``B % BOOTSTRAP_B_ADMISSIBLE_MODULUS == BOOTSTRAP_B_ADMISSIBLE_REMAINDER``
#: (``B ≡ 39 mod 40``: 39, 79, ..., 1,999). A ``B`` failing this check is rejected
#: loudly by ``core.eval_stats.order_statistic_ci`` rather than silently rounded.
BOOTSTRAP_B_ADMISSIBLE_MODULUS = 40
BOOTSTRAP_B_ADMISSIBLE_REMAINDER = 39

#: The single order-statistic CI rule's two quantiles (tasks/m4/001 pin 7): the 95%
#: interval's endpoints sit at ranks ``(B+1)*BOOTSTRAP_CI_LOWER_QUANTILE`` and
#: ``(B+1)*BOOTSTRAP_CI_UPPER_QUANTILE`` (1-indexed order statistics of the sorted
#: replicate values) -- the one convention used at every admissible ``B``, never a
#: second quantile rule.
BOOTSTRAP_CI_LOWER_QUANTILE = 0.025
BOOTSTRAP_CI_UPPER_QUANTILE = 0.975

# --- profiled-plateau rule constants (design doc §12 M4) -------------------------
# Changing any pin requires a protocol version and eval namespace change.
PLATEAU_WINDOW_M = 16
PLATEAU_MK_ALPHA = 0.05
PLATEAU_HALF_WINDOW_RULE = "ceil(M/2)"
#: Precision gate: CI width must be strictly below this threshold.
PLATEAU_CI_WIDTH_THRESHOLD_ELO = 150.0
#: Location gate: both endpoints must lie strictly inside (-margin, +margin).
PLATEAU_EQUIVALENCE_MARGIN_ELO = 75.0
PLATEAU_GPU_HOURS_MIN = 8.0
#: Persistence across overlapping snapshots, not independent statistical evidence.
PLATEAU_CONFIRMATION_COUNT = 2

# Statistical and evidence conventions covered by section 9, pins 7-10.
DELTA_WINDOW_DIVISOR = 3
DELTA_GATE_THRESHOLD = 0.0
MK_MIN_OBSERVATIONS = 3
VIRTUAL_DRAW_SCORE = 0.5
VIRTUAL_DRAW_GAMES = 1
STATISTICAL_CONVENTIONS = {
    "bootstrap_resampling": "within-cell-paired-records-with-replacement",
    "bootstrap_fit": "joint-refit-each-replicate-warm-started",
    "bootstrap_iteration_order": "sorted-cell-id-then-stored-record-order",
    "delta_window_rounding": "ceiling",
    "delta_gate_comparison": "lower-ci-strictly-greater-than-threshold",
    "mann_kendall_variance": "tie-corrected",
    "mann_kendall_continuity": "subtract-sign-s",
    "mann_kendall_p": "two-sided-normal",
    "mann_kendall_insufficient": "s-z-p-null",
    "mann_kendall_zero_variance": "s=0,z=0,p=1",
    "snapshot_scope": "complete-contiguous-member-prefix-only",
    "delta_snapshot_gate": "prefix-equals-k-target",
    "authoritative_gate": "complete-k-set-and-production-b",
    "finite_fit": "one-virtual-draw-per-unordered-matchup",
}

#: Every covered constant, by name -- the input to :func:`protocol_fingerprint`.
#: Additive only (see the module docstring): a later task adds keys here, never
#: repurposes one to mean something else.
REGISTRY: dict[str, Any] = {
    "protocol_version": PROTOCOL_VERSION,
    "schema_version": SCHEMA_VERSION,
    "seed_label_pair": SEED_LABEL_PAIR,
    "seed_label_seat_a": SEED_LABEL_SEAT_A,
    "seed_label_seat_b": SEED_LABEL_SEAT_B,
    "seed_label_eval": SEED_LABEL_EVAL,
    "seed_label_bootstrap": SEED_LABEL_BOOTSTRAP,
    "seed_label_replicate": SEED_LABEL_REPLICATE,
    "delta_window_divisor": DELTA_WINDOW_DIVISOR,
    "delta_gate_threshold": DELTA_GATE_THRESHOLD,
    "mk_min_observations": MK_MIN_OBSERVATIONS,
    "virtual_draw_score": VIRTUAL_DRAW_SCORE,
    "virtual_draw_games": VIRTUAL_DRAW_GAMES,
    **STATISTICAL_CONVENTIONS,
    "pairs_per_cell": PAIRS_PER_CELL,
    "eval_sims": EVAL_SIMS,
    "rung8_lag_divisor": RUNG8_LAG_DIVISOR,
    "rung8_earliest_version": RUNG8_EARLIEST_VERSION,
    "bootstrap_b_production": BOOTSTRAP_B_PRODUCTION,
    "bootstrap_b_admissible_modulus": BOOTSTRAP_B_ADMISSIBLE_MODULUS,
    "bootstrap_b_admissible_remainder": BOOTSTRAP_B_ADMISSIBLE_REMAINDER,
    "bootstrap_ci_lower_quantile": BOOTSTRAP_CI_LOWER_QUANTILE,
    "bootstrap_ci_upper_quantile": BOOTSTRAP_CI_UPPER_QUANTILE,
    "plateau_window_m": PLATEAU_WINDOW_M,
    "plateau_mk_alpha": PLATEAU_MK_ALPHA,
    "plateau_half_window_rule": PLATEAU_HALF_WINDOW_RULE,
    "plateau_ci_width_threshold_elo": PLATEAU_CI_WIDTH_THRESHOLD_ELO,
    "plateau_equivalence_margin_elo": PLATEAU_EQUIVALENCE_MARGIN_ELO,
    "plateau_gpu_hours_min": PLATEAU_GPU_HOURS_MIN,
    "plateau_confirmation_count": PLATEAU_CONFIRMATION_COUNT,
}


def protocol_fingerprint() -> str:
    """Return the sha256 hex digest of :data:`REGISTRY`'s canonical JSON.

    Canonical = sorted keys, compact separators -- one deterministic byte string per
    registry content, independent of dict insertion order or whitespace.

    Returns:
        A 64-character lowercase hex digest. Adding a covered constant or changing
        one's value always changes this digest; the whole point is that nobody has
        to remember to bump anything separately for that to be true.
    """
    canonical = json.dumps(REGISTRY, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def eval_config_snapshot() -> dict[str, Any]:
    """Return the pinned eval-config values a cell header snapshots (pins 1/4/5).

    A plain, JSON-safe view of just the *config* constants (as opposed to the
    protocol-shape/seed-label entries also in :data:`REGISTRY`) -- what a real
    caller's own config data (e.g. a loaded ``configs/m4_eval.json``) is expected to
    mirror. Deliberately a separate mechanism from :func:`protocol_fingerprint`:
    ``core.eval_store`` compares a resumed cell's stored snapshot against a
    caller-supplied *current* snapshot, which catches a caller-side config drift
    (e.g. an edited JSON file) independently of whether this module's own constants
    ever changed.

    Returns:
        ``{"pairs_per_cell", "eval_sims", "rung8_lag_divisor",
        "rung8_earliest_version"}`` at their current pinned values.
    """
    return {
        "pairs_per_cell": PAIRS_PER_CELL,
        "eval_sims": EVAL_SIMS,
        "rung8_lag_divisor": RUNG8_LAG_DIVISOR,
        "rung8_earliest_version": RUNG8_EARLIEST_VERSION,
    }
