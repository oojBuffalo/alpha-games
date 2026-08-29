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

**Doc-first status, as observable on this branch.** tasks/m4/001's actual design-doc
amendment -- the §9/status-header/§12 edits that pin pairs-per-cell, the eval
search-form sim budget, and the rung-8 rule -- lives on a sibling branch
(``docs/m4-pin-eval-protocol``) that, as of this module's own commit, is not an
ancestor of this line of history: ``metadocs/blokus-duo-az-design-v0_5.md`` checked
out here still carries the "to pin doc-first at M4" flags that amendment resolves.
The constants below already match that pending amendment's values, so this module is
the *code* side of the pin, staged ahead of the doc branch landing -- not a claim
that the doc has already been amended in this history. Per the project's "design-doc
changes precede divergent code" rule, the two branches must merge together (or the
doc branch first) before any "pin" reference below should be read as citing an
already-merged doc section rather than the value the pending amendment specifies.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: This amendment's version (tasks/m4/001 pin 10). A human-legible label, bumped
#: deliberately alongside a doc amendment that changes a covered convention's value --
#: distinct from :func:`protocol_fingerprint`, which changes automatically on *any*
#: registry drift regardless of whether this constant was remembered to move too.
PROTOCOL_VERSION = 1

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
    "pairs_per_cell": PAIRS_PER_CELL,
    "eval_sims": EVAL_SIMS,
    "rung8_lag_divisor": RUNG8_LAG_DIVISOR,
    "rung8_earliest_version": RUNG8_EARLIEST_VERSION,
    "bootstrap_b_production": BOOTSTRAP_B_PRODUCTION,
    "bootstrap_b_admissible_modulus": BOOTSTRAP_B_ADMISSIBLE_MODULUS,
    "bootstrap_b_admissible_remainder": BOOTSTRAP_B_ADMISSIBLE_REMAINDER,
    "bootstrap_ci_lower_quantile": BOOTSTRAP_CI_LOWER_QUANTILE,
    "bootstrap_ci_upper_quantile": BOOTSTRAP_CI_UPPER_QUANTILE,
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
