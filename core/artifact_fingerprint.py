"""The canonical artifact fingerprint (design doc §5.1/§12 M3, read-side).

One small, game-agnostic helper that both the replay-shard writer/reader
(``core.replay_shard``) and the checkpoint loader (a later milestone) share
verbatim: it turns an adapter's *declared* surface into a canonical,
JSON-serializable structure, and it fails loudly — never a warning, never a
silent coercion — the moment a stored fingerprint disagrees with the live one
recomputed from the adapter that is about to read it.

"Declared surface" is deliberate: this module reads only ``Game``-ABC
properties (``policy_shape``, ``input_planes``, ``input_shape``,
``value_targets``, ``orientation_table_hash``, ``encoding_conventions``) plus,
duck-typed, an adapter's optional ``config`` attribute — never a hardcoded
Blokus constant (no literal ``109``, no literal ``14``). A game that declares
nothing beyond the M0 surface (Tic-Tac-Toe, Connect Four, Othello) still gets a
complete, meaningful fingerprint; it just carries ``None``/``{}`` in the slots
those games don't use.

The fingerprint's own shape is versioned by :data:`SCHEMA_VERSION`, itself one
of the fingerprint's fields — so a schema bump is *automatically* a "mismatch"
under :func:`compare_fingerprints` (the live fingerprint always carries the
current code's version; an old shard's stored version can never equal it once
the constant moves), with no separate "unsupported version" code path needed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from core.game import Game

# Bumped whenever this module's fingerprint *shape* changes (a field added,
# renamed, or reinterpreted) — never for an adapter merely changing its own
# declared values (a new orientation hash from a config change is a normal,
# correctly-detected mismatch, not a schema bump). Pinned in code, integer, per
# the M3 design constraint; every reader compares against this exact value.
SCHEMA_VERSION = 1

# Sentinel distinguishing "key absent" from "key present with value None" when
# diffing two fingerprints — ``None`` is a legitimate stored value (a game with
# no orientation table), so it cannot double as the "missing" marker.
_MISSING = object()


class FingerprintMismatchError(Exception):
    """Raised when a stored artifact fingerprint disagrees with the live one.

    The message names every mismatched top-level field (schema version, game
    identity, shapes, encoding conventions, value-target scaling, orientation
    hash) with its stored and live values — never a bare "mismatch", and never
    a warning: a mismatched fingerprint means the bytes that follow cannot be
    trusted to mean what the reader thinks they mean.
    """


def _config_identity(game: Game) -> str | None:
    """Return a stable identity string for a config-parameterized adapter.

    Duck-typed on an optional ``config`` attribute (``BlokusDuo.config``,
    §5.3): a frozen, hashable dataclass whose ``repr`` is a pure function of
    its field values (never a memory address), so it is stable across
    processes and across the same config re-constructed from scratch. Adapters
    that are not config-parameterized (TTT, Connect 4's plain ints, Othello)
    have no such attribute and get ``None`` — not a hardcoded per-game case
    here.

    Args:
        game: The adapter instance.

    Returns:
        ``repr(game.config)``, or ``None`` if the adapter declares no
        ``config`` attribute.
    """
    config = getattr(game, "config", None)
    if config is None:
        return None
    return repr(config)


def build_fingerprint(game: Game) -> dict[str, Any]:
    """Build the canonical fingerprint of ``game``'s declared encoding surface.

    Every value is a plain JSON-native type (``str``, ``int``, ``float``,
    ``bool``, ``None``, ``list``, ``dict``) — deliberately never a ``tuple`` —
    so that ``build_fingerprint(game) == json.loads(canonical_json(fingerprint))``
    holds: a fingerprint read back from a shard/checkpoint compares equal to a
    freshly-built one field for field, with no tuple-vs-list false mismatch.

    Args:
        game: The adapter to fingerprint.

    Returns:
        A dict with keys ``schema_version``, ``game_identity``,
        ``policy_shape``, ``input_planes``, ``input_shape``,
        ``encoding_conventions``, ``value_target_scaling``,
        ``orientation_hash`` — the exact six-field-plus-version layout
        described in design doc §12 M3.
    """
    spec = game.value_targets
    return {
        "schema_version": SCHEMA_VERSION,
        "game_identity": {
            "adapter_class": f"{type(game).__module__}.{type(game).__qualname__}",
            "config": _config_identity(game),
        },
        "policy_shape": list(game.policy_shape),
        "input_planes": game.input_planes,
        "input_shape": list(game.input_shape),
        "encoding_conventions": json.loads(json.dumps(dict(game.encoding_conventions))),
        "value_target_scaling": {
            "primary_name": spec.primary_name,
            "aux_names": list(spec.aux_names),
            "aux_loss_weights": list(spec.aux_loss_weights),
        },
        "orientation_hash": game.orientation_table_hash,
    }


def canonical_json(fingerprint: Mapping[str, Any]) -> str:
    """Serialize a fingerprint to its canonical JSON form.

    Sorted keys and a fixed separator style make the bytes independent of
    Python dict-construction order — the same fingerprint content always
    serializes identically, whatever process or PYTHONHASHSEED produced it
    (Invariant 4's "never set-iteration order" concern, generalized).

    Args:
        fingerprint: A fingerprint dict, typically from :func:`build_fingerprint`.

    Returns:
        The canonical JSON string.
    """
    return json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))


def fingerprint_digest(fingerprint: Mapping[str, Any]) -> str:
    """Return the sha256 hex digest of a fingerprint's canonical JSON.

    The same canonicalize-then-sha256 pattern as
    ``games.blokus_duo.pieces.orientation_table_hash`` — a short, stable
    handle for logs/filenames when the full structure is unnecessary.

    Args:
        fingerprint: A fingerprint dict, typically from :func:`build_fingerprint`.

    Returns:
        The hex sha256 digest.
    """
    return hashlib.sha256(canonical_json(fingerprint).encode("ascii")).hexdigest()


def compare_fingerprints(stored: Mapping[str, Any], live: Mapping[str, Any]) -> None:
    """Assert two fingerprints agree, or fail loudly naming every difference.

    Comparison is at top-level-field granularity (``schema_version``,
    ``game_identity``, ``policy_shape``, ``input_planes``, ``input_shape``,
    ``encoding_conventions``, ``value_target_scaling``, ``orientation_hash``):
    each mismatched field is reported with its stored and live values, so a
    caller sees exactly what disagreed rather than an opaque "not equal". An
    unsupported/older schema version is not a special case — it always shows
    up as a ``schema_version`` mismatch, because ``live`` is always built by
    the current code and therefore always carries the current
    :data:`SCHEMA_VERSION`.

    Args:
        stored: The fingerprint read back from a shard or checkpoint.
        live: The fingerprint freshly built from the adapter about to read it
            (:func:`build_fingerprint`).

    Raises:
        FingerprintMismatchError: If any top-level field disagrees. The
            message names every mismatched field.
    """
    keys = sorted(set(stored) | set(live))
    mismatched = [k for k in keys if stored.get(k, _MISSING) != live.get(k, _MISSING)]
    if not mismatched:
        return
    details = "; ".join(
        f"{k}: stored={stored.get(k, _MISSING)!r} live={live.get(k, _MISSING)!r}"
        for k in mismatched
    )
    raise FingerprintMismatchError(f"artifact fingerprint mismatch on field(s) -- {details}")
