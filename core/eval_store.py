"""Durable per-cell eval record store (design doc §9; tasks/m4/005, subtasks 5.2/5.3).

Stdlib ``json`` only -- no torch, no ``games.*`` import. This module owns three
durable artifacts under ``<run_dir>/eval/``:

  * ``cells/<cell_id>.jsonl`` -- one append-only file per (candidate, rung, opponent)
    cell, single writer, header line first (constant-per-cell provenance) then one
    line per mirrored pair the §1 bootstrap resamples over. Byte-deterministic given
    its seeds: no wall-clock field appears anywhere in a cell file.
  * ``manifest.json`` -- the scheduled -> complete state machine over cell ids, plus
    every member version's full required cell-id set (registered atomically at
    scheduling time) and the *only* wall-clock fields this module writes
    (``scheduled_at`` / ``completed_at``), explicitly outside every determinism claim
    below.
  * The read side, :func:`load_snapshot` -- one atomic manifest read frozen into an
    immutable :class:`EvalSnapshot`: completed cells are immutable once complete, so a
    snapshot never races a live writer, and a partial cell is structurally invisible
    to it.

**Cell identity (bijective, filesystem-safe).** A cell is the triple
``(candidate_version, rung, opponent_id)`` -- the candidate's checkpoint version and
agent-form rung (5, 6, or 7) crossed with one opponent identity string (a frozen
network-free rung, or a rung-8 historical ``rung7-v1-<u>``). :func:`build_cell_id` /
:func:`parse_cell_id` canonicalize this into (and back out of) one filename-safe
string, so a changed sim budget or algorithm is a different opponent/candidate
identity string and therefore a different cell id -- it can never silently append
under an existing cell (review S3).

**Two independent drift checks, on purpose.** Every header stamps
``(protocol_version, protocol_fingerprint)`` freshly recomputed from
``core.eval_protocol`` at both write and resume time -- catches a *code*-level
convention change. It separately stamps a caller-supplied ``eval_config`` snapshot
(pairs-per-cell, S, the rung-8 rule as the caller's own config currently has them) --
catches a *caller config* drift (e.g. an edited ``configs/m4_eval.json``) that never
touched ``core.eval_protocol`` at all. :func:`open_cell_for_resume` asserts both,
raising the specific, differently-typed error so a caller can tell which one moved.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from core.elo import Match
from core.eval_protocol import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    protocol_fingerprint,
)
from core.runner import PairResult

_EVAL_DIRNAME = "eval"
_CELLS_DIRNAME = "cells"
_MANIFEST_NAME = "manifest.json"

_STATUS_SCHEDULED = "scheduled"
_STATUS_COMPLETE = "complete"


class SchemaVersionError(Exception):
    """Raised when a stored header/manifest's ``schema_version`` is not understood.

    Never a warning and never a silent best-effort parse: an unrecognized shape means
    the fields that follow cannot be trusted to mean what this code thinks they mean.
    """


class ProtocolMismatchError(Exception):
    """Raised on resume when a stored header's protocol stamp disagrees with the
    current ``core.eval_protocol`` registry -- a code-level convention changed since
    this cell was opened. Distinct from :class:`ConfigMismatchError`: this fires even
    if the caller's own config snapshot is untouched.
    """


class ConfigMismatchError(Exception):
    """Raised on resume when a stored header's identity/config fields disagree with
    the caller's current values -- e.g. an edited eval config, a different run id, or
    a candidate checkpoint whose artifact fingerprint moved. Distinct from
    :class:`ProtocolMismatchError`: this can fire with the protocol registry
    unchanged.
    """


class ManifestError(Exception):
    """Raised on an illegal manifest transition (re-registering a member with a
    different required-cell set, completing a never-scheduled cell) or on a
    manifest/cell-file disagreement caught while building a snapshot (a cell the
    manifest marks complete whose file is missing, mismatched, or short of its
    pinned pair count).
    """


class CorruptedCellError(Exception):
    """Raised when a cell file has a newline-terminated line that fails to decode
    as UTF-8 or parse as JSON -- real corruption (bit rot, a bug, a stray second
    writer), not the crash-torn partial write :func:`_truncate_torn_tail` exists to
    discard.

    Deliberately never conflated with an absent tail: only a file's *final* line
    lacking its trailing newline is a crash-torn write (plain binary file iteration
    guarantees no earlier line can lack one), so a line that already has its
    newline but still fails to parse is corruption sitting mid-stream, not a torn
    end -- silently truncating from there would also discard every subsequent,
    already-durable, possibly-valid line with no record that anything was lost.
    """


# ---------------------------------------------------------------------------------
# Cell identity: canonical, bijective, filesystem-safe.
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class CellId:
    """One (candidate, rung, opponent) evaluation cell -- the §1 resampling grain.

    Attributes:
        candidate_version: The candidate checkpoint's model-version ordinal.
        rung: The candidate's agent-form rung number (5, 6, or 7 in v1).
        opponent_id: The opponent agent's identity string -- one of the game's
            frozen network-free rungs (e.g. ``"random"``) or a historical rung-7
            form (``"rung7-v1-<u>"``, rung 8). Opaque to this module.
    """

    candidate_version: int
    rung: int
    opponent_id: str

    @property
    def candidate_identity(self) -> str:
        """The candidate's own form-versioned identity string (e.g. ``"rung7-v1-12"``).

        Matches ``core.eval_agents``' ``f"rung{form}-v1-{model_version}"`` naming
        exactly -- this is a read of that convention, not a second source of truth
        for it (a real caller always has the actual string on hand and should pass
        it explicitly to :func:`build_header` rather than relying on this).
        """
        return f"rung{self.rung}-v1-{self.candidate_version}"

    def to_string(self) -> str:
        """Return this cell's canonical id string (see :func:`build_cell_id`)."""
        return build_cell_id(self.candidate_version, self.rung, self.opponent_id)


def build_cell_id(candidate_version: int, rung: int, opponent_id: str) -> str:
    """Canonically encode a cell triple as one filename-safe, bijective string.

    Format: ``"<candidate_version>.<rung>.<percent-encoded opponent_id>"``.
    ``candidate_version``/``rung`` are formatted via plain ``str(int)`` (a bijection
    on the integer domain, and never containing ``"."``); ``opponent_id`` is
    percent-encoded with every unsafe-for-a-filename character escaped
    (``urllib.parse.quote(opponent_id, safe="")``), most importantly ``"/"`` (so an
    opponent id can never smuggle a path separator into the cell filename).

    This does *not* escape ``"."``: ``quote`` never percent-encodes it, under any
    ``safe=`` setting, because ``.`` sits in its permanently-unreserved character
    set (verify: ``quote("a.b/c", safe="") == "a.b%2Fc"`` -- the dot survives, only
    ``/`` is escaped). Bijectivity does not depend on it being escaped:
    :func:`parse_cell_id` splits on ``"."`` with ``maxsplit=2``, so the first two
    fields consume exactly the integer prefix and everything after the second dot
    -- however many further literal dots ``opponent_id`` contains -- always
    reassembles into one opponent field untouched. See :func:`parse_cell_id` for
    the inverse.

    Args:
        candidate_version: The candidate checkpoint's version ordinal.
        rung: The candidate's agent-form rung number.
        opponent_id: The opponent's identity string; any string round-trips.

    Returns:
        The canonical cell id string (used verbatim as ``<cell_id>.jsonl``'s stem).
    """
    return f"{candidate_version}.{rung}.{quote(opponent_id, safe='')}"


def parse_cell_id(cell_id: str) -> CellId:
    """Invert :func:`build_cell_id`.

    Args:
        cell_id: A string previously returned by :func:`build_cell_id`.

    Returns:
        The original :class:`CellId` triple.

    Raises:
        ValueError: If ``cell_id`` does not have the ``"<int>.<int>.<encoded>"``
            shape (fewer than 3 dot-separated fields, or a non-integer first/second
            field).
    """
    parts = cell_id.split(".", 2)
    if len(parts) != 3:
        raise ValueError(
            f"malformed cell id {cell_id!r}: expected 3 dot-separated fields, got {len(parts)}"
        )
    version_s, rung_s, opponent_enc = parts
    try:
        candidate_version = int(version_s)
        rung = int(rung_s)
    except ValueError as exc:
        raise ValueError(
            f"malformed cell id {cell_id!r}: non-integer candidate_version/rung field"
        ) from exc
    return CellId(candidate_version=candidate_version, rung=rung, opponent_id=unquote(opponent_enc))


def eval_dir(run_dir: Path | str) -> Path:
    """Return the eval root under a run directory.

    Args:
        run_dir: The run's root directory.

    Returns:
        ``<run_dir>/eval``.
    """
    return Path(run_dir) / _EVAL_DIRNAME


def cells_dir(run_dir: Path | str) -> Path:
    """Return the per-cell file directory under a run directory.

    Args:
        run_dir: The run's root directory.

    Returns:
        ``<run_dir>/eval/cells``.
    """
    return eval_dir(run_dir) / _CELLS_DIRNAME


def cell_path(run_dir: Path | str, cell_id: str) -> Path:
    """Return the on-disk path for one cell's ``.jsonl`` file.

    Args:
        run_dir: The run's root directory.
        cell_id: A canonical cell id (see :func:`build_cell_id`).

    Returns:
        ``<run_dir>/eval/cells/<cell_id>.jsonl``.
    """
    return cells_dir(run_dir) / f"{cell_id}.jsonl"


def manifest_path(run_dir: Path | str) -> Path:
    """Return the cell manifest's path under a run directory.

    Args:
        run_dir: The run's root directory.

    Returns:
        ``<run_dir>/eval/manifest.json``.
    """
    return eval_dir(run_dir) / _MANIFEST_NAME


# ---------------------------------------------------------------------------------
# Cell header (constant-per-cell provenance; the file's first line).
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class CellHeader:
    """Constant-per-cell provenance -- a cell file's first line, written once.

    Never carries wall-clock (that lives in the manifest only, review S5): every
    field here is a pure function of the run/config/candidate that produced the
    cell, which is exactly what makes cell files byte-deterministic given their
    seeds.

    Attributes:
        schema_version: The on-disk record shape version
            (``core.eval_protocol.SCHEMA_VERSION`` at write time).
        protocol_version: ``core.eval_protocol.PROTOCOL_VERSION`` at write time.
        protocol_fingerprint: ``core.eval_protocol.protocol_fingerprint()`` at write
            time -- any covered-constant drift changes this and is caught loudly on
            resume (see :func:`open_cell_for_resume`).
        run_id: The watched production run's identity.
        cell_id: This cell's ``(candidate_version, rung, opponent_id)`` triple.
        candidate_identity: The candidate agent's full form-versioned identity
            string (e.g. ``"rung7-v1-12"``), supplied explicitly rather than
            re-derived so the header always reflects what the runner actually built.
        opponent_identity: The opponent agent's full identity string, likewise
            supplied explicitly (equal to ``cell_id.opponent_id`` by construction,
            carried here too so a reader never has to reconstruct it).
        eval_config: A caller-supplied, JSON-safe snapshot of the run's pinned eval
            values (pairs-per-cell, S, the rung-8 rule) -- see
            ``core.eval_protocol.eval_config_snapshot``. Compared structurally
            against the caller's *current* snapshot on resume; independent of
            ``protocol_fingerprint`` (see the module docstring).
        candidate_fingerprint: The candidate checkpoint's artifact fingerprint
            (``core.artifact_fingerprint.build_fingerprint``'s payload); opaque
            here, just round-tripped.
    """

    schema_version: int
    protocol_version: int
    protocol_fingerprint: str
    run_id: str
    cell_id: CellId
    candidate_identity: str
    opponent_identity: str
    eval_config: Mapping[str, Any]
    candidate_fingerprint: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return this header as a flat, JSON-serializable dict."""
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "protocol_fingerprint": self.protocol_fingerprint,
            "run_id": self.run_id,
            "candidate_version": self.cell_id.candidate_version,
            "rung": self.cell_id.rung,
            "opponent_id": self.cell_id.opponent_id,
            "candidate_identity": self.candidate_identity,
            "opponent_identity": self.opponent_identity,
            "eval_config": dict(self.eval_config),
            "candidate_fingerprint": dict(self.candidate_fingerprint),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CellHeader:
        """Reconstruct a header from :meth:`to_dict`'s output (or its JSON round-trip)."""
        return cls(
            schema_version=int(payload["schema_version"]),
            protocol_version=int(payload["protocol_version"]),
            protocol_fingerprint=str(payload["protocol_fingerprint"]),
            run_id=str(payload["run_id"]),
            cell_id=CellId(
                candidate_version=int(payload["candidate_version"]),
                rung=int(payload["rung"]),
                opponent_id=str(payload["opponent_id"]),
            ),
            candidate_identity=str(payload["candidate_identity"]),
            opponent_identity=str(payload["opponent_identity"]),
            eval_config=dict(payload["eval_config"]),
            candidate_fingerprint=dict(payload["candidate_fingerprint"]),
        )


def build_header(
    *,
    run_id: str,
    cell_id: CellId,
    candidate_identity: str,
    opponent_identity: str,
    eval_config: Mapping[str, Any],
    candidate_fingerprint: Mapping[str, Any],
) -> CellHeader:
    """Build a fresh :class:`CellHeader`, stamping the current protocol registry.

    Args:
        run_id: The watched production run's identity.
        cell_id: This cell's triple.
        candidate_identity: The candidate agent's full identity string.
        opponent_identity: The opponent agent's full identity string.
        eval_config: The caller's current pinned-eval-config snapshot (typically
            ``core.eval_protocol.eval_config_snapshot()``).
        candidate_fingerprint: The candidate checkpoint's artifact fingerprint.

    Returns:
        A header with ``schema_version``/``protocol_version``/``protocol_fingerprint``
        freshly read from ``core.eval_protocol``.
    """
    return CellHeader(
        schema_version=SCHEMA_VERSION,
        protocol_version=PROTOCOL_VERSION,
        protocol_fingerprint=protocol_fingerprint(),
        run_id=run_id,
        cell_id=cell_id,
        candidate_identity=candidate_identity,
        opponent_identity=opponent_identity,
        eval_config=dict(eval_config),
        candidate_fingerprint=dict(candidate_fingerprint),
    )


def _assert_known_schema_version(schema_version: int, *, context: str) -> None:
    """Raise :class:`SchemaVersionError` unless ``schema_version`` is understood."""
    if schema_version != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{context}: unsupported schema_version {schema_version!r} "
            f"(this code understands schema_version {SCHEMA_VERSION!r} only)"
        )


# ---------------------------------------------------------------------------------
# Pair records.
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class GameRecordSnapshot:
    """The per-game fields a pair record stores (§1 opening-balance audit input).

    Attributes:
        plies: Number of actions applied from the initial state to terminal.
        opening: The game's first action.
    """

    plies: int
    opening: int

    def to_dict(self) -> dict[str, int]:
        """Return this snapshot as a flat, JSON-serializable dict."""
        return {"plies": self.plies, "opening": self.opening}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GameRecordSnapshot:
        """Reconstruct a snapshot from :meth:`to_dict`'s output (or its JSON round-trip)."""
        return cls(plies=int(payload["plies"]), opening=int(payload["opening"]))


@dataclass(frozen=True)
class PairRecord:
    """One mirrored pair's stored outcome -- the §1 bootstrap's resampling unit.

    Deliberately carries no ``score_b`` (implicit: ``2 - score_a``, review S5/D1
    scoring convention) and no wall-clock field.

    Attributes:
        pair_index: Absolute pair index within its cell.
        pair_seed: ``derive_seed(cell_seed, "pair", pair_index)`` -- recorded so a
            stored pair is independently re-derivable without replaying it.
        score_a: Agent A's score over both games of the pair (0-2; draws 0.5/game).
        games: The two games' ``{plies, opening}`` snapshots, seat 0 (A first) then
            seat 1 (seats swapped).
    """

    pair_index: int
    pair_seed: int
    score_a: float
    games: tuple[GameRecordSnapshot, GameRecordSnapshot]

    def to_dict(self) -> dict[str, Any]:
        """Return this pair record as a flat, JSON-serializable dict."""
        return {
            "pair_index": self.pair_index,
            "pair_seed": self.pair_seed,
            "score_a": self.score_a,
            "games": [g.to_dict() for g in self.games],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PairRecord:
        """Reconstruct a pair record from :meth:`to_dict`'s output (or its JSON round-trip).

        Raises:
            ValueError: If ``payload["games"]`` does not have exactly 2 entries.
        """
        games_payload = payload["games"]
        if len(games_payload) != 2:
            raise ValueError(f"pair record must carry exactly 2 games, got {len(games_payload)}")
        return cls(
            pair_index=int(payload["pair_index"]),
            pair_seed=int(payload["pair_seed"]),
            score_a=float(payload["score_a"]),
            games=(
                GameRecordSnapshot.from_dict(games_payload[0]),
                GameRecordSnapshot.from_dict(games_payload[1]),
            ),
        )


def pair_result_to_record(result: PairResult) -> PairRecord:
    """Convert a runner :class:`~core.runner.PairResult` to a storable :class:`PairRecord`.

    Args:
        result: The mirrored-pair outcome from ``core.runner.play_pairs``.

    Returns:
        The equivalent :class:`PairRecord` (``score_b``/full utilities dropped --
        implicit and unneeded respectively, per the pinned pair-record schema).
    """
    fwd, rev = result.games
    return PairRecord(
        pair_index=result.pair_index,
        pair_seed=result.pair_seed,
        score_a=result.score_a,
        games=(
            GameRecordSnapshot(plies=fwd.plies, opening=fwd.opening),
            GameRecordSnapshot(plies=rev.plies, opening=rev.opening),
        ),
    )


def records_to_match(
    candidate_identity: str, opponent_identity: str, records: Sequence[PairRecord]
) -> Match:
    """Aggregate a cell's stored pair records into one ``core.elo.Match``.

    Mirrors ``core.elo.matches_from_pairs``'s aggregation exactly, over stored
    records instead of live ``PairResult``s, so every consumer (live match, or a
    replayed cell file) shares one aggregation rule.

    Args:
        candidate_identity: Ladder name of the candidate (agent "A" side).
        opponent_identity: Ladder name of the opponent (agent "B" side).
        records: The cell's pair records.

    Returns:
        ``(candidate_identity, opponent_identity, total_score_a, 2 * len(records))``.
    """
    total = sum(r.score_a for r in records)
    return (candidate_identity, opponent_identity, total, 2 * len(records))


# ---------------------------------------------------------------------------------
# Cell file I/O: append-only writer with torn-tail crash recovery.
# ---------------------------------------------------------------------------------


def _read_valid_lines(path: Path) -> tuple[list[str], int]:
    """Return every whole, parseable JSON line in ``path``, and the bytes they span.

    Only a missing trailing ``b"\\n"`` marks a line as an absent, crash-torn tail --
    and, because plain binary file iteration always splits on ``b"\\n"`` and keeps
    the terminator, that can only ever be true of the *last* line iteration yields;
    every earlier line is guaranteed newline-terminated. A newline-terminated line
    that still fails to decode as UTF-8 or parse as JSON is therefore never treated
    as an absent tail -- it is real corruption (bit rot, a bug, a stray second
    writer), and this raises :class:`CorruptedCellError` loudly rather than
    silently discarding it and everything durably written after it. Bytes beyond
    what the returned lines span are exactly the torn tail a resume must discard.

    Args:
        path: The cell file to scan.

    Returns:
        ``(lines, consumed_bytes)`` -- ``lines`` without their trailing newlines, in
        file order; ``consumed_bytes`` the byte offset immediately after the last
        whole line (equals the file size iff there is no torn tail).

    Raises:
        CorruptedCellError: If a newline-terminated line fails to decode/parse.
    """
    lines: list[str] = []
    consumed = 0
    with open(path, "rb") as fh:
        for raw_line in fh:
            if not raw_line.endswith(b"\n"):
                # Only the file's final line can land here (see docstring) -- a
                # genuine crash-torn partial write, correctly treated as absent.
                break
            try:
                text = raw_line.decode("utf-8")
                json.loads(text)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CorruptedCellError(
                    f"{path}: line {len(lines) + 1} is newline-terminated (not a "
                    f"crash-torn tail) but failed to decode/parse -- refusing to "
                    f"silently truncate it and every line after it: {exc}"
                ) from exc
            lines.append(text[:-1])
            consumed += len(raw_line)
    return lines, consumed


def _truncate_torn_tail(path: Path) -> list[str]:
    """Discard a crash-torn trailing partial line from ``path``, in place.

    Args:
        path: The cell file to recover.

    Returns:
        The file's whole lines (header included), post-truncation.

    Raises:
        CorruptedCellError: If a newline-terminated line in ``path`` fails to
            decode/parse -- real corruption, never truncated away (see
            :func:`_read_valid_lines`).
    """
    lines, consumed = _read_valid_lines(path)
    if path.stat().st_size != consumed:
        with open(path, "r+b") as fh:
            fh.truncate(consumed)
            fh.flush()
            os.fsync(fh.fileno())
    return lines


def write_header(path: Path, header: CellHeader) -> None:
    """Create a brand-new cell file with ``header`` as its sole, first line.

    Args:
        path: The cell file to create; its parent directory is created if missing.
        header: The header to write.

    Raises:
        FileExistsError: If ``path`` already exists -- use :func:`open_cell_for_resume`
            (or :func:`open_cell_for_write`) for an existing cell.
    """
    if path.exists():
        raise FileExistsError(f"cell file already exists: {path} (use open_cell_for_resume)")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(header.to_dict(), sort_keys=True, separators=(",", ":"))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def append_pair_record(path: Path, record: PairRecord) -> None:
    """Append one pair record as a JSON line, durable before returning.

    Args:
        path: An existing cell file (its header already written).
        record: The pair record to append.
    """
    line = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _assert_header_matches(stored: CellHeader, expected: CellHeader, path: Path) -> None:
    """Assert ``stored`` (from disk) agrees with ``expected`` (the caller's current
    view), raising the specific mismatch type so a caller can tell which moved.

    Args:
        stored: The header read back from ``path``.
        expected: The header the caller currently believes describes this cell.
        path: The cell file (for the error message only).

    Raises:
        ProtocolMismatchError: If ``protocol_version`` or ``protocol_fingerprint``
            differ -- a code-level convention changed. (``schema_version`` is
            checked separately, before this is ever called -- see
            :func:`open_cell_for_resume`.)
        ConfigMismatchError: If any other field differs -- a caller-side config or
            identity drift.
    """
    registry_fields = ("protocol_version", "protocol_fingerprint")
    mismatched_registry = [f for f in registry_fields if getattr(stored, f) != getattr(expected, f)]
    if mismatched_registry:
        raise ProtocolMismatchError(
            f"cell {path}: stored header disagrees with the current protocol registry "
            f"on {mismatched_registry}: stored="
            f"{ {f: getattr(stored, f) for f in mismatched_registry}!r}, current="
            f"{ {f: getattr(expected, f) for f in mismatched_registry}!r}"
        )
    other_fields = (
        "run_id",
        "cell_id",
        "candidate_identity",
        "opponent_identity",
        "eval_config",
        "candidate_fingerprint",
    )
    mismatched = [f for f in other_fields if getattr(stored, f) != getattr(expected, f)]
    if mismatched:
        raise ConfigMismatchError(
            f"cell {path}: stored header disagrees with the current config on "
            f"{mismatched}: stored={ {f: getattr(stored, f) for f in mismatched}!r}, "
            f"current={ {f: getattr(expected, f) for f in mismatched}!r}"
        )


def open_cell_for_resume(path: Path, expected_header: CellHeader) -> int:
    """Recover and validate an existing cell file, returning its next pair index.

    Truncates a crash-torn trailing partial line to whole lines first (so a
    resumed append can never land on the same physical line as a partial record),
    then asserts the stored header agrees with ``expected_header`` *before*
    returning anything -- the loud immutability guard (review S3): no append may
    follow a header that no longer describes the running config or protocol.

    Args:
        path: An existing cell file.
        expected_header: The header the caller currently expects this cell to
            carry (freshly built from its current run id, config, and the live
            ``core.eval_protocol`` registry).

    Returns:
        The next ``pair_index`` to write (``play_pairs(..., start_pair_index=...)``):
        one past the highest ``pair_index`` recorded, or ``0`` if only the header is
        present.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        CorruptedCellError: If a newline-terminated line in ``path`` fails to
            decode/parse -- real corruption, distinct from (and never conflated
            with) a crash-torn tail; nothing is truncated in this case.
        SchemaVersionError: If the stored header's ``schema_version`` is unknown.
        ProtocolMismatchError: If the stored header's protocol stamp disagrees with
            the current registry.
        ConfigMismatchError: If the stored header's config/identity fields disagree
            with ``expected_header``'s.
        ValueError: If the file has no header line at all.
    """
    if not path.exists():
        raise FileNotFoundError(f"cell file does not exist: {path}")
    lines = _truncate_torn_tail(path)
    if not lines:
        raise ValueError(f"cell file {path} has no header line")
    stored_header = CellHeader.from_dict(json.loads(lines[0]))
    _assert_known_schema_version(stored_header.schema_version, context=f"cell file {path}")
    _assert_header_matches(stored_header, expected_header, path)
    pair_indices = [json.loads(line)["pair_index"] for line in lines[1:]]
    return (max(pair_indices) + 1) if pair_indices else 0


def open_cell_for_write(run_dir: Path | str, header: CellHeader) -> int:
    """Open ``header.cell_id``'s file for writing, creating or resuming it as needed.

    The single entry point a writer should call before appending pair records: a
    brand-new cell gets its header written; an existing one is validated and
    crash-recovered via :func:`open_cell_for_resume`.

    Args:
        run_dir: The run's root directory.
        header: The header this cell should carry (built fresh from the caller's
            current run id, config, and protocol registry).

    Returns:
        The next ``pair_index`` this cell should write.

    Raises:
        FileNotFoundError: If the cell file is removed out from under this call
            between the existence check above and the resume below (propagated
            from :func:`open_cell_for_resume`; not reachable absent that race).
        CorruptedCellError: If an existing cell file has a newline-terminated line
            that fails to decode/parse (propagated from :func:`open_cell_for_resume`).
        SchemaVersionError: If an existing cell's stored header carries an
            unrecognized ``schema_version`` (propagated from
            :func:`open_cell_for_resume`).
        ProtocolMismatchError: If an existing cell's stored header disagrees with
            the current ``core.eval_protocol`` registry (propagated from
            :func:`open_cell_for_resume`).
        ConfigMismatchError: If an existing cell's stored header disagrees with
            ``header``'s config/identity fields (propagated from
            :func:`open_cell_for_resume`).
        ValueError: If an existing cell file has no header line at all
            (propagated from :func:`open_cell_for_resume`).
    """
    path = cell_path(run_dir, header.cell_id.to_string())
    if not path.exists():
        write_header(path, header)
        return 0
    return open_cell_for_resume(path, header)


def read_cell(path: Path | str) -> tuple[CellHeader, list[PairRecord]]:
    """Read one cell file's header and every recorded pair.

    Args:
        path: The cell file to read.

    Returns:
        ``(header, records)`` in on-disk (pair-index) order.

    Raises:
        ValueError: If the file is empty.
        SchemaVersionError: If the header's ``schema_version`` is unknown.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"cell file {path} is empty (no header line)")
    header = CellHeader.from_dict(json.loads(lines[0]))
    _assert_known_schema_version(header.schema_version, context=f"cell file {path}")
    records = [PairRecord.from_dict(json.loads(line)) for line in lines[1:] if line.strip()]
    return header, records


# ---------------------------------------------------------------------------------
# Manifest: scheduled -> complete, temp-then-os.replace, wall-clock lives here only.
# ---------------------------------------------------------------------------------


def _empty_manifest() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "members": {}, "cells": {}}


def _read_manifest(run_dir: Path | str) -> dict[str, Any]:
    """Read the manifest, or a fresh empty one if it does not exist yet.

    Raises:
        SchemaVersionError: If an existing manifest's ``schema_version`` is unknown.
    """
    path = manifest_path(run_dir)
    if not path.exists():
        return _empty_manifest()
    payload = json.loads(path.read_text(encoding="utf-8"))
    _assert_known_schema_version(payload.get("schema_version"), context=f"manifest {path}")
    return payload


def _atomic_write_manifest(run_dir: Path | str, payload: dict[str, Any]) -> None:
    """Write the manifest atomically (temp-name-then-``os.replace``).

    A reader can never observe a partially-written manifest: it either doesn't
    exist yet, or exists complete (the same primitive ``core.checkpoint`` /
    ``core.replay_shard`` use for their own replaceable artifacts).
    """
    path = manifest_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def register_member(
    run_dir: Path | str, member_version: int, required_cell_ids: Sequence[str]
) -> None:
    """Register ``member_version``'s full required cell-id set ("scheduled").

    Idempotent: re-registering the same member with an identical set (as a set --
    order and duplicates don't matter) is a no-op. Registering it again with a
    *different* set is a hard error -- a member's required-cell set must never
    change after scheduling, since :func:`load_snapshot`'s completeness check
    depends on it being fixed for the member's lifetime.

    Args:
        run_dir: The run's root directory.
        member_version: The checkpoint member version (>= 1; v0 is never a member).
        required_cell_ids: Every cell id this member must complete. Each must parse
            (via :func:`parse_cell_id`) to ``candidate_version == member_version``.

    Raises:
        ValueError: If ``member_version < 1``, ``required_cell_ids`` is empty, or
            some id's parsed ``candidate_version`` does not match ``member_version``.
        ManifestError: If ``member_version`` is already registered with a different
            required-cell set.
    """
    if member_version < 1:
        raise ValueError(
            f"member_version must be >= 1 (v0 is never a member), got {member_version}"
        )
    normalized = sorted(set(required_cell_ids))
    if not normalized:
        raise ValueError("required_cell_ids must be non-empty")
    for cid in normalized:
        parsed = parse_cell_id(cid)
        if parsed.candidate_version != member_version:
            raise ValueError(
                f"cell id {cid!r} belongs to candidate_version {parsed.candidate_version}, "
                f"not member_version {member_version}"
            )

    manifest = _read_manifest(run_dir)
    key = str(member_version)
    existing = manifest["members"].get(key)
    if existing is not None:
        if list(existing["required_cells"]) == normalized:
            return  # idempotent no-op: identical re-registration.
        raise ManifestError(
            f"member {member_version} is already registered with a different required-cell "
            f"set: stored={existing['required_cells']!r}, requested={normalized!r}"
        )

    manifest["members"][key] = {"required_cells": normalized}
    now = time.time()
    for cid in normalized:
        if cid not in manifest["cells"]:
            manifest["cells"][cid] = {
                "status": _STATUS_SCHEDULED,
                "scheduled_at": now,
                "completed_at": None,
            }
    _atomic_write_manifest(run_dir, manifest)


def complete_cell(run_dir: Path | str, cell_id: str) -> None:
    """Mark ``cell_id`` complete ("scheduled" -> "complete").

    Idempotent and one-directional: completing an already-complete cell is a
    no-op (its original ``completed_at`` is preserved) -- a completed cell is
    never reopened.

    Args:
        run_dir: The run's root directory.
        cell_id: The cell to complete.

    Raises:
        ManifestError: If ``cell_id`` was never scheduled (registered via
            :func:`register_member`).
    """
    manifest = _read_manifest(run_dir)
    entry = manifest["cells"].get(cell_id)
    if entry is None:
        raise ManifestError(f"cell {cell_id} was never scheduled; cannot complete")
    if entry["status"] == _STATUS_COMPLETE:
        return  # idempotent no-op: a completed cell is never reopened.
    entry["status"] = _STATUS_COMPLETE
    entry["completed_at"] = time.time()
    _atomic_write_manifest(run_dir, manifest)


def is_cell_complete(run_dir: Path | str, cell_id: str) -> bool:
    """Return whether ``cell_id`` is marked complete in the manifest right now.

    Args:
        run_dir: The run's root directory.
        cell_id: The cell to check.

    Returns:
        ``False`` for a cell that was never scheduled at all.
    """
    manifest = _read_manifest(run_dir)
    entry = manifest["cells"].get(cell_id)
    return entry is not None and entry["status"] == _STATUS_COMPLETE


# ---------------------------------------------------------------------------------
# Snapshot reader: the one atomic, race-free analysis view.
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSnapshot:
    """One immutable analysis view of an eval run's manifest (design doc §9; P2.2).

    Frozen at :func:`load_snapshot` time from a single manifest read: completed
    cells are immutable once complete, so this never races a live writer, and a
    partial (merely "scheduled") cell is structurally absent from it.

    Attributes:
        run_dir: The run directory this snapshot was read from.
        member_prefix: The maximal contiguous member-version prefix ``1..k`` (0 if
            member 1 itself is unregistered or incomplete) whose full registered
            required-cell set is entirely within ``completed_cell_ids`` -- what a
            consumer needing "the complete K-set" gates on.
        completed_cell_ids: Every cell id complete anywhere in the manifest at read
            time, independent of ``member_prefix`` contiguity (a later member's
            cells may be complete even if an earlier one has a hole; per-checkpoint
            live reporting reads this set directly).
        snapshot_fingerprint: sha256 over ``(schema_version, sorted completed cell
            ids, each cell file's content hash, member_prefix)`` -- byte-stable
            while a writer appends to an incomplete cell (such cells are excluded by
            construction), changes iff a *completed* cell's content changes, and is
            invariant to every manifest wall-clock field.
    """

    run_dir: Path
    member_prefix: int
    completed_cell_ids: frozenset[str]
    snapshot_fingerprint: str


def _compute_member_prefix(members: Mapping[str, Any], completed: frozenset[str]) -> int:
    """Return the maximal contiguous complete member prefix starting at version 1."""
    prefix = 0
    version = 1
    while True:
        entry = members.get(str(version))
        if entry is None:
            break
        required = entry["required_cells"]
        if not all(cid in completed for cid in required):
            break
        prefix = version
        version += 1
    return prefix


def load_snapshot(run_dir: Path | str) -> EvalSnapshot:
    """Read the manifest once and freeze it into an immutable :class:`EvalSnapshot`.

    For every cell the manifest marks complete, this also opens its file to hash
    its bytes and to cross-check that the file actually backs the manifest's claim
    (its header's own triple matches the cell id it is filed under, and it carries
    exactly the pinned pair count) -- a manifest/cell-file disagreement here is a
    corrupted store, not a race (completed cells never change), so it fails loudly
    rather than silently trusting the manifest.

    Args:
        run_dir: The run's root directory.

    Returns:
        The frozen snapshot.

    Raises:
        SchemaVersionError: If the manifest's or a cell file's ``schema_version`` is
            unknown.
        ManifestError: If a cell the manifest marks complete is missing on disk, its
            header's triple disagrees with its cell id, or its recorded pair count
            does not equal its own header's pinned ``pairs_per_cell``.
    """
    run_dir = Path(run_dir)
    manifest = _read_manifest(run_dir)
    members = manifest["members"]
    cells = manifest["cells"]

    completed_ids = sorted(
        cid for cid, entry in cells.items() if entry["status"] == _STATUS_COMPLETE
    )
    cell_hashes: dict[str, str] = {}
    for cid in completed_ids:
        path = cell_path(run_dir, cid)
        if not path.exists():
            raise ManifestError(
                f"manifest marks cell {cid} complete but its file is missing: {path}"
            )
        header, records = read_cell(path)
        parsed = parse_cell_id(cid)
        stored_triple = (
            header.cell_id.candidate_version,
            header.cell_id.rung,
            header.cell_id.opponent_id,
        )
        parsed_triple = (parsed.candidate_version, parsed.rung, parsed.opponent_id)
        if stored_triple != parsed_triple:
            raise ManifestError(
                f"cell {cid}: stored header's triple disagrees with its own filename"
            )
        try:
            expected_pairs = header.eval_config["pairs_per_cell"]
        except KeyError:
            raise ManifestError(
                f"cell {cid}: stored eval_config snapshot has no 'pairs_per_cell' key — "
                "the header shape has drifted from the recorded schema"
            ) from None
        if len(records) != expected_pairs:
            raise ManifestError(
                f"manifest marks cell {cid} complete but it has {len(records)} recorded pairs, "
                f"not the pinned {expected_pairs}"
            )
        cell_hashes[cid] = hashlib.sha256(path.read_bytes()).hexdigest()

    completed_frozen = frozenset(completed_ids)
    member_prefix = _compute_member_prefix(members, completed_frozen)
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "cell_ids": completed_ids,
        "cell_hashes": cell_hashes,
        "member_prefix": member_prefix,
    }
    snapshot_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return EvalSnapshot(
        run_dir=run_dir,
        member_prefix=member_prefix,
        completed_cell_ids=completed_frozen,
        snapshot_fingerprint=snapshot_fingerprint,
    )


def iter_cells(snapshot: EvalSnapshot) -> Iterator[Path]:
    """Yield every completed cell's file path in the snapshot, in sorted cell-id order.

    Args:
        snapshot: A snapshot from :func:`load_snapshot`.

    Yields:
        Each completed cell's path (see :func:`read_cell` to parse one).
    """
    for cid in sorted(snapshot.completed_cell_ids):
        yield cell_path(snapshot.run_dir, cid)
