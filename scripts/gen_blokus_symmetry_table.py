"""Generate the checked-in (g,a)→a′ symmetry-table fixtures, one per pinned instance.

Writes ``tests/fixtures/blokus/symmetry_table.json`` (full 14×14 game) and
``tests/fixtures/blokus_micro/symmetry_table.json`` (the §5.3 micro instance),
each with **its own** orientation-table hash and encoding conventions embedded
(write-side hash serialization, §5.1; micro ids are re-derived within the piece
subset, so its digest differs by construction). Deterministic: re-running on
unchanged code must be byte-identical.

Usage:
    python3 scripts/gen_blokus_symmetry_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from games.blokus_duo.actions import action_codec  # noqa: E402
from games.blokus_duo.config import FULL_CONFIG, MICRO_CONFIG, BlokusConfig  # noqa: E402
from games.blokus_duo.pieces import orientation_table_hash  # noqa: E402
from games.blokus_duo.symmetry import symmetry_group  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Fixture directory per pinned instance, relative to ``tests/fixtures``.
INSTANCES: tuple[tuple[str, BlokusConfig], ...] = (
    ("blokus", FULL_CONFIG),
    ("blokus_micro", MICRO_CONFIG),
)


def build_payload(config: BlokusConfig) -> dict:
    """Build one instance's symmetry-table fixture payload.

    Args:
        config: The instance to enumerate.

    Returns:
        The fixture dict: orientation hash, encoding conventions, the sorted
        in-bounds action ids, and per group element the parallel list of images.
    """
    codec = action_codec(config)
    group = symmetry_group(config)
    maps = group.action_maps()
    actions = codec.in_bounds_actions
    return {
        "orientation_hash": orientation_table_hash(config),
        "conventions": codec.fixture_conventions,
        "actions": list(actions),
        "maps": {g: [maps[g][a] for a in actions] for g in group.names},
    }


def write_fixture(config: BlokusConfig, out_dir: Path) -> Path:
    """Write one instance's fixture as canonical JSON.

    Args:
        config: The instance to enumerate.
        out_dir: Directory to write ``symmetry_table.json`` into (created if
            missing).

    Returns:
        The path written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "symmetry_table.json"
    payload = build_payload(config)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def main(root: Path = ROOT) -> None:
    """Write every pinned instance's symmetry-table fixture.

    Args:
        root: Repo root to write under; overridden by the byte-stability test,
            which regenerates into a temp tree and diffs against the committed
            fixtures.
    """
    for name, config in INSTANCES:
        path = write_fixture(config, root / "tests" / "fixtures" / name)
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
