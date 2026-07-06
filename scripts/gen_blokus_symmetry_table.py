"""Generate the checked-in Klein-4 (g,a)→a′ symmetry-table fixture.

Writes ``tests/fixtures/blokus/symmetry_table.json`` with the orientation-table
hash and encoding conventions embedded (write-side hash serialization, §5.1).
Deterministic: re-running on unchanged code must be byte-identical.

Usage:
    python3 scripts/gen_blokus_symmetry_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from games.blokus_duo.actions import FIXTURE_CONVENTIONS, IN_BOUNDS_ACTIONS  # noqa: E402
from games.blokus_duo.pieces import orientation_table_hash  # noqa: E402
from games.blokus_duo.symmetry import GROUP_NAMES, build_action_maps  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "blokus"


def main() -> None:
    """Build the symmetry-table fixture and write it as canonical JSON."""
    maps = build_action_maps()
    payload = {
        "orientation_hash": orientation_table_hash(),
        "conventions": FIXTURE_CONVENTIONS,
        "actions": list(IN_BOUNDS_ACTIONS),
        "maps": {g: [maps[g][a] for a in IN_BOUNDS_ACTIONS] for g in GROUP_NAMES},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "symmetry_table.json"
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
