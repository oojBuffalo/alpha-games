"""requirements.txt / requirements-dev.txt must mirror pyproject.toml's dependencies.

The two requirements files exist only for `pip install -r`-style setup (README); if they
drift from pyproject.toml (the source of truth), installs from the two paths silently diverge.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read_requirements(path: Path) -> list[str]:
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-r "):
            continue
        lines.append(line)
    return lines


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_requirements_txt_matches_pyproject_dependencies():
    assert _read_requirements(ROOT / "requirements.txt") == _pyproject()["project"]["dependencies"]


def test_requirements_dev_txt_matches_pyproject_dev_dependencies():
    dev_deps = _pyproject()["project"]["optional-dependencies"]["dev"]
    assert _read_requirements(ROOT / "requirements-dev.txt") == dev_deps
