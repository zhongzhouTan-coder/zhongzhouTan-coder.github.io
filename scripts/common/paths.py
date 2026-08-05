"""Repository path discovery shared by command-line tools."""

from __future__ import annotations

from pathlib import Path


def find_repository_root(location: str | Path) -> Path:
    """Find the repository root by walking upward from a script location."""
    path = Path(location).resolve()
    start = path.parent if path.is_file() else path

    for candidate in (start, *start.parents):
        if (candidate / "scripts").is_dir() and (candidate / "sources.json").is_file():
            return candidate

    raise RuntimeError(f"repository root not found from {path}")
