#!/usr/bin/env python3
"""Check source filenames against the knowledge-base naming policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source_id_suffix(source_id: str) -> str:
    prefix, _, value = source_id.partition(":")
    if prefix == "arxiv":
        return f"arxiv-{value}"
    if prefix in {"github", "nvidia", "paper", "web"}:
        return prefix
    return source_id.replace(":", "-")


def expected_raw_path(entry: dict, raw_path: str) -> str | None:
    path = Path(raw_path)
    category = entry["category"]
    suffix = path.suffix
    if suffix == ".pdf":
        source_suffix = entry.get("source_suffix", source_id_suffix(entry["id"]))
        return f"raw/{category}/{entry['slug']}--{source_suffix}.pdf"
    if suffix in {".md", ".mdx"} and len(entry.get("raw_paths", [])) == 1:
        return f"raw/{category}/{entry['slug']}{suffix}"
    return None


def expected_derived_path(entry: dict) -> str | None:
    if not entry.get("derived_path"):
        return None
    return f"derived/pdf-markdown/{entry['category']}/{entry['slug']}.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="sources.json")
    args = parser.parse_args()

    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))["sources"]
    mismatches = 0

    for entry in manifest:
        for raw_path in entry.get("raw_paths", []):
            expected = expected_raw_path(entry, raw_path)
            if expected and raw_path != expected:
                print(f"{raw_path} -> {expected}")
                mismatches += 1

        expected_derived = expected_derived_path(entry)
        if expected_derived and entry["derived_path"] != expected_derived:
            print(f"{entry['derived_path']} -> {expected_derived}")
            mismatches += 1

    if mismatches:
        print(f"{mismatches} path(s) do not match the naming policy")
        return 1

    print("source names ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
