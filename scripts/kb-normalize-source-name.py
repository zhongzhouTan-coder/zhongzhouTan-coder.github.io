#!/usr/bin/env python3
"""Check source filenames against the knowledge-base naming policy."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def source_id_suffix(source_id: str) -> str:
    prefix, _, value = source_id.partition(":")
    if prefix == "arxiv":
        return f"arxiv-{value}"
    if prefix in {"github", "nvidia", "paper", "web"}:
        return prefix
    return source_id.replace(":", "-")


def expected_raw_path(entry: dict[str, Any], raw_path: str) -> str | None:
    path = Path(raw_path)
    category = entry["category"]
    suffix = path.suffix
    if entry.get("kind") == "web":
        captured_at = entry.get("captured_at", "")
        revision = entry.get("revision", "")
        if (
            isinstance(captured_at, str)
            and len(captured_at) >= 10
            and SHA256_RE.fullmatch(revision)
        ):
            snapshot = (
                f"{entry['slug']}--web-{captured_at[:10]}-{revision[:12]}"
            )
            if raw_path.endswith(".metadata.json"):
                return f"raw/{category}/{snapshot}.metadata.json"
            if suffix == ".html":
                return f"raw/{category}/{snapshot}.html"
        return None
    if entry.get("kind") == "repository" and suffix in {".md", ".mdx"}:
        repo_slug = entry.get("repo_slug")
        revision = entry.get("revision", "")
        if repo_slug and SHA_RE.fullmatch(revision):
            return (
                f"raw/{category}/{repo_slug}-codebase--github-"
                f"{revision[:12]}{suffix}"
            )
        return None
    if suffix == ".pdf":
        source_suffix = entry.get("source_suffix", source_id_suffix(entry["id"]))
        return f"raw/{category}/{entry['slug']}--{source_suffix}.pdf"
    if suffix in {".md", ".mdx"} and len(entry.get("raw_paths", [])) == 1:
        return f"raw/{category}/{entry['slug']}{suffix}"
    return None


def expected_derived_path(entry: dict[str, Any]) -> str | None:
    actual_path = entry.get("derived_path")
    if not actual_path:
        return None
    if entry.get("kind") == "repository":
        repo_slug = entry.get("repo_slug")
        revision = entry.get("revision")
        if repo_slug and SHA_RE.fullmatch(revision or ""):
            return (
                f"derived/repo-analysis/{entry['category']}/"
                f"{repo_slug}/{revision}/"
            )
        return None
    if entry.get("kind") == "web":
        captured_at = entry.get("captured_at", "")
        revision = entry.get("revision", "")
        if (
            isinstance(captured_at, str)
            and len(captured_at) >= 10
            and SHA256_RE.fullmatch(revision)
        ):
            snapshot = (
                f"{entry['slug']}--web-{captured_at[:10]}-{revision[:12]}"
            )
            return f"derived/web-markdown/{entry['category']}/{snapshot}.md"
        return None
    flat_path = f"derived/pdf-markdown/{entry['category']}/{entry['slug']}.md"
    folder_path = (
        f"derived/pdf-markdown/{entry['category']}/"
        f"{entry['slug']}/{entry['slug']}.md"
    )
    if actual_path in {flat_path, folder_path}:
        return actual_path
    return flat_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Knowledge-base root (defaults to this script's repository).",
    )
    parser.add_argument("--manifest", default="sources.json")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable result on stdout.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = json.loads(
        (root / args.manifest).read_text(encoding="utf-8")
    )["sources"]
    mismatches: list[dict[str, str]] = []

    for entry in manifest:
        for raw_path in entry.get("raw_paths", []):
            expected = expected_raw_path(entry, raw_path)
            if expected and raw_path != expected:
                mismatches.append(
                    {"kind": "raw", "actual": raw_path, "expected": expected}
                )

        expected_derived = expected_derived_path(entry)
        if expected_derived and entry["derived_path"] != expected_derived:
            mismatches.append(
                {
                    "kind": "derived",
                    "actual": entry["derived_path"],
                    "expected": expected_derived,
                }
            )

    if mismatches:
        if args.json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "mismatch_count": len(mismatches),
                        "mismatches": mismatches,
                    },
                    indent=2,
                )
            )
            return 1
        for mismatch in mismatches:
            print(f"{mismatch['actual']} -> {mismatch['expected']}")
        print(f"{len(mismatches)} path(s) do not match the naming policy")
        return 1

    if args.json:
        print(
            json.dumps(
                {"ok": True, "mismatch_count": 0, "mismatches": []},
                indent=2,
            )
        )
        return 0
    print("source names ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
