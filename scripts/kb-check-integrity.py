#!/usr/bin/env python3
"""Validate knowledge-base source/category integrity."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REF_RE = re.compile(r"^\s*-\s+(.+?)\s*$")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def front_matter_sources(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        return set()

    in_sources = False
    sources: set[str] = set()
    for line in lines[1:]:
        if line == "---":
            break
        if re.match(r"^[A-Za-z_]+:", line):
            in_sources = line.startswith("sources:")
            continue
        if in_sources:
            match = SOURCE_REF_RE.match(line)
            if match:
                sources.add(match.group(1).strip("\"'"))
    return sources


def source_id_suffix(source_id: str) -> str:
    prefix, _, value = source_id.partition(":")
    if prefix == "arxiv":
        return f"arxiv-{value}"
    if prefix in {"github", "nvidia", "paper", "web"}:
        return prefix
    return source_id.replace(":", "-")


def expected_raw_name(entry: dict, raw_path: str) -> str | None:
    path = Path(raw_path)
    suffix = path.suffix
    if suffix == ".pdf":
        source_suffix = entry.get("source_suffix", source_id_suffix(entry["id"]))
        return f"{entry['slug']}--{source_suffix}.pdf"
    if suffix in {".md", ".mdx"} and len(entry.get("raw_paths", [])) == 1:
        return f"{entry['slug']}{suffix}"
    return None


def main() -> int:
    categories = load_json(ROOT / "kb-categories.json")["categories"]
    manifest = load_json(ROOT / "sources.json")["sources"]
    errors: list[str] = []
    seen_ids: set[str] = set()
    manifest_raw_paths: set[str] = set()
    manifest_derived_paths: set[str] = set()

    for entry in manifest:
        source_id = entry["id"]
        category = entry["category"]
        if source_id in seen_ids:
            errors.append(f"duplicate source id: {source_id}")
        seen_ids.add(source_id)

        if category not in categories:
            errors.append(f"{source_id}: unknown category {category}")
            continue

        raw_prefix = categories[category]["raw_prefix"]
        derived_prefix = categories[category]["derived_prefix"]
        docs_prefix = categories[category]["docs_prefix"]

        for raw_path in entry.get("raw_paths", []):
            manifest_raw_paths.add(raw_path)
            full_raw_path = ROOT / raw_path
            if not raw_path.startswith(raw_prefix):
                errors.append(f"{source_id}: raw path outside category prefix: {raw_path}")
            if not full_raw_path.exists():
                errors.append(f"{source_id}: missing raw path: {raw_path}")
            expected_name = expected_raw_name(entry, raw_path)
            if expected_name and Path(raw_path).name != expected_name:
                errors.append(
                    f"{source_id}: raw filename should be {expected_name}, got {Path(raw_path).name}"
                )

        derived_path = entry.get("derived_path")
        if derived_path:
            manifest_derived_paths.add(derived_path)
            if not derived_prefix:
                errors.append(f"{source_id}: category has no derived prefix: {category}")
            elif not derived_path.startswith(derived_prefix):
                errors.append(f"{source_id}: derived path outside category prefix: {derived_path}")
            if not (ROOT / derived_path).exists():
                errors.append(f"{source_id}: missing derived path: {derived_path}")
            expected_derived_name = f"{entry['slug']}.md"
            if Path(derived_path).name != expected_derived_name:
                errors.append(
                    f"{source_id}: derived filename should be {expected_derived_name}, "
                    f"got {Path(derived_path).name}"
                )

        docs_path = entry.get("docs_path")
        if docs_path:
            if docs_prefix and not docs_path.startswith(docs_prefix):
                errors.append(f"{source_id}: docs path outside category prefix: {docs_path}")
            full_docs_path = ROOT / docs_path
            if not full_docs_path.exists():
                errors.append(f"{source_id}: missing docs path: {docs_path}")
            else:
                doc_sources = front_matter_sources(full_docs_path)
                for raw_path in entry.get("raw_paths", []):
                    if raw_path.endswith((".pdf", ".md", ".mdx")) and raw_path not in doc_sources:
                        errors.append(f"{source_id}: docs front matter missing source {raw_path}")
                if derived_path and derived_path not in doc_sources:
                    errors.append(f"{source_id}: docs front matter missing derived source {derived_path}")

    for raw_path in sorted(p for p in (ROOT / "raw").rglob("*") if p.is_file()):
        rel = raw_path.relative_to(ROOT).as_posix()
        if rel not in manifest_raw_paths and not rel.startswith("raw/codex/"):
            errors.append(f"raw file missing from sources.json: {rel}")

    for derived_path in sorted(p for p in (ROOT / "derived/pdf-markdown").rglob("*.md")):
        rel = derived_path.relative_to(ROOT).as_posix()
        if rel not in manifest_derived_paths:
            errors.append(f"derived markdown missing from sources.json: {rel}")

    if errors:
        for error in errors:
            print(f"kb integrity: {error}", file=sys.stderr)
        return 1

    print("kb integrity ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
