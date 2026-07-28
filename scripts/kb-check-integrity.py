#!/usr/bin/env python3
"""Validate knowledge-base source, revision, and category integrity."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY_ID_RE = re.compile(
    r"^github:(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)@"
    r"(?P<commit>[0-9a-f]{40})$"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_front_matter(path: Path) -> dict[str, str | list[str]]:
    """Parse the scalar and list subset used by this repository."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        return {}

    result: dict[str, str | list[str]] = {}
    list_key: str | None = None
    for line in lines[1:]:
        if line == "---":
            break
        list_item = re.match(r"^\s+-\s+(.+?)\s*$", line)
        if list_key and list_item:
            value = list_item.group(1).strip("\"'")
            current = result.setdefault(list_key, [])
            if isinstance(current, list):
                current.append(value)
            continue

        field = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):(?:\s*(.*))?$", line)
        if not field:
            list_key = None
            continue
        key, raw_value = field.groups()
        if raw_value:
            result[key] = raw_value.strip().strip("\"'")
            list_key = None
        else:
            result[key] = []
            list_key = key
    return result


def front_matter_sources(path: Path) -> set[str]:
    sources = parse_front_matter(path).get("sources", [])
    return set(sources) if isinstance(sources, list) else set()


def body_near_top(path: Path, line_limit: int = 60) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    body_start = 0
    if lines and lines[0] == "---":
        for index, line in enumerate(lines[1:], start=1):
            if line == "---":
                body_start = index + 1
                break
    return "\n".join(lines[body_start : body_start + line_limit])


def source_id_suffix(source_id: str) -> str:
    prefix, _, value = source_id.partition(":")
    if prefix == "arxiv":
        return f"arxiv-{value}"
    if prefix in {"github", "nvidia", "paper", "web"}:
        return prefix
    return source_id.replace(":", "-")


def expected_raw_name(entry: dict[str, Any], raw_path: str) -> str | None:
    path = Path(raw_path)
    suffix = path.suffix
    if entry.get("kind") == "repository" and suffix in {".md", ".mdx"}:
        repo_slug = entry.get("repo_slug")
        revision = entry.get("revision", "")
        if repo_slug and SHA_RE.fullmatch(revision):
            return f"{repo_slug}-codebase--github-{revision[:12]}{suffix}"
        return None
    if suffix == ".pdf":
        source_suffix = entry.get("source_suffix", source_id_suffix(entry["id"]))
        return f"{entry['slug']}--{source_suffix}.pdf"
    if suffix in {".md", ".mdx"} and len(entry.get("raw_paths", [])) == 1:
        return f"{entry['slug']}{suffix}"
    return None


def normalized_path(path: str) -> str:
    return Path(path.rstrip("/")).as_posix()


def is_safe_relative_path(path: str, prefix: str) -> bool:
    candidate = Path(path)
    return (
        not candidate.is_absolute()
        and ".." not in candidate.parts
        and bool(candidate.parts)
        and candidate.parts[0] == prefix
    )


def cites_file_beneath(root: Path, doc_sources: set[str], directory: str) -> bool:
    full_directory = (root / directory).resolve()
    for source in doc_sources:
        full_source = (root / source).resolve()
        if full_source == full_directory or not full_source.is_file():
            continue
        try:
            full_source.relative_to(full_directory)
        except ValueError:
            continue
        return True
    return False


def valid_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def validate_repository_entry(
    root: Path,
    entry: dict[str, Any],
    categories: dict[str, Any],
    errors: list[str],
    manifest_repo_analysis_paths: set[str],
) -> None:
    source_id = entry["id"]
    match = REPOSITORY_ID_RE.fullmatch(source_id)
    if not match:
        errors.append(
            f"{source_id}: repository id must be github:owner/repo@<40-char-sha>"
        )
        return

    owner = match.group("owner")
    repo = match.group("repo")
    id_commit = match.group("commit")
    category = entry["category"]
    repo_slug = entry.get("repo_slug")
    revision = entry.get("revision")
    raw_paths = entry.get("raw_paths", [])
    derived_path = entry.get("derived_path")
    docs_paths = entry.get("docs_paths")

    if not isinstance(repo_slug, str) or not repo_slug:
        errors.append(f"{source_id}: repository entry missing repo_slug")
    if entry.get("slug") != f"{repo_slug}-codebase":
        errors.append(
            f"{source_id}: repository slug must be {repo_slug}-codebase"
        )
    if revision != id_commit:
        errors.append(f"{source_id}: revision does not match id commit")
    if "docs_path" in entry:
        errors.append(f"{source_id}: repository entries must use docs_paths")
    if entry.get("status") != "ingested":
        errors.append(f"{source_id}: completed repository revision must be ingested")
    if not isinstance(docs_paths, list) or not docs_paths:
        errors.append(f"{source_id}: repository entry requires non-empty docs_paths")
        docs_paths = []
    if len(raw_paths) != 1:
        errors.append(f"{source_id}: repository entry requires exactly one raw path")
        return

    raw_path = raw_paths[0]
    full_raw_path = root / raw_path
    if not full_raw_path.exists():
        return

    metadata = parse_front_matter(full_raw_path)
    expected_url = f"https://github.com/{owner}/{repo}"
    expected_metadata = {
        "kind": "repository-source",
        "repository_url": expected_url,
        "commit": id_commit,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            errors.append(
                f"{source_id}: raw metadata {key} must be {expected}, "
                f"got {metadata.get(key)}"
            )

    checkout = metadata.get("local_checkout")
    if not isinstance(checkout, str):
        errors.append(f"{source_id}: raw metadata missing local_checkout")
    else:
        checkout_path = Path(checkout)
        if (
            checkout_path.is_absolute()
            or ".." in checkout_path.parts
            or not checkout_path.parts
            or checkout_path.parts[0] != "external-repos"
        ):
            errors.append(
                f"{source_id}: local_checkout must be beneath external-repos/: "
                f"{checkout}"
            )

    if not metadata.get("ref"):
        errors.append(f"{source_id}: raw metadata missing ref")
    inspected = metadata.get("inspected")
    if not isinstance(inspected, str) or not valid_iso_date(inspected):
        errors.append(f"{source_id}: raw metadata inspected must be YYYY-MM-DD")
    checkout_state = metadata.get("checkout_state")
    if checkout_state not in {"clean", "dirty"}:
        errors.append(
            f"{source_id}: raw metadata checkout_state must be clean or dirty"
        )

    raw_text = full_raw_path.read_text(encoding="utf-8")
    for heading in ("## Reading Scope", "## Important Entry Files", "## Limitations"):
        if heading not in raw_text:
            errors.append(f"{source_id}: raw source record missing {heading}")

    repo_analysis_prefix = categories[category].get("repo_analysis_prefix")
    expected_derived_path = (
        f"{repo_analysis_prefix}{repo_slug}/{id_commit}/"
        if repo_analysis_prefix and repo_slug
        else None
    )
    if derived_path != expected_derived_path:
        errors.append(
            f"{source_id}: derived_path must be {expected_derived_path}, "
            f"got {derived_path}"
        )
        return

    manifest_repo_analysis_paths.add(normalized_path(derived_path))
    full_derived_path = root / derived_path
    important_files = full_derived_path / "important-files.md"
    if not full_derived_path.is_dir():
        errors.append(f"{source_id}: missing repo analysis directory: {derived_path}")
    if not important_files.is_file():
        errors.append(
            f"{source_id}: missing required repo analysis file: "
            f"{derived_path}important-files.md"
        )
    else:
        analysis_metadata = parse_front_matter(important_files)
        expected_analysis = {
            "kind": "repository-analysis",
            "repository_id": source_id,
            "commit": id_commit,
            "source_record": raw_path,
        }
        for key, expected in expected_analysis.items():
            if analysis_metadata.get(key) != expected:
                errors.append(
                    f"{source_id}: important-files metadata {key} must be "
                    f"{expected}, got {analysis_metadata.get(key)}"
                )
        generated = analysis_metadata.get("generated")
        if not isinstance(generated, str) or not valid_iso_date(generated):
            errors.append(
                f"{source_id}: important-files generated must be YYYY-MM-DD"
            )

    for docs_path in docs_paths:
        if (
            not isinstance(docs_path, str)
            or not is_safe_relative_path(docs_path, "docs")
        ):
            errors.append(f"{source_id}: invalid repository docs path: {docs_path}")
            continue
        full_docs_path = root / docs_path
        if not full_docs_path.is_file():
            errors.append(f"{source_id}: missing docs path: {docs_path}")
            continue
        doc_metadata = parse_front_matter(full_docs_path)
        doc_sources = front_matter_sources(full_docs_path)
        if raw_path not in doc_sources:
            errors.append(
                f"{source_id}: {docs_path} front matter missing source {raw_path}"
            )
        if not cites_file_beneath(root, doc_sources, derived_path):
            errors.append(
                f"{source_id}: {docs_path} front matter missing derived source "
                f"beneath {derived_path}"
            )
        if id_commit not in body_near_top(full_docs_path):
            errors.append(
                f"{source_id}: {docs_path} must state full commit near the top"
            )
        if checkout_state == "dirty" and doc_metadata.get("confidence") != "low":
            errors.append(
                f"{source_id}: dirty checkout requires confidence low in {docs_path}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Knowledge-base root (defaults to this script's repository).",
    )
    parser.add_argument("--manifest", default="sources.json")
    args = parser.parse_args()

    root = args.root.resolve()
    categories = load_json(root / "kb-categories.json")["categories"]
    manifest = load_json(root / args.manifest)["sources"]
    errors: list[str] = []
    seen_ids: set[str] = set()
    manifest_raw_paths: set[str] = set()
    manifest_derived_paths: set[str] = set()
    manifest_repo_analysis_paths: set[str] = set()
    repository_consumers: dict[str, tuple[str, set[str]]] = {}

    for entry in manifest:
        missing_fields = [
            field
            for field in ("id", "title", "slug", "category", "kind", "raw_paths")
            if field not in entry
        ]
        if missing_fields:
            errors.append(f"manifest entry missing fields: {', '.join(missing_fields)}")
            continue

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
            if raw_path in manifest_raw_paths:
                errors.append(f"{source_id}: raw path reused by another source: {raw_path}")
            manifest_raw_paths.add(raw_path)
            full_raw_path = root / raw_path
            if not raw_path.startswith(raw_prefix):
                errors.append(f"{source_id}: raw path outside category prefix: {raw_path}")
            if not full_raw_path.exists():
                errors.append(f"{source_id}: missing raw path: {raw_path}")
            expected_name = expected_raw_name(entry, raw_path)
            if expected_name and Path(raw_path).name != expected_name:
                errors.append(
                    f"{source_id}: raw filename should be {expected_name}, "
                    f"got {Path(raw_path).name}"
                )

        if entry.get("kind") == "repository":
            validate_repository_entry(
                root,
                entry,
                categories,
                errors,
                manifest_repo_analysis_paths,
            )
            docs_paths = entry.get("docs_paths", [])
            for raw_path in entry.get("raw_paths", []):
                if isinstance(docs_paths, list):
                    repository_consumers[raw_path] = (
                        source_id,
                        set(docs_paths),
                    )
            continue

        if "docs_paths" in entry:
            errors.append(f"{source_id}: non-repository entries must use docs_path")
        derived_path = entry.get("derived_path")
        if derived_path:
            manifest_derived_paths.add(derived_path)
            if not derived_prefix:
                errors.append(f"{source_id}: category has no derived prefix: {category}")
            elif not derived_path.startswith(derived_prefix):
                errors.append(
                    f"{source_id}: derived path outside category prefix: {derived_path}"
                )
            elif not (root / derived_path).exists():
                errors.append(f"{source_id}: missing derived path: {derived_path}")
            expected_derived_name = f"{entry['slug']}.md"
            if Path(derived_path).name != expected_derived_name:
                errors.append(
                    f"{source_id}: derived filename should be {expected_derived_name}, "
                    f"got {Path(derived_path).name}"
                )

        docs_path = entry.get("docs_path")
        if not docs_path:
            errors.append(f"{source_id}: non-repository entry missing docs_path")
            continue
        if docs_prefix and not docs_path.startswith(docs_prefix):
            errors.append(f"{source_id}: docs path outside category prefix: {docs_path}")
        full_docs_path = root / docs_path
        if not full_docs_path.exists():
            errors.append(f"{source_id}: missing docs path: {docs_path}")
            continue
        doc_sources = front_matter_sources(full_docs_path)
        for raw_path in entry.get("raw_paths", []):
            if raw_path.endswith((".pdf", ".md", ".mdx")) and raw_path not in doc_sources:
                errors.append(f"{source_id}: docs front matter missing source {raw_path}")
        if derived_path and derived_path not in doc_sources:
            errors.append(
                f"{source_id}: docs front matter missing derived source {derived_path}"
            )

    for raw_path in sorted(p for p in (root / "raw").rglob("*") if p.is_file()):
        rel = raw_path.relative_to(root).as_posix()
        if rel not in manifest_raw_paths and not rel.startswith("raw/codex/"):
            errors.append(f"raw file missing from sources.json: {rel}")

    pdf_markdown_root = root / "derived/pdf-markdown"
    if pdf_markdown_root.exists():
        for derived_file in sorted(pdf_markdown_root.rglob("*.md")):
            rel = derived_file.relative_to(root).as_posix()
            if rel not in manifest_derived_paths:
                errors.append(f"derived markdown missing from sources.json: {rel}")

    repo_analysis_root = root / "derived/repo-analysis"
    if repo_analysis_root.exists():
        for important_file in sorted(repo_analysis_root.rglob("important-files.md")):
            rel = important_file.parent.relative_to(root).as_posix()
            if rel not in manifest_repo_analysis_paths:
                errors.append(f"repository analysis missing from sources.json: {rel}/")
        for analysis_file in sorted(repo_analysis_root.rglob("*.md")):
            rel = analysis_file.relative_to(root).as_posix()
            if not any(
                rel == directory or rel.startswith(f"{directory}/")
                for directory in manifest_repo_analysis_paths
            ):
                errors.append(f"orphan repository analysis file: {rel}")

    docs_root = root / "docs"
    if docs_root.exists():
        for docs_file in sorted(docs_root.rglob("*.md")):
            docs_path = docs_file.relative_to(root).as_posix()
            for source in front_matter_sources(docs_file):
                consumer = repository_consumers.get(source)
                if consumer and docs_path not in consumer[1]:
                    errors.append(
                        f"{consumer[0]}: {docs_path} cites {source} but is missing "
                        "from docs_paths"
                    )

    if errors:
        for error in errors:
            print(f"kb integrity: {error}", file=sys.stderr)
        return 1

    print("kb integrity ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
