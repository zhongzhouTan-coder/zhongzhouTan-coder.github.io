#!/usr/bin/env python3
"""Scaffold or reuse an immutable repository source revision."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

from repository_remote import parse_repository_remote


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise SystemExit(message)


def run_git(checkout: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        fail(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        fail(f"cannot derive a repository slug from {value!r}")
    return slug


def parse_important_file(value: str) -> tuple[str, str]:
    path, separator, reason = value.partition("::")
    if not separator or not path.strip() or not reason.strip():
        fail("--important-file must use PATH::WHY format")
    file_path = Path(path.strip())
    if file_path.is_absolute() or ".." in file_path.parts:
        fail(f"important file must be repository-relative: {path}")
    return file_path.as_posix(), reason.strip()


def markdown_list(values: list[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


def format_manifest_entry(entry: dict[str, Any]) -> str:
    rendered = json.dumps(entry, indent=2, ensure_ascii=False)
    return rendered.replace("\n", "\n    ")


def update_manifest_text(
    text: str, entry: dict[str, Any]
) -> tuple[str, bool]:
    """Replace one source object or append it without reformatting the manifest."""
    sources_match = re.search(r'"sources"\s*:\s*\[', text)
    if not sources_match:
        fail("sources.json has no sources array")

    decoder = json.JSONDecoder()
    position = sources_match.end()
    last_end = position
    matched_span: tuple[int, int] | None = None
    existing_entry: dict[str, Any] | None = None
    entry_count = 0

    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position < len(text) and text[position] == "]":
            break
        if position < len(text) and text[position] == ",":
            position += 1
            continue
        start = position
        parsed, end = decoder.raw_decode(text, position)
        if not isinstance(parsed, dict):
            fail("sources array contains a non-object entry")
        entry_count += 1
        if parsed.get("id") == entry["id"]:
            matched_span = (start, end)
            existing_entry = parsed
        last_end = end
        position = end

    if existing_entry is not None and matched_span is not None:
        expected_immutable = (
            "title",
            "slug",
            "repo_slug",
            "revision",
            "category",
            "kind",
            "provider",
            "repository_url",
            "raw_paths",
            "derived_path",
        )
        for key in expected_immutable:
            if existing_entry.get(key) != entry.get(key):
                fail(
                    f"existing repository revision has conflicting {key}: "
                    f"{existing_entry.get(key)!r}"
                )
        merged = dict(existing_entry)
        merged_docs = list(existing_entry.get("docs_paths", []))
        for docs_path in entry["docs_paths"]:
            if docs_path not in merged_docs:
                merged_docs.append(docs_path)
        merged["docs_paths"] = merged_docs
        if merged == existing_entry:
            return text, False
        start, end = matched_span
        replacement = format_manifest_entry(merged)
        return text[:start] + replacement + text[end:], False

    rendered = format_manifest_entry(entry)
    separator = "," if entry_count else ""
    insertion = f"{separator}\n    {rendered}"
    return text[:last_end] + insertion + text[last_end:], True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create or reuse an immutable raw/derived source revision for a "
            "clean GitHub or GitCode checkout under external-repos/."
        )
    )
    parser.add_argument("checkout")
    parser.add_argument("--category", required=True)
    parser.add_argument("--docs-path", action="append", required=True)
    parser.add_argument("--scope", action="append", required=True)
    parser.add_argument(
        "--important-file",
        action="append",
        required=True,
        help="Repository-relative PATH::WHY; repeat as needed.",
    )
    parser.add_argument("--limitation", action="append", default=[])
    parser.add_argument("--title")
    parser.add_argument("--repo-slug")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    root = args.root.resolve()
    categories_path = root / "kb-categories.json"
    manifest_path = root / "sources.json"
    code_repositories_path = root / "docs/_data/code_repositories.json"
    categories = json.loads(categories_path.read_text(encoding="utf-8"))["categories"]
    if args.category not in categories:
        fail(f"unknown category: {args.category}")

    checkout_input = Path(args.checkout)
    checkout = (
        checkout_input.resolve()
        if checkout_input.is_absolute()
        else (root / checkout_input).resolve()
    )
    external_repos = (root / "external-repos").resolve()
    try:
        checkout.relative_to(external_repos)
    except ValueError:
        fail("checkout must live beneath external-repos/")
    if not checkout.is_dir():
        fail(f"checkout does not exist: {checkout}")
    git_root = Path(run_git(checkout, "rev-parse", "--show-toplevel")).resolve()
    if git_root != checkout:
        fail(f"checkout must name the repository root: {git_root}")

    checkout_rel = checkout.relative_to(root).as_posix()
    ignored = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "-q", "--", checkout_rel],
        check=False,
    )
    if ignored.returncode != 0:
        fail(f"checkout is not ignored by git: {checkout_rel}")

    remote = run_git(checkout, "remote", "get-url", "origin")
    try:
        repository_remote = parse_repository_remote(remote)
    except ValueError as exc:
        fail(str(exc))
    repo = repository_remote.repository_path.rsplit("/", 1)[-1]
    commit = run_git(checkout, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        fail(f"git returned a non-canonical commit SHA: {commit}")
    branch = run_git(checkout, "branch", "--show-current")
    ref = branch or run_git(
        checkout, "describe", "--tags", "--exact-match", check=False
    ) or "detached"
    dirty_output = run_git(checkout, "status", "--porcelain")
    checkout_state = "dirty" if dirty_output else "clean"
    if checkout_state == "dirty" and not args.allow_dirty:
        fail("checkout is dirty; clean it or pass --allow-dirty")

    short_sha = commit[:12]
    repository_id = repository_remote.source_id(commit)
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_data = json.loads(manifest_text)
    existing_entry = next(
        (
            source
            for source in manifest_data["sources"]
            if source.get("id") == repository_id
        ),
        None,
    )
    if checkout_state == "dirty" and existing_entry is not None:
        fail(
            "a dirty checkout cannot reuse an immutable commit revision; "
            "clean or commit the changes first"
        )
    repo_slug = slugify(
        args.repo_slug
        or (existing_entry or {}).get("repo_slug")
        or repo
    )
    title = (
        args.title
        or (existing_entry or {}).get("title")
        or f"{repo} Codebase"
    )
    source_slug = f"{repo_slug}-codebase"
    raw_path = (
        f"raw/{args.category}/{repo_slug}-codebase--"
        f"{repository_remote.provider}-{short_sha}.md"
    )
    derived_path = (
        f"derived/repo-analysis/{args.category}/{repo_slug}/{commit}/"
    )

    docs_paths: list[str] = []
    for value in args.docs_path:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not value.startswith("docs/"):
            fail(f"docs path must be repository-relative beneath docs/: {value}")
        normalized = path.as_posix()
        if normalized not in docs_paths:
            docs_paths.append(normalized)

    important_files = [parse_important_file(value) for value in args.important_file]
    inspected = date.today().isoformat()
    limitations = args.limitation or [
        "Static code reading only; runtime behavior was not executed."
    ]
    if checkout_state == "dirty":
        limitations.append(
            "The checkout contained uncommitted changes that are not "
            "reproducible from the pinned commit alone."
        )

    raw_content = f"""---
kind: repository-source
provider: {repository_remote.provider}
clone_url: {repository_remote.clone_url}
repository_url: {repository_remote.repository_url}
local_checkout: {checkout_rel}/
commit: {commit}
ref: {ref}
inspected: {inspected}
checkout_state: {checkout_state}
---

# {title} Source Record

## Reading Scope

{markdown_list(args.scope)}

## Important Entry Files

{markdown_list([f"`{path}` — {reason}" for path, reason in important_files])}

## Limitations

{markdown_list(limitations)}
"""

    analysis_content = f"""---
kind: repository-analysis
repository_id: {repository_id}
commit: {commit}
source_record: {raw_path}
generated: {inspected}
---

# {title} Important Files

## Evidence Map

{markdown_list([f"`{path}` — {reason}" for path, reason in important_files])}

## Reproduction Commands

Record exact read-only search or counting commands here when the docs make
quantitative codebase claims.
"""

    entry = {
        "id": repository_id,
        "title": title,
        "slug": source_slug,
        "repo_slug": repo_slug,
        "revision": commit,
        "category": args.category,
        "kind": "repository",
        "provider": repository_remote.provider,
        "repository_url": repository_remote.repository_url,
        "raw_paths": [raw_path],
        "derived_path": derived_path,
        "docs_paths": docs_paths,
        "status": (existing_entry or {}).get("status", "pending"),
    }

    code_repository_key = f"{repo_slug}-{short_sha}"
    code_repository = {
        "local_checkout": checkout_rel,
        "provider": repository_remote.provider,
        "repository_url": repository_remote.repository_url,
        "revision": commit,
    }
    code_repositories = (
        json.loads(code_repositories_path.read_text(encoding="utf-8"))
        if code_repositories_path.is_file()
        else {}
    )
    existing_code_repository = code_repositories.get(code_repository_key)
    if (
        existing_code_repository is not None
        and existing_code_repository != code_repository
    ):
        fail(
            f"code repository key {code_repository_key} has conflicting metadata"
        )
    for key, registered_repository in code_repositories.items():
        if (
            key != code_repository_key
            and registered_repository.get("local_checkout") == checkout_rel
            and registered_repository.get("revision") != commit
        ):
            fail(
                f"checkout {checkout_rel} is already registered at another revision; "
                "use a revision-specific checkout path"
            )
    code_repositories[code_repository_key] = code_repository

    updated_manifest, is_new = update_manifest_text(manifest_text, entry)
    raw_file = root / raw_path
    analysis_file = root / derived_path / "important-files.md"

    if args.dry_run:
        action = "create" if is_new else "reuse"
        print(f"action: {action}")
        print(f"raw: {raw_path}")
        print(f"derived: {derived_path}")
        print(f"code repository: {code_repository_key}")
        print(json.dumps(entry, indent=2, ensure_ascii=False))
        return 0

    if is_new:
        if raw_file.exists() or analysis_file.exists():
            fail("refusing to overwrite an unregistered immutable revision")
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        analysis_file.parent.mkdir(parents=True, exist_ok=True)
        raw_file.write_text(raw_content, encoding="utf-8")
        analysis_file.write_text(analysis_content, encoding="utf-8")
    elif not raw_file.is_file() or not analysis_file.is_file():
        fail("registered revision is missing its immutable raw or analysis file")

    manifest_path.write_text(updated_manifest, encoding="utf-8")
    code_repositories_path.parent.mkdir(parents=True, exist_ok=True)
    code_repositories_path.write_text(
        f"{json.dumps(code_repositories, indent=2, ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    print(f"{'created' if is_new else 'reused'} {repository_id}")
    print("Complete the docs citations, set status to ingested, then run integrity checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
