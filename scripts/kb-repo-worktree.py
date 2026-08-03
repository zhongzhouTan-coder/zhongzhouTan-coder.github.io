#!/usr/bin/env python3
"""Refresh repository evidence without duplicating full Git histories."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from repository_remote import parse_repository_remote


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
REVISION_RE = re.compile(r"[0-9a-f]{40}")


def fail(message: str) -> None:
    raise SystemExit(message)


def run_git(
    *args: str,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        fail(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path}: {exc}")


def registry_entry(root: Path, key: str) -> dict[str, str]:
    registry = load_json(root / "docs/_data/code_repositories.json")
    if not isinstance(registry, dict) or key not in registry:
        fail(f"unknown code repository key: {key}")
    entry = registry[key]
    if not isinstance(entry, dict):
        fail(f"invalid code repository entry: {key}")
    required = {"local_checkout", "provider", "repository_url", "revision"}
    missing = sorted(required - entry.keys())
    if missing:
        fail(f"code repository {key} is missing: {', '.join(missing)}")
    revision = entry["revision"]
    if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
        fail(f"code repository {key} has an invalid revision")
    return entry


def repo_slug(root: Path, entry: dict[str, str]) -> str:
    manifest = load_json(root / "sources.json")
    for source in manifest.get("sources", []):
        if (
            source.get("kind") == "repository"
            and source.get("provider") == entry["provider"]
            and source.get("repository_url", "").rstrip("/")
            == entry["repository_url"].rstrip("/")
            and source.get("revision") == entry["revision"]
        ):
            value = source.get("repo_slug")
            if isinstance(value, str) and value:
                return value
    repository = parse_repository_remote(entry["repository_url"])
    return repository.repository_path.rsplit("/", 1)[-1]


def cache_path(root: Path, entry: dict[str, str]) -> Path:
    repository = parse_repository_remote(entry["repository_url"])
    relative = PurePosixPath(repository.repository_path)
    if relative.is_absolute() or ".." in relative.parts:
        fail("repository URL produced an unsafe cache path")
    return (
        root
        / "external-repos/.cache"
        / repository.provider
        / relative.parent
        / f"{relative.name}.git"
    )


def checkout_path(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) < 2
        or relative.parts[0] != "external-repos"
        or relative.parts[1] == ".cache"
    ):
        fail(f"checkout must be beneath external-repos/: {value}")
    return root.joinpath(*relative.parts)


def ensure_cache(
    root: Path,
    entry: dict[str, str],
    remote_url: str | None,
) -> Path:
    cache = cache_path(root, entry)
    source_url = remote_url or entry["repository_url"]
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        run_git(
            "clone",
            "--bare",
            "--filter=blob:none",
            "--origin",
            "origin",
            source_url,
            str(cache),
        )
        run_git(
            "--git-dir",
            str(cache),
            "config",
            "remote.origin.fetch",
            "+refs/heads/*:refs/remotes/origin/*",
        )
    elif run_git(
        "--git-dir", str(cache), "rev-parse", "--is-bare-repository"
    ).stdout.strip() != "true":
        fail(f"shared cache is not a bare repository: {cache}")

    if remote_url:
        run_git(
            "--git-dir", str(cache), "remote", "set-url", "origin", remote_url
        )
    else:
        actual_url = run_git(
            "--git-dir", str(cache), "remote", "get-url", "origin"
        ).stdout.strip()
        try:
            actual = parse_repository_remote(actual_url)
            expected = parse_repository_remote(entry["repository_url"])
        except ValueError as exc:
            fail(f"cannot validate shared cache origin: {exc}")
        if (
            actual.provider != expected.provider
            or actual.repository_url != expected.repository_url
        ):
            fail(f"shared cache origin does not match {entry['repository_url']}")
    return cache


def has_commit(cache: Path, revision: str) -> bool:
    result = run_git(
        "--git-dir",
        str(cache),
        "cat-file",
        "-e",
        f"{revision}^{{commit}}",
        check=False,
    )
    return result.returncode == 0


def ensure_commit(cache: Path, revision: str) -> None:
    if has_commit(cache, revision):
        return
    result = run_git(
        "--git-dir",
        str(cache),
        "fetch",
        "origin",
        revision,
        check=False,
    )
    if result.returncode != 0 or not has_commit(cache, revision):
        fail(
            f"shared cache does not contain revision {revision}, and origin "
            "could not supply it"
        )


def protect_revision(cache: Path, revision: str) -> None:
    ensure_commit(cache, revision)
    run_git(
        "--git-dir",
        str(cache),
        "update-ref",
        f"refs/kb/revisions/{revision}",
        revision,
    )


def fetch(cache: Path) -> None:
    run_git(
        "--git-dir",
        str(cache),
        "fetch",
        "--prune",
        "origin",
        "+refs/heads/*:refs/remotes/origin/*",
    )


def resolve_tip(cache: Path, remote_ref: str | None) -> tuple[str, str]:
    candidates = (
        [remote_ref]
        if remote_ref
        else [
            "refs/remotes/origin/HEAD",
            "refs/remotes/origin/main",
            "refs/remotes/origin/master",
        ]
    )
    for candidate in candidates:
        if not candidate:
            continue
        result = run_git(
            "--git-dir",
            str(cache),
            "rev-parse",
            "--verify",
            f"{candidate}^{{commit}}",
            check=False,
        )
        revision = result.stdout.strip()
        if result.returncode == 0 and REVISION_RE.fullmatch(revision):
            return candidate, revision
    fail("cannot resolve upstream tip; pass --remote-ref explicitly")


def validate_materialized_checkout(
    target: Path,
    cache: Path,
    revision: str,
) -> None:
    actual = run_git("-C", str(target), "rev-parse", "HEAD").stdout.strip()
    if actual != revision:
        fail(f"{target} is at {actual}, expected {revision}")
    common_value = run_git(
        "-C", str(target), "rev-parse", "--path-format=absolute", "--git-common-dir"
    ).stdout.strip()
    if Path(common_value).resolve() != cache.resolve():
        fail(f"{target} is not backed by shared cache {cache}")
    dirty = run_git("-C", str(target), "status", "--porcelain").stdout.strip()
    if dirty:
        fail(f"worktree is dirty: {target}")


def materialize(
    cache: Path,
    target: Path,
    revision: str,
    sparse_paths: list[str],
) -> bool:
    protect_revision(cache, revision)
    if target.exists():
        validate_materialized_checkout(target, cache, revision)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    run_git(
        "--git-dir",
        str(cache),
        "worktree",
        "add",
        "--detach",
        str(target),
        revision,
    )
    if sparse_paths:
        run_git("-C", str(target), "sparse-checkout", "init", "--no-cone")
        run_git(
            "-C",
            str(target),
            "sparse-checkout",
            "set",
            "--no-cone",
            "--",
            *sparse_paths,
        )
    return True


def relevant_changes(
    cache: Path,
    base: str,
    tip: str,
    paths: list[str],
) -> bool:
    command = ["--git-dir", str(cache), "diff", "--quiet", base, tip, "--"]
    command.extend(paths)
    result = run_git(*command, check=False)
    if result.returncode not in {0, 1}:
        fail(result.stderr.strip() or "git diff failed")
    return result.returncode == 1


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("repository_key")
    parser.add_argument(
        "--remote-url",
        help="Fetch from a mirror or local test remote instead of the canonical URL.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help=argparse.SUPPRESS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser(
        "sync",
        help="Fetch upstream and materialize a new revision only when scope changed.",
    )
    add_common_arguments(sync)
    sync.add_argument("--path", action="append", default=[])
    sync.add_argument("--remote-ref")
    sync.add_argument("--no-fetch", action="store_true")
    sync.add_argument("--target")
    sync.add_argument("--sparse", action="append", default=[])

    restore = subparsers.add_parser(
        "materialize",
        help="Restore the pinned worktree recorded by a registry entry.",
    )
    add_common_arguments(restore)
    restore.add_argument("--sparse", action="append", default=[])

    retire = subparsers.add_parser(
        "retire",
        help="Remove a clean shared worktree while preserving its pinned commit.",
    )
    add_common_arguments(retire)
    return parser.parse_args()


def command_sync(args: argparse.Namespace, root: Path, entry: dict[str, str]) -> None:
    cache = ensure_cache(root, entry, args.remote_url)
    if not args.no_fetch:
        fetch(cache)
    base = entry["revision"]
    ensure_commit(cache, base)
    protect_revision(cache, base)
    remote_ref, tip = resolve_tip(cache, args.remote_ref)
    changed = relevant_changes(cache, base, tip, args.path)

    print(f"cache: {cache.relative_to(root)}")
    print(f"pinned: {base}")
    print(f"upstream: {tip} ({remote_ref})")
    print(f"scope: {', '.join(args.path) if args.path else '<entire repository>'}")
    if not changed:
        print("decision: reuse")
        return

    target_value = args.target or f"external-repos/{repo_slug(root, entry)}-{tip[:12]}"
    target = checkout_path(root, target_value)
    created = materialize(cache, target, tip, args.sparse)
    print("decision: new revision")
    print(f"checkout: {target.relative_to(root)}")
    print(f"worktree: {'created' if created else 'reused'}")
    print("next: run scripts/kb-init-repo-source.py for this checkout")


def command_materialize(
    args: argparse.Namespace,
    root: Path,
    entry: dict[str, str],
) -> None:
    cache = ensure_cache(root, entry, args.remote_url)
    if not has_commit(cache, entry["revision"]):
        fetch(cache)
    ensure_commit(cache, entry["revision"])
    target = checkout_path(root, entry["local_checkout"])
    created = materialize(cache, target, entry["revision"], args.sparse)
    print(f"checkout: {target.relative_to(root)}")
    print(f"worktree: {'created' if created else 'already materialized'}")


def command_retire(
    args: argparse.Namespace,
    root: Path,
    entry: dict[str, str],
) -> None:
    cache = ensure_cache(root, entry, args.remote_url)
    target = checkout_path(root, entry["local_checkout"])
    if not target.exists():
        print(f"checkout already retired: {target.relative_to(root)}")
        return
    validate_materialized_checkout(target, cache, entry["revision"])
    protect_revision(cache, entry["revision"])
    run_git("--git-dir", str(cache), "worktree", "remove", str(target))
    print(f"retired: {target.relative_to(root)}")
    print(f"protected: refs/kb/revisions/{entry['revision']}")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    entry = registry_entry(root, args.repository_key)
    if args.command == "sync":
        command_sync(args, root, entry)
    elif args.command == "materialize":
        command_materialize(args, root, entry)
    else:
        command_retire(args, root, entry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
