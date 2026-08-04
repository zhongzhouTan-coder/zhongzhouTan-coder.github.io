#!/usr/bin/env python3
"""Refresh repository evidence without duplicating full Git histories."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from repository_remote import parse_repository_remote


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIN_REVISION_INTERVAL_DAYS = 14
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


def load_registry(root: Path) -> dict[str, dict[str, str]]:
    registry = load_json(root / "docs/_data/code_repositories.json")
    if not isinstance(registry, dict):
        fail("code repository registry must be a JSON object")
    for key, entry in registry.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            fail("code repository registry contains an invalid entry")
    return registry


def registry_entry(root: Path, key: str) -> dict[str, str]:
    registry = load_registry(root)
    if key not in registry:
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


def selected_registry_entries(
    root: Path, keys: list[str]
) -> list[tuple[str, dict[str, str]]]:
    registry = load_registry(root)
    selected_keys = keys or sorted(registry)
    return [(key, registry_entry(root, key)) for key in selected_keys]


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


def source_entry(root: Path, entry: dict[str, str]) -> dict[str, Any]:
    manifest = load_json(root / "sources.json")
    for source in manifest.get("sources", []):
        if (
            source.get("kind") == "repository"
            and source.get("provider") == entry["provider"]
            and source.get("repository_url", "").rstrip("/")
            == entry["repository_url"].rstrip("/")
            and source.get("revision") == entry["revision"]
        ):
            return source
    fail(
        "cannot find the pinned repository revision in sources.json; "
        "repair its provenance or pass --force-new-revision"
    )


def revision_inspected_date(root: Path, entry: dict[str, str]) -> date:
    source = source_entry(root, entry)
    raw_paths = source.get("raw_paths")
    if not isinstance(raw_paths, list):
        fail(
            "repository source has no raw_paths; repair its provenance or "
            "pass --force-new-revision"
        )
    for raw_value in raw_paths:
        if not isinstance(raw_value, str) or not raw_value.endswith(".md"):
            continue
        raw_path = root / raw_value
        try:
            raw_text = raw_path.read_text(encoding="utf-8")
        except OSError as exc:
            fail(f"cannot read repository source record {raw_path}: {exc}")
        match = re.search(
            r"^inspected:\s*(\d{4}-\d{2}-\d{2})\s*$",
            raw_text,
            re.MULTILINE,
        )
        if not match:
            continue
        try:
            return date.fromisoformat(match.group(1))
        except ValueError:
            fail(
                "repository source record has an invalid inspected date: "
                f"{raw_path}"
            )
    fail(
        "repository source record has no inspected date; repair its provenance "
        "or pass --force-new-revision"
    )


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


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


def add_batch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "repository_keys",
        nargs="*",
        help="Registry keys to process; omit to process every registered revision.",
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
    sync.add_argument(
        "--min-revision-interval-days",
        type=non_negative_int,
        default=DEFAULT_MIN_REVISION_INTERVAL_DAYS,
        help=(
            "Minimum days between immutable evidence revisions (default: "
            f"{DEFAULT_MIN_REVISION_INTERVAL_DAYS}; use 0 to disable)."
        ),
    )
    sync.add_argument(
        "--force-new-revision",
        action="store_true",
        help="Bypass the revision interval when scoped implementation changed.",
    )

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

    status = subparsers.add_parser(
        "status",
        help="Report whether registered pinned worktrees are ready in this workspace.",
    )
    add_batch_arguments(status)
    status.add_argument("--json", action="store_true")

    materialize_all = subparsers.add_parser(
        "materialize-all",
        help="Materialize selected or all registered pinned worktrees.",
    )
    add_batch_arguments(materialize_all)
    materialize_all.add_argument(
        "--remote-url",
        help="Use a mirror for one explicitly selected repository key.",
    )
    return parser.parse_args()


def checkout_status(
    root: Path, key: str, entry: dict[str, str]
) -> dict[str, str]:
    target = checkout_path(root, entry["local_checkout"])
    cache = cache_path(root, entry)
    result = {
        "key": key,
        "status": "not-materialized",
        "revision": entry["revision"],
        "local_checkout": entry["local_checkout"],
        "repository_url": entry["repository_url"],
        "cache": "ready" if cache.is_dir() else "missing",
    }
    if not target.exists():
        return result
    if not target.is_dir():
        result["status"] = "invalid-path"
        return result

    revision_result = run_git(
        "-C", str(target), "rev-parse", "HEAD", check=False
    )
    if revision_result.returncode != 0:
        result["status"] = "not-a-git-checkout"
        return result
    actual_revision = revision_result.stdout.strip()
    if actual_revision != entry["revision"]:
        result["status"] = "wrong-revision"
        result["actual_revision"] = actual_revision
        return result

    common_result = run_git(
        "-C",
        str(target),
        "rev-parse",
        "--path-format=absolute",
        "--git-common-dir",
        check=False,
    )
    if (
        common_result.returncode != 0
        or Path(common_result.stdout.strip()).resolve() != cache.resolve()
    ):
        result["status"] = "unmanaged-checkout"
        return result

    dirty_result = run_git(
        "-C", str(target), "status", "--porcelain", check=False
    )
    if dirty_result.returncode != 0:
        result["status"] = "invalid-checkout"
    elif dirty_result.stdout.strip():
        result["status"] = "dirty"
    else:
        result["status"] = "ready"
    return result


def command_status(args: argparse.Namespace, root: Path) -> None:
    rows = [
        checkout_status(root, key, entry)
        for key, entry in selected_registry_entries(root, args.repository_keys)
    ]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return

    headers = ("KEY", "STATUS", "CACHE", "REVISION", "CHECKOUT")
    values = [
        (
            row["key"],
            row["status"],
            row["cache"],
            row["revision"][:12],
            row["local_checkout"],
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    for row in values:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def command_materialize_all(args: argparse.Namespace, root: Path) -> None:
    entries = selected_registry_entries(root, args.repository_keys)
    if args.remote_url and len(entries) != 1:
        fail("--remote-url requires exactly one repository key")
    for index, (key, entry) in enumerate(entries):
        if index:
            print()
        print(f"repository: {key}")
        command_materialize(
            argparse.Namespace(remote_url=args.remote_url, sparse=[]),
            root,
            entry,
        )


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

    if not args.force_new_revision and args.min_revision_interval_days:
        inspected = revision_inspected_date(root, entry)
        today = date.today()
        if inspected > today:
            fail(f"pinned revision inspection date {inspected.isoformat()} is in the future")
        eligible_on = inspected + timedelta(days=args.min_revision_interval_days)
        print(f"last snapshot: {inspected.isoformat()}")
        print(f"minimum revision interval: {args.min_revision_interval_days} days")
        if today < eligible_on:
            print("decision: defer")
            print(f"eligible on: {eligible_on.isoformat()}")
            print("override: pass --force-new-revision for an intentional urgent refresh")
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
    if args.command == "status":
        command_status(args, root)
        return 0
    if args.command == "materialize-all":
        command_materialize_all(args, root)
        return 0

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
