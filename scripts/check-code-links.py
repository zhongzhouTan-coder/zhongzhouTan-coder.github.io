#!/usr/bin/env python3
"""Validate revision-aware Jekyll code-link includes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from repository_remote import parse_repository_remote


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "docs" / "_data" / "code_repositories.json"
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
CODE_LINK_ELEMENT_RE = re.compile(
    r"<a\b[^>]*\bclass=[\"'][^\"']*\bcode-link\b[^\"']*[\"'][^>]*>"
    r".*?</a>",
    re.DOTALL | re.IGNORECASE,
)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
REPOSITORY_FILE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".h",
    ".hpp",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
EVIDENCE_TABLE_HEADER = (
    "docs page",
    "finding",
    "file",
    "symbol",
    "start",
    "end",
)


class CodeEvidence(NamedTuple):
    source_path: Path
    source_line: int
    docs_path: PurePosixPath
    finding: str
    code_path: PurePosixPath
    symbol: str
    start_line: int
    end_line: int | None


def mask_non_newlines(value: str) -> str:
    """Hide Markdown content while preserving offsets and line numbers."""
    return re.sub(r"[^\r\n]", " ", value)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_table_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    return [cell.strip().strip("`") for cell in stripped[1:-1].split("|")]


def parse_code_evidence(markdown_path: Path) -> tuple[list[CodeEvidence], list[str]]:
    """Read machine-checkable code evidence tables from one derived note."""
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    declarations: list[CodeEvidence] = []
    errors: list[str] = []
    index = 0
    while index < len(lines):
        cells = parse_table_row(lines[index])
        normalized = tuple(cell.lower() for cell in cells) if cells else ()
        if normalized != EVIDENCE_TABLE_HEADER:
            index += 1
            continue

        index += 2  # Skip the Markdown separator row.
        while index < len(lines):
            cells = parse_table_row(lines[index])
            if cells is None:
                break
            source_line = index + 1
            location = f"{display_path(markdown_path)}:{source_line}"
            row_errors: list[str] = []
            if len(cells) != len(EVIDENCE_TABLE_HEADER):
                errors.append(
                    f"{location}: code evidence row must have "
                    f"{len(EVIDENCE_TABLE_HEADER)} columns"
                )
                index += 1
                continue

            docs_value, finding, code_value, symbol, start_value, end_value = cells
            docs_path = PurePosixPath(docs_value)
            code_path = PurePosixPath(code_value)
            if (
                docs_path.is_absolute()
                or ".." in docs_path.parts
                or not docs_path.parts
                or docs_path.parts[0] != "docs"
                or docs_path.suffix != ".md"
            ):
                row_errors.append(
                    f"{location}: evidence docs page must be a repository-relative "
                    "Markdown path beneath docs/"
                )
            if code_path.is_absolute() or ".." in code_path.parts or not code_path.parts:
                row_errors.append(
                    f"{location}: evidence file must be repository-relative"
                )
            if not finding:
                row_errors.append(f"{location}: evidence finding cannot be empty")
            if not symbol:
                row_errors.append(f"{location}: evidence symbol cannot be empty")
            if not start_value.isdigit() or int(start_value) < 1:
                row_errors.append(
                    f"{location}: evidence start must be a positive integer"
                )
            normalized_end = "" if end_value in {"", "-", "—"} else end_value
            if normalized_end and (
                not normalized_end.isdigit() or int(normalized_end) < 1
            ):
                row_errors.append(
                    f"{location}: evidence end must be blank or a positive integer"
                )
            if (
                start_value.isdigit()
                and normalized_end.isdigit()
                and int(normalized_end) < int(start_value)
            ):
                row_errors.append(f"{location}: evidence end cannot precede start")
            if row_errors:
                errors.extend(row_errors)
                index += 1
                continue

            declarations.append(
                CodeEvidence(
                    source_path=markdown_path,
                    source_line=source_line,
                    docs_path=docs_path,
                    finding=finding,
                    code_path=code_path,
                    symbol=symbol,
                    start_line=int(start_value),
                    end_line=int(normalized_end) if normalized_end else None,
                )
            )
            index += 1
        continue
    return declarations, errors


def find_code_evidence() -> tuple[list[CodeEvidence], list[str]]:
    declarations: list[CodeEvidence] = []
    errors: list[str] = []
    analysis_root = ROOT / "derived" / "repo-analysis"
    for markdown_path in sorted(analysis_root.rglob("*.md")):
        found, parse_errors = parse_code_evidence(markdown_path)
        declarations.extend(found)
        errors.extend(parse_errors)
    return declarations, errors


class CodeLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []

    def handle_starttag(
        self, tag: str, attributes: list[tuple[str, str | None]]
    ) -> None:
        if tag != "a":
            return
        values = {key: value or "" for key, value in attributes}
        if "code-link" in values.get("class", "").split():
            self.links.append(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        action="store_true",
        help="also verify local checkout revisions, files, and line numbers",
    )
    return parser.parse_args()


def load_registry() -> dict[str, dict[str, str]]:
    try:
        value = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {REGISTRY_PATH.relative_to(ROOT)}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("code repository registry must be a JSON object")
    return value


def repository_sources() -> set[tuple[str, str, str]]:
    manifest = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    result: set[tuple[str, str]] = set()
    for source in manifest["sources"]:
        if source.get("kind") != "repository":
            continue
        result.add(
            (
                source.get("provider", ""),
                source.get("repository_url", ""),
                source.get("revision", ""),
            )
        )
    return result


def remove_fenced_blocks(content: str) -> str:
    output: list[str] = []
    fence_character = ""
    fence_length = 0
    for line in content.splitlines(keepends=True):
        match = FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if not fence_character:
                fence_character = marker[0]
                fence_length = len(marker)
                output.append(mask_non_newlines(line))
                continue
            if marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = ""
                fence_length = 0
                output.append(mask_non_newlines(line))
                continue
        if not fence_character:
            output.append(line)
        else:
            output.append(mask_non_newlines(line))
    return "".join(output)


def find_code_links() -> list[tuple[Path, dict[str, str]]]:
    links: list[tuple[Path, dict[str, str]]] = []
    for markdown_path in sorted((ROOT / "docs").rglob("*.md")):
        if "_site" in markdown_path.parts:
            continue
        content = remove_fenced_blocks(markdown_path.read_text(encoding="utf-8"))
        parser = CodeLinkParser()
        parser.feed(content)
        links.extend((markdown_path, attributes) for attributes in parser.links)
    return links


def find_strict_evidence_pages() -> list[Path]:
    pages: list[Path] = []
    for markdown_path in sorted((ROOT / "docs").rglob("*.md")):
        if "_site" in markdown_path.parts:
            continue
        content = markdown_path.read_text(encoding="utf-8")
        if re.search(r"^code_evidence:\s*strict\s*$", content, re.MULTILINE):
            pages.append(markdown_path)
    return pages


def find_unlinked_repository_paths(markdown_path: Path) -> list[tuple[int, str]]:
    content = markdown_path.read_text(encoding="utf-8")
    if not re.search(r"^code_links:\s*strict\s*$", content, re.MULTILINE):
        return []
    content = remove_fenced_blocks(content)
    content = CODE_LINK_ELEMENT_RE.sub(
        lambda match: mask_non_newlines(match.group(0)), content
    )
    findings: list[tuple[int, str]] = []
    for match in INLINE_CODE_RE.finditer(content):
        value = match.group(1).strip()
        if not value or any(character.isspace() for character in value):
            continue
        suffix = PurePosixPath(value).suffix.lower()
        if suffix not in REPOSITORY_FILE_SUFFIXES:
            continue
        line = content.count("\n", 0, match.start()) + 1
        findings.append((line, value))
    return findings


def validate_registry(
    registry: dict[str, dict[str, str]], sources: set[tuple[str, str, str]]
) -> list[str]:
    errors: list[str] = []
    required = {"local_checkout", "provider", "repository_url", "revision"}
    checkout_revisions: dict[str, str] = {}
    for key, repository in registry.items():
        if not isinstance(repository, dict):
            errors.append(f"registry entry {key!r} must be an object")
            continue
        missing = sorted(required - repository.keys())
        if missing:
            errors.append(f"registry entry {key!r} is missing: {', '.join(missing)}")
            continue
        revision = repository["revision"]
        checkout = PurePosixPath(repository["local_checkout"])
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            errors.append(f"registry entry {key!r} has an invalid full revision")
        if checkout.is_absolute() or not checkout.parts or checkout.parts[0] != "external-repos":
            errors.append(f"registry entry {key!r} must use a checkout beneath external-repos/")
        if ".." in checkout.parts:
            errors.append(f"registry entry {key!r} local checkout cannot contain '..'")
        checkout_name = repository["local_checkout"]
        previous_revision = checkout_revisions.setdefault(checkout_name, revision)
        if previous_revision != revision:
            errors.append(
                f"registry entry {key!r} reuses {checkout_name!r} for a different revision"
            )
        identity = (
            repository["provider"],
            repository["repository_url"].rstrip("/"),
            revision,
        )
        if identity not in sources:
            errors.append(f"registry entry {key!r} has no matching repository source")
    return errors


def validate_include(
    markdown_path: Path,
    attributes: dict[str, str],
    registry: dict[str, dict[str, str]],
) -> list[str]:
    location = markdown_path.relative_to(ROOT)
    errors: list[str] = []
    required = ("href", "data-code-repo", "data-code-path", "data-code-line")
    missing = [name for name in required if not attributes.get(name)]
    if missing:
        return [f"{location}: code link is missing: {', '.join(missing)}"]

    repository_key = attributes["data-code-repo"]
    if repository_key not in registry:
        errors.append(f"{location}: unknown code repository {repository_key!r}")
        return errors

    code_path = PurePosixPath(attributes["data-code-path"])
    if code_path.is_absolute() or ".." in code_path.parts:
        errors.append(f"{location}: code path must be repository-relative")

    for field in ("data-code-line", "data-code-end-line"):
        value = attributes.get(field)
        if value is not None and (not value.isdigit() or int(value) < 1):
            errors.append(f"{location}: {field} must be a positive integer")
    line_value = attributes.get("data-code-line", "")
    end_line_value = attributes.get("data-code-end-line", "")
    if line_value.isdigit() and end_line_value.isdigit():
        if int(end_line_value) < int(line_value):
            errors.append(f"{location}: data-code-end-line cannot precede data-code-line")

    local_href, _, fragment = attributes["href"].partition("#")
    local_path = PurePosixPath(local_href)
    if local_path.is_absolute() or ".." not in local_path.parts:
        errors.append(f"{location}: href must be a relative path to external-repos/")
    expected_path = ROOT / registry[repository_key]["local_checkout"] / code_path
    resolved_path = (markdown_path.parent / local_path).resolve()
    if resolved_path != expected_path.resolve():
        errors.append(f"{location}: href does not match the registered local code path")
    if line_value.isdigit() and fragment != f"L{line_value}":
        errors.append(f"{location}: href fragment must be #L{line_value}")
    return errors


def validate_local(
    links: list[tuple[Path, dict[str, str]]],
    registry: dict[str, dict[str, str]],
) -> list[str]:
    errors: list[str] = []
    checked_revisions: dict[str, str | None] = {}
    checked_remotes: dict[str, tuple[str, str] | None] = {}
    for markdown_path, attributes in links:
        repository = registry.get(attributes.get("data-code-repo", ""))
        code_path = PurePosixPath(attributes.get("data-code-path", ""))
        if (
            not repository
            or not code_path.parts
            or code_path.is_absolute()
            or ".." in code_path.parts
            or not attributes.get("data-code-line", "").isdigit()
        ):
            continue
        location = markdown_path.relative_to(ROOT)
        checkout_name = repository["local_checkout"]
        checkout = ROOT / checkout_name
        if checkout_name not in checked_revisions:
            if not checkout.is_dir():
                checked_revisions[checkout_name] = None
                errors.append(f"{location}: local checkout is missing: {checkout_name}")
            else:
                result = subprocess.run(
                    ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                checked_revisions[checkout_name] = result.stdout.strip() if result.returncode == 0 else None
                remote_result = subprocess.run(
                    ["git", "-C", str(checkout), "remote", "get-url", "origin"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                try:
                    remote = parse_repository_remote(remote_result.stdout.strip())
                except ValueError:
                    checked_remotes[checkout_name] = None
                else:
                    checked_remotes[checkout_name] = (
                        remote.provider,
                        remote.repository_url,
                    )
        actual_revision = checked_revisions[checkout_name]
        if actual_revision != repository["revision"]:
            errors.append(
                f"{location}: {checkout_name} is at {actual_revision or 'an unknown revision'}, "
                f"expected {repository['revision']}"
            )
            continue
        expected_remote = (repository["provider"], repository["repository_url"])
        if checked_remotes.get(checkout_name) != expected_remote:
            errors.append(
                f"{location}: {checkout_name} origin does not match "
                f"{repository['repository_url']}"
            )
            continue
        source_path = checkout / code_path
        if not source_path.is_file():
            errors.append(f"{location}: local code file is missing: {source_path.relative_to(ROOT)}")
            continue
        line_count = sum(1 for _ in source_path.open(encoding="utf-8", errors="replace"))
        final_line = int(
            attributes.get("data-code-end-line", attributes["data-code-line"])
        )
        if final_line > line_count:
            errors.append(
                f"{location}: requested line {final_line} exceeds "
                f"{source_path.relative_to(ROOT)} ({line_count} lines)"
            )
    return errors


def validate_evidence_coverage(
    declarations: list[CodeEvidence],
    links: list[tuple[Path, dict[str, str]]],
    root: Path = ROOT,
) -> list[str]:
    """Require every declared finding to appear as a matching docs code link."""
    errors: list[str] = []
    links_by_page: dict[Path, list[dict[str, str]]] = {}
    for markdown_path, attributes in links:
        links_by_page.setdefault(markdown_path.resolve(), []).append(attributes)

    for evidence in declarations:
        location = f"{display_path(evidence.source_path)}:{evidence.source_line}"
        docs_path = root / evidence.docs_path
        if not docs_path.is_file():
            errors.append(
                f"{location}: evidence docs page does not exist: {evidence.docs_path}"
            )
            continue

        candidates = links_by_page.get(docs_path.resolve(), [])
        matched = False
        for attributes in candidates:
            if attributes.get("data-code-path") != str(evidence.code_path):
                continue
            if attributes.get("data-code-line") != str(evidence.start_line):
                continue
            if evidence.end_line is not None and attributes.get(
                "data-code-end-line"
            ) != str(evidence.end_line):
                continue
            matched = True
            break
        if matched:
            continue

        line_range = f"L{evidence.start_line}"
        if evidence.end_line is not None:
            line_range += f"-L{evidence.end_line}"
        errors.append(
            f"{location}: {evidence.docs_path} is missing declared code evidence "
            f"{evidence.finding!r} ({evidence.code_path}#{line_range}, "
            f"symbol {evidence.symbol!r})"
        )
    return errors


def validate_required_evidence_pages(
    required_pages: list[Path], declarations: list[CodeEvidence]
) -> list[str]:
    declared_pages = {(ROOT / evidence.docs_path).resolve() for evidence in declarations}
    return [
        f"{display_path(page)}: code_evidence: strict requires at least one "
        "Required Code Evidence row in a derived repository-analysis note"
        for page in required_pages
        if page.resolve() not in declared_pages
    ]


def main() -> int:
    args = parse_args()
    try:
        registry = load_registry()
        sources = repository_sources()
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"code link validation failed: {exc}", file=sys.stderr)
        return 1

    links = find_code_links()
    evidence, evidence_errors = find_code_evidence()
    strict_evidence_pages = find_strict_evidence_pages()
    errors = validate_registry(registry, sources)
    errors.extend(evidence_errors)
    errors.extend(validate_required_evidence_pages(strict_evidence_pages, evidence))
    for markdown_path, attributes in links:
        errors.extend(validate_include(markdown_path, attributes, registry))
    errors.extend(validate_evidence_coverage(evidence, links))
    if args.local:
        errors.extend(validate_local(links, registry))
    for markdown_path in sorted((ROOT / "docs").rglob("*.md")):
        if "_site" in markdown_path.parts:
            continue
        for line, value in find_unlinked_repository_paths(markdown_path):
            errors.append(
                f"{markdown_path.relative_to(ROOT)}:{line}: repository path "
                f"{value!r} must use a validated code-link anchor"
            )

    if errors:
        for error in errors:
            print(f"code link error: {error}", file=sys.stderr)
        return 1
    print(
        f"code links valid: {len(links)} link(s), "
        f"{len(evidence)} evidence declaration(s), "
        f"{len(registry)} repository revision(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
