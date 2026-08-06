#!/usr/bin/env python3
"""Validate glossary metadata, backlinks, and unlinked term mentions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(SCRIPTS_ROOT))

from common.paths import find_repository_root  # noqa: E402


DEFAULT_ROOT = find_repository_root(__file__)
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^]]+)]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
IMAGE_RE = re.compile(r"!\[[^]]*]\([^)]+\)")
HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Issue:
    severity: str
    kind: str
    path: str
    line: int | None
    message: str
    term: str | None = None


@dataclass(frozen=True)
class Term:
    path: Path
    relative_path: str
    title: str
    aliases: tuple[str, ...]
    appears_in: frozenset[str]

    @property
    def names(self) -> tuple[str, ...]:
        return (self.title, *self.aliases)


def parse_front_matter(path: Path) -> tuple[dict[str, str | list[str]], int]:
    """Parse the scalar/list YAML subset used by glossary front matter."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, 0

    result: dict[str, str | list[str]] = {}
    list_key: str | None = None
    for index, line in enumerate(lines[1:], start=2):
        if line.strip() == "---":
            return result, index
        list_item = re.match(r"^\s+-\s+(.+?)\s*$", line)
        if list_key and list_item:
            value = list_item.group(1).strip().strip("\"'")
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
    return result, 0


def string_list(metadata: dict[str, str | list[str]], key: str) -> list[str]:
    value = metadata.get(key, [])
    return list(value) if isinstance(value, list) else []


def normalize_name(name: str) -> str:
    return " ".join(name.casefold().split())


def repo_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def eligible_lines(path: Path) -> list[tuple[int, str]]:
    """Return prose lines eligible for term mention/link validation."""
    lines = path.read_text(encoding="utf-8").splitlines()
    _, front_matter_end = parse_front_matter(path)
    in_fence = False
    fence_marker = ""
    previous_was_image = False
    result: list[tuple[int, str]] = []

    for line_number, line in enumerate(lines, start=1):
        if front_matter_end and line_number <= front_matter_end:
            continue
        stripped = line.lstrip()
        fence = re.match(r"^(```+|~~~+)", stripped)
        if fence:
            marker = fence.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue
        if in_fence or re.match(r"^\s{0,3}#{1,6}(?:\s|$)", line):
            continue
        if previous_was_image and re.match(r"^\s*[*_]", line):
            previous_was_image = False
            continue
        previous_was_image = bool(IMAGE_RE.search(line))
        scrubbed = IMAGE_RE.sub("", line)
        scrubbed = INLINE_CODE_RE.sub("", scrubbed)
        scrubbed = HTML_TAG_RE.sub("", scrubbed)
        result.append((line_number, scrubbed))
    return result


def markdown_links(path: Path) -> list[tuple[int, str, str]]:
    links: list[tuple[int, str, str]] = []
    for line_number, line in eligible_lines(path):
        for match in MARKDOWN_LINK_RE.finditer(line):
            links.append((line_number, match.group(1), match.group(2)))
    return links


def resolve_link(root: Path, source: Path, target: str) -> Path | None:
    target = target.split("#", 1)[0].split("?", 1)[0].strip()
    if not target or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
        return None
    if target.startswith("/"):
        candidate = root / target.lstrip("/")
    else:
        candidate = source.parent / target
    return candidate.resolve()


def where_it_appears_links(path: Path) -> list[tuple[int, str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    in_section = False
    links: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(lines, start=1):
        if re.match(r"^##\s+Where It Appears\s*$", line, re.IGNORECASE):
            in_section = True
            continue
        if in_section and re.match(r"^##\s+", line):
            break
        if in_section:
            for match in MARKDOWN_LINK_RE.finditer(line):
                links.append((line_number, match.group(1), match.group(2)))
    return links


def name_pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w]){re.escape(name)}(?![\w])", re.IGNORECASE)


def first_mention(path: Path, term: Term) -> tuple[int, str] | None:
    patterns = [(name, name_pattern(name)) for name in term.names if name]
    for line_number, line in eligible_lines(path):
        for name, pattern in patterns:
            if pattern.search(line):
                return line_number, name
    return None


def load_terms(root: Path, issues: list[Issue]) -> list[Term]:
    terms_dir = root / "docs" / "terms"
    terms: list[Term] = []
    for path in sorted(terms_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        metadata, _ = parse_front_matter(path)
        relative = repo_path(root, path)
        for field in ("title", "summary", "category"):
            if not isinstance(metadata.get(field), str) or not metadata[field]:
                issues.append(
                    Issue(
                        "error",
                        "missing-field",
                        relative,
                        None,
                        f"term front matter is missing scalar field '{field}'",
                    )
                )
        title = metadata.get("title", path.stem)
        if not isinstance(title, str):
            title = path.stem
        terms.append(
            Term(
                path=path.resolve(),
                relative_path=relative,
                title=title,
                aliases=tuple(string_list(metadata, "aliases")),
                appears_in=frozenset(string_list(metadata, "appears_in")),
            )
        )
    return terms


def validate(root: Path, *, strict_mentions: bool = False) -> list[Issue]:
    issues: list[Issue] = []
    terms = load_terms(root, issues)
    term_by_path = {term.path: term for term in terms}

    owners: dict[str, Term] = {}
    for term in terms:
        for name in term.names:
            normalized = normalize_name(name)
            if not normalized:
                continue
            owner = owners.get(normalized)
            if owner and owner.path != term.path:
                issues.append(
                    Issue(
                        "error",
                        "name-collision",
                        term.relative_path,
                        None,
                        f"name or alias '{name}' is already owned by {owner.relative_path}",
                        term.title,
                    )
                )
            else:
                owners[normalized] = term

    docs_files = [
        path
        for path in sorted((root / "docs").rglob("*.md"))
        if "_site" not in path.parts
        and "terms" not in path.relative_to(root / "docs").parts[:1]
        and "logs" not in path.relative_to(root / "docs").parts[:1]
    ]
    links_by_doc: dict[str, dict[Path, list[int]]] = {}
    for path in docs_files:
        relative = repo_path(root, path)
        term_links: dict[Path, list[int]] = {}
        for line_number, _, target in markdown_links(path):
            resolved = resolve_link(root, path, target)
            if resolved in term_by_path:
                term_links.setdefault(resolved, []).append(line_number)
        links_by_doc[relative] = term_links

    index_path = root / "docs" / "terms" / "index.md"
    index_targets = {
        resolved
        for _, _, target in markdown_links(index_path)
        if (resolved := resolve_link(root, index_path, target)) is not None
    }

    for term in terms:
        if term.path not in index_targets:
            issues.append(
                Issue(
                    "error",
                    "missing-index-entry",
                    term.relative_path,
                    None,
                    "term is not linked from docs/terms/index.md",
                    term.title,
                )
            )

        where_targets: dict[str, int] = {}
        for line_number, _, target in where_it_appears_links(term.path):
            resolved = resolve_link(root, term.path, target)
            if resolved is None:
                continue
            try:
                relative_target = repo_path(root, resolved)
            except ValueError:
                continue
            if relative_target.startswith("docs/") and not relative_target.startswith(
                "docs/terms/"
            ):
                where_targets[relative_target] = line_number

        for document in sorted(term.appears_in):
            # Term-to-term navigation belongs under Related Terms, not in the
            # consumer-page backlink contract.
            if document.startswith("docs/terms/"):
                continue
            full_document = (root / document).resolve()
            if not full_document.is_file():
                issues.append(
                    Issue(
                        "error",
                        "missing-appears-in-page",
                        term.relative_path,
                        None,
                        f"appears_in path does not exist: {document}",
                        term.title,
                    )
                )
                continue
            if term.path not in links_by_doc.get(document, {}):
                issues.append(
                    Issue(
                        "error",
                        "missing-term-link",
                        document,
                        None,
                        f"appears_in requires a Markdown link to {term.relative_path}",
                        term.title,
                    )
                )
            if document not in where_targets:
                issues.append(
                    Issue(
                        "error",
                        "missing-where-it-appears-link",
                        term.relative_path,
                        None,
                        f"Where It Appears is missing {document}",
                        term.title,
                    )
                )

        for document, line_number in sorted(where_targets.items()):
            if document not in term.appears_in:
                issues.append(
                    Issue(
                        "error",
                        "unregistered-where-it-appears-link",
                        term.relative_path,
                        line_number,
                        f"Where It Appears links {document}, but appears_in does not list it",
                        term.title,
                    )
                )

    for document, term_links in links_by_doc.items():
        path = root / document
        for term_path, line_numbers in term_links.items():
            term = term_by_path[term_path]
            if document not in term.appears_in:
                issues.append(
                    Issue(
                        "error",
                        "unregistered-term-link",
                        document,
                        line_numbers[0],
                        f"links {term.relative_path}, but that term's appears_in does not list this page",
                        term.title,
                    )
                )

        for term in terms:
            mention = first_mention(path, term)
            if mention is None or term.path in term_links:
                continue
            line_number, matched_name = mention
            severity = "error" if strict_mentions else "warning"
            issues.append(
                Issue(
                    severity,
                    "unlinked-term-mention",
                    document,
                    line_number,
                    f"mentions '{matched_name}' but never links {term.relative_path}",
                    term.title,
                )
            )

    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument(
        "--strict-mentions",
        action="store_true",
        help="treat unlinked plain-text term mentions as errors",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    issues = validate(root, strict_mentions=args.strict_mentions)
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]

    if args.json_output:
        print(
            json.dumps(
                {
                    "ok": not errors,
                    "error_count": len(errors),
                    "warning_count": len(warnings),
                    "issues": [asdict(issue) for issue in issues],
                },
                indent=2,
            )
        )
    else:
        for issue in issues:
            location = issue.path
            if issue.line is not None:
                location += f":{issue.line}"
            print(
                f"term links {issue.severity}: {location}: "
                f"{issue.message} [{issue.kind}]"
            )
        if errors:
            print(
                f"term links found {len(errors)} error(s) and "
                f"{len(warnings)} warning(s)"
            )
        else:
            print(f"term links ok ({len(warnings)} warning(s))")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
