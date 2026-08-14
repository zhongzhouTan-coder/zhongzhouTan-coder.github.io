#!/usr/bin/env python3
"""Validate glossary metadata, backlinks, and unlinked term mentions."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
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
TERMLINT_IGNORE_MARKER = "termlint-ignore:"
TERMLINT_IGNORE_RE = re.compile(
    r"<!--\s*termlint-ignore:\s*([a-z0-9]+(?:-[a-z0-9]+)*)\s+--\s+"
    r"(\S(?:.*?\S)?)\s*-->",
    re.IGNORECASE,
)


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
    mention_aliases: tuple[str, ...]
    mention_lint: str
    appears_in: frozenset[str]

    @property
    def names(self) -> tuple[str, ...]:
        return (self.title, *self.aliases)

    @property
    def mention_names(self) -> tuple[str, ...]:
        if self.mention_lint == "off":
            return ()
        if self.mention_lint == "aliases":
            return (self.title, *self.mention_aliases)
        return (self.title,)

    @property
    def slug(self) -> str:
        return self.path.stem


@dataclass(frozen=True)
class EligibleLine:
    number: int
    raw: str
    scrubbed: str


@dataclass(frozen=True)
class IgnoreDirective:
    line: int
    target_line: int | None
    slug: str
    reason: str


@dataclass(frozen=True)
class Fix:
    term_path: str
    document_path: str
    added_to_where_it_appears: bool


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


def eligible_line_details(path: Path) -> list[EligibleLine]:
    """Return prose lines eligible for term mention/link validation."""
    lines = path.read_text(encoding="utf-8").splitlines()
    _, front_matter_end = parse_front_matter(path)
    in_fence = False
    fence_marker = ""
    previous_was_image = False
    result: list[EligibleLine] = []

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
        result.append(EligibleLine(line_number, line, scrubbed))
    return result


def eligible_lines(path: Path) -> list[tuple[int, str]]:
    return [(line.number, line.scrubbed) for line in eligible_line_details(path)]


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


def unlinked_prose(line: str) -> str:
    """Remove existing Markdown links before looking for unlinked mentions."""
    return MARKDOWN_LINK_RE.sub("", line)


def first_mention(
    path: Path, term: Term, *, ignored_lines: frozenset[int] = frozenset()
) -> tuple[int, str] | None:
    patterns = [(name, name_pattern(name)) for name in term.mention_names if name]
    for line_number, line in eligible_lines(path):
        if line_number in ignored_lines:
            continue
        line = unlinked_prose(line)
        for name, pattern in patterns:
            if pattern.search(line):
                return line_number, name
    return None


def ignore_directives(path: Path) -> tuple[list[IgnoreDirective], list[tuple[int, str]]]:
    """Parse reviewed term-warning suppressions from eligible prose."""
    details = eligible_line_details(path)
    directives: list[IgnoreDirective] = []
    malformed: list[tuple[int, str]] = []
    previous_prose_line: int | None = None

    for detail in details:
        marker_count = detail.raw.casefold().count(TERMLINT_IGNORE_MARKER)
        matches = list(TERMLINT_IGNORE_RE.finditer(detail.raw))
        if marker_count != len(matches):
            malformed.append((detail.number, detail.raw.strip()))
        for match in matches:
            without_directive = detail.scrubbed.strip()
            target_line = detail.number
            if not without_directive:
                target_line = (
                    previous_prose_line
                    if previous_prose_line == detail.number - 1
                    else None
                )
            directives.append(
                IgnoreDirective(
                    line=detail.number,
                    target_line=target_line,
                    slug=match.group(1).casefold(),
                    reason=match.group(2).strip(),
                )
            )
        if detail.scrubbed.strip():
            previous_prose_line = detail.number
    return directives, malformed


def docs_files(root: Path) -> list[Path]:
    """Return consumer docs checked by the bidirectional term-link contract."""
    return [
        path
        for path in sorted((root / "docs").rglob("*.md"))
        if "_site" not in path.parts
        and "terms" not in path.relative_to(root / "docs").parts[:1]
        and "logs" not in path.relative_to(root / "docs").parts[:1]
    ]


def consumer_term_links(
    root: Path, terms: list[Term]
) -> dict[str, dict[Path, list[int]]]:
    term_by_path = {term.path: term for term in terms}
    result: dict[str, dict[Path, list[int]]] = {}
    for path in docs_files(root):
        relative = repo_path(root, path)
        term_links: dict[Path, list[int]] = {}
        for line_number, _, target in markdown_links(path):
            resolved = resolve_link(root, path, target)
            if resolved in term_by_path:
                term_links.setdefault(resolved, []).append(line_number)
        result[relative] = term_links
    return result


def front_matter_title(path: Path) -> str:
    metadata, _ = parse_front_matter(path)
    title = metadata.get("title")
    return title if isinstance(title, str) and title else path.stem


def set_updated_date(lines: list[str], updated: str) -> None:
    for index, line in enumerate(lines):
        if re.match(r"^updated:\s*", line):
            lines[index] = f"updated: {updated}"
            return


def add_where_it_appears_links(
    root: Path, term: Term, lines: list[str], documents: list[str]
) -> bool:
    """Append deterministic backlinks to an existing glossary section."""
    if not documents:
        return False
    section_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r"^##\s+Where It Appears\s*$", line, re.IGNORECASE)
        ),
        None,
    )
    if section_index is None:
        return False

    insertion = len(lines)
    for index in range(section_index + 1, len(lines)):
        if re.match(r"^##\s+", lines[index]):
            insertion = index
            break
    while insertion > section_index + 1 and not lines[insertion - 1].strip():
        insertion -= 1

    bullets = []
    for document in documents:
        document_path = root / document
        target = os.path.relpath(document_path, term.path.parent).replace(os.sep, "/")
        bullets.append(f"- [{front_matter_title(document_path)}]({target})")
    if insertion > section_index + 1:
        bullets.insert(0, "")
    bullets.append("")
    lines[insertion:insertion] = bullets
    return True


def apply_safe_fixes(root: Path, *, updated: str | None = None) -> list[Fix]:
    """Synchronize curated Where It Appears links from appears_in metadata.

    Consumer links do not promote pages into the curated appears_in set. This
    deliberately does not choose curated pages, create prose links, or repair
    aliases, stale paths, and name collisions; those require semantic judgment.
    """
    load_issues: list[Issue] = []
    terms = load_terms(root, load_issues)
    links_by_doc = consumer_term_links(root, terms)
    updated = updated or dt.date.today().isoformat()
    fixes: list[Fix] = []

    for term in terms:
        existing_where = {
            repo_path(root, resolved)
            for _, _, target in where_it_appears_links(term.path)
            if (resolved := resolve_link(root, term.path, target)) is not None
            and resolved.is_relative_to(root)
        }
        add_to_where = [
            document
            for document in sorted(term.appears_in)
            if document not in existing_where
            and term.path in links_by_doc.get(document, {})
        ]
        if not add_to_where:
            continue

        lines = term.path.read_text(encoding="utf-8").splitlines()
        where_changed = add_where_it_appears_links(root, term, lines, add_to_where)
        if add_to_where and not where_changed:
            continue
        set_updated_date(lines, updated)
        term.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for document in add_to_where:
            fixes.append(
                Fix(
                    term_path=term.relative_path,
                    document_path=document,
                    added_to_where_it_appears=True,
                )
            )
    return fixes


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
        aliases = tuple(string_list(metadata, "aliases"))
        mention_aliases = tuple(string_list(metadata, "mention_aliases"))
        mention_lint = metadata.get("mention_lint", "canonical")
        if not isinstance(mention_lint, str) or mention_lint not in {
            "off",
            "canonical",
            "aliases",
        }:
            issues.append(
                Issue(
                    "error",
                    "invalid-mention-lint",
                    relative,
                    None,
                    "mention_lint must be one of: off, canonical, aliases",
                    title,
                )
            )
            mention_lint = "canonical"
        normalized_aliases = {normalize_name(alias) for alias in aliases}
        for mention_alias in mention_aliases:
            if normalize_name(mention_alias) not in normalized_aliases:
                issues.append(
                    Issue(
                        "error",
                        "unregistered-mention-alias",
                        relative,
                        None,
                        (
                            f"mention_aliases entry '{mention_alias}' must also "
                            "appear in aliases"
                        ),
                        title,
                    )
                )
        terms.append(
            Term(
                path=path.resolve(),
                relative_path=relative,
                title=title,
                aliases=aliases,
                mention_aliases=mention_aliases,
                mention_lint=mention_lint,
                appears_in=frozenset(string_list(metadata, "appears_in")),
            )
        )
    return terms


def review_documents(root: Path, paths: tuple[Path, ...]) -> set[str]:
    """Resolve an optional file/directory scope for mention review."""
    documents = docs_files(root)
    docs_root = (root / "docs").resolve()
    if not paths:
        return {repo_path(root, path) for path in documents}

    selected: set[str] = set()
    for requested in paths:
        candidate = requested if requested.is_absolute() else root / requested
        candidate = candidate.resolve()
        if not candidate.is_relative_to(docs_root):
            raise ValueError(f"mention-review path is outside docs/: {requested}")
        if candidate.is_file():
            if candidate not in documents:
                raise ValueError(f"path is not a consumer docs page: {requested}")
            selected.add(repo_path(root, candidate))
            continue
        if candidate.is_dir():
            selected.update(
                repo_path(root, document)
                for document in documents
                if document.is_relative_to(candidate)
            )
            continue
        raise ValueError(f"mention-review path does not exist: {requested}")
    return selected


def validate(
    root: Path,
    *,
    strict_mentions: bool = False,
    review_paths: tuple[Path, ...] | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    terms = load_terms(root, issues)
    term_by_slug = {term.slug: term for term in terms}

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

    links_by_doc = consumer_term_links(root, terms)
    if strict_mentions and review_paths is None:
        review_paths = ()
    reviewed_documents = (
        review_documents(root, review_paths) if review_paths is not None else set()
    )

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
        directives, malformed_directives = ignore_directives(path)
        ignored_lines_by_term: dict[Path, set[int]] = {}

        for line_number, _ in malformed_directives:
            issues.append(
                Issue(
                    "error",
                    "invalid-termlint-ignore",
                    document,
                    line_number,
                    (
                        "termlint-ignore must use '<!-- termlint-ignore: "
                        "term-slug -- review reason -->'"
                    ),
                )
            )

        eligible_by_number = {
            line_number: line for line_number, line in eligible_lines(path)
        }
        for directive in directives:
            term = term_by_slug.get(directive.slug)
            if term is None:
                issues.append(
                    Issue(
                        "error",
                        "unknown-termlint-ignore",
                        document,
                        directive.line,
                        f"termlint-ignore references unknown term slug '{directive.slug}'",
                    )
                )
                continue
            target_text = unlinked_prose(
                eligible_by_number.get(directive.target_line or -1, "")
            )
            mentions_term = any(
                name_pattern(name).search(target_text)
                for name in term.mention_names
                if name
            )
            if term.path in term_links or not mentions_term:
                issues.append(
                    Issue(
                        "error",
                        "stale-termlint-ignore",
                        document,
                        directive.line,
                        (
                            f"termlint-ignore for '{directive.slug}' no longer suppresses "
                            "an unlinked detectable mention"
                        ),
                        term.title,
                    )
                )
                continue
            if directive.target_line is not None:
                ignored_lines_by_term.setdefault(term.path, set()).add(
                    directive.target_line
                )

        if document not in reviewed_documents:
            continue
        for term in terms:
            mention = first_mention(
                path,
                term,
                ignored_lines=frozenset(ignored_lines_by_term.get(term.path, set())),
            )
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
        "--fix",
        action="store_true",
        help=(
            "add missing Where It Appears links for pages already curated in "
            "appears_in; consumer links never expand the curated set"
        ),
    )
    parser.add_argument(
        "--review-mentions",
        nargs="*",
        type=Path,
        metavar="PATH",
        help=(
            "review unlinked mentions in all consumer docs, or only beneath "
            "the supplied files/directories; omitted during structural lint"
        ),
    )
    parser.add_argument(
        "--strict-mentions",
        action="store_true",
        help="treat unlinked plain-text term mentions as errors",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    fixes = apply_safe_fixes(root) if args.fix else []
    try:
        issues = validate(
            root,
            strict_mentions=args.strict_mentions,
            review_paths=(
                tuple(args.review_mentions)
                if args.review_mentions is not None
                else None
            ),
        )
    except ValueError as error:
        print(f"term links error: {error}", file=sys.stderr)
        return 2
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]

    if args.json_output:
        print(
            json.dumps(
                {
                    "ok": not errors,
                    "error_count": len(errors),
                    "warning_count": len(warnings),
                    "fixes": [asdict(fix) for fix in fixes],
                    "issues": [asdict(issue) for issue in issues],
                },
                indent=2,
            )
        )
    else:
        for fix in fixes:
            print(
                f"term links fixed: {fix.term_path}: added curated "
                f"Where It Appears link for {fix.document_path}"
            )
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
