#!/usr/bin/env python3
"""Check that inline TeX survives both Kramdown and VS Code/KaTeX rendering."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from html import unescape
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(SCRIPTS_ROOT))

from common.paths import find_repository_root  # noqa: E402


REPO_ROOT = find_repository_root(__file__)
DOCS_ROOT = REPO_ROOT / "docs"
INLINE_MATH_RE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")
INLINE_CODE_RE = re.compile(r"(?P<delimiter>`+)(?P<content>.*?)(?P=delimiter)")
INLINE_CODE_MATH_RE = re.compile(r"(?<!\$)\$(?![\s$])[^$\n]*?(?<!\s)\$(?!\$)")
RENDERED_MATH_RE = re.compile(r"(?:\$[^$\n]*\$|\\\([^\n]*?\\\))")


def source_issues(text: str) -> list[tuple[int, str]]:
    """Return KaTeX-incompatible or preview-breaking inline-math constructs."""
    issues: list[tuple[int, str]] = []
    in_fence = False
    display_math_start: int | None = None
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.strip() == "$$":
            display_math_start = line_number if display_math_start is None else None
            continue
        if display_math_start is not None and "$$" in line:
            issues.append(
                (
                    line_number,
                    "redundant/nested $$ delimiter inside display math block "
                    f"opened on line {display_math_start}",
                )
            )
        for code_match in INLINE_CODE_RE.finditer(line):
            if INLINE_CODE_MATH_RE.search(code_match.group("content")):
                issues.append(
                    (
                        line_number,
                        "inline $...$ math is inside a code span and will not render",
                    )
                )
        for match in INLINE_MATH_RE.finditer(line):
            formula = match.group(1)
            if r"\sb" in formula:
                issues.append((line_number, r"unsupported KaTeX command \sb"))
            if r"\_" in formula:
                issues.append(
                    (line_number, r"escaped underscore \_ renders literally in VS Code")
                )
    if display_math_start is not None:
        issues.append(
            (display_math_start, "unclosed display math block opened with $$")
        )
    return issues


def rendered_issue_details(text: str) -> list[tuple[int, str]]:
    """Return generated HTML lines and math fragments corrupted by Markdown."""
    issues: list[tuple[int, str]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        for match in RENDERED_MATH_RE.finditer(line):
            fragment = match.group(0)
            if "<em>" in fragment or "</em>" in fragment:
                issues.append((line_number, fragment))
                break
            if "<strong>" in fragment or "</strong>" in fragment:
                issues.append((line_number, fragment))
                break
    return issues


def rendered_issues(text: str) -> list[int]:
    """Return generated HTML lines where Markdown formatting corrupted math."""
    return [line_number for line_number, _ in rendered_issue_details(text)]


def source_location_for_fragment(
    fragment: str, markdown_sources: dict[Path, str]
) -> tuple[Path, int] | None:
    """Locate a rendered, emphasis-corrupted math fragment in Markdown sources."""
    source_fragment = unescape(
        fragment.replace("<em>", "_")
        .replace("</em>", "_")
        .replace("<strong>", "**")
        .replace("</strong>", "**")
    )
    matches: list[tuple[Path, int]] = []
    for markdown_path, source_text in markdown_sources.items():
        for line_number, line in enumerate(source_text.splitlines(), 1):
            if source_fragment in line:
                matches.append((markdown_path, line_number))
    if len(matches) == 1:
        return matches[0]
    return None


def build_site(destination: Path) -> None:
    subprocess.run(
        ["bundle", "exec", "jekyll", "build", "--destination", str(destination)],
        cwd=DOCS_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def main() -> int:
    found_issue = False
    markdown_sources = {
        markdown_path: markdown_path.read_text()
        for markdown_path in sorted(DOCS_ROOT.rglob("*.md"))
        if "_site" not in markdown_path.parts
    }
    for markdown_path, source_text in markdown_sources.items():
        for line_number, message in source_issues(source_text):
            relative_path = markdown_path.relative_to(REPO_ROOT)
            print(f"math formulation: {relative_path}:{line_number}: {message}")
            found_issue = True

    with tempfile.TemporaryDirectory(prefix="kb-math-render-") as temp_dir:
        destination = Path(temp_dir)
        try:
            build_site(destination)
        except FileNotFoundError as error:
            print(f"math formulation: unable to build Jekyll site: {error}")
            return 1
        except subprocess.CalledProcessError as error:
            print(f"math formulation: unable to build Jekyll site: {error}")
            return 1

        for html_path in sorted(destination.rglob("*.html")):
            for line_number, fragment in rendered_issue_details(html_path.read_text()):
                generated_path = html_path.relative_to(destination)
                source_location = source_location_for_fragment(
                    fragment, markdown_sources
                )
                if source_location is None:
                    location = f"generated/{generated_path}:{line_number}"
                else:
                    markdown_path, source_line = source_location
                    source_path = markdown_path.relative_to(REPO_ROOT)
                    location = (
                        f"{source_path}:{source_line} "
                        f"(generated/{generated_path}:{line_number})"
                    )
                print(
                    f"math formulation: {location}: "
                    "Markdown emphasis markup appeared inside a math delimiter"
                )
                found_issue = True

    if found_issue:
        return 1
    print("math formulations ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
