#!/usr/bin/env python3
"""Check that inline TeX survives both Kramdown and VS Code/KaTeX rendering."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(SCRIPTS_ROOT))

from common.paths import find_repository_root  # noqa: E402


REPO_ROOT = find_repository_root(__file__)
DOCS_ROOT = REPO_ROOT / "docs"
INLINE_MATH_RE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")
RENDERED_MATH_RE = re.compile(r"(?:\$[^$\n]*\$|\\\([^\n]*?\\\))")


def source_issues(text: str) -> list[tuple[int, str]]:
    """Return KaTeX-incompatible or preview-breaking inline-math constructs."""
    issues: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in INLINE_MATH_RE.finditer(line):
            formula = match.group(1)
            if r"\sb" in formula:
                issues.append((line_number, r"unsupported KaTeX command \sb"))
            if r"\_" in formula:
                issues.append(
                    (line_number, r"escaped underscore \_ renders literally in VS Code")
                )
    return issues


def rendered_issues(text: str) -> list[int]:
    """Return generated HTML lines where Markdown formatting corrupted math."""
    issues: list[int] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        for match in RENDERED_MATH_RE.finditer(line):
            fragment = match.group(0)
            if "<em>" in fragment or "</em>" in fragment:
                issues.append(line_number)
                break
            if "<strong>" in fragment or "</strong>" in fragment:
                issues.append(line_number)
                break
    return issues


def build_site(destination: Path) -> None:
    subprocess.run(
        ["bundle", "exec", "jekyll", "build", "--destination", str(destination)],
        cwd=DOCS_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def main() -> int:
    found_issue = False
    for markdown_path in sorted(DOCS_ROOT.rglob("*.md")):
        if "_site" in markdown_path.parts:
            continue
        for line_number, message in source_issues(markdown_path.read_text()):
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
            for line_number in rendered_issues(html_path.read_text()):
                relative_path = html_path.relative_to(destination)
                print(
                    "math formulation: "
                    f"generated/{relative_path}:{line_number}: "
                    "Markdown emphasis markup appeared inside a math delimiter"
                )
                found_issue = True

    if found_issue:
        return 1
    print("math formulations ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
