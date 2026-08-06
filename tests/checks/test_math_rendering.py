from importlib.util import module_from_spec, spec_from_file_location
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "checks" / "math_rendering.py"
SPEC = spec_from_file_location("check_math_rendering", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MathRenderingTests(unittest.TestCase):
    def test_source_issues_reject_cross_renderer_workarounds(self) -> None:
        text = r"""
Good: $\underset{t}{\mathbf{h}}$
Bad for preview: $\mathbf{h}\_t$
Bad for KaTeX: $\mathbf{h}\sb{t}$
```text
$\mathbf{h}\sb{ignored}$
```
"""

        self.assertEqual(
            MODULE.source_issues(text),
            [
                (3, r"escaped underscore \_ renders literally in VS Code"),
                (4, r"unsupported KaTeX command \sb"),
            ],
        )

    def test_source_issues_reject_nested_display_math_delimiters(self) -> None:
        text = r"""Before
$$
$$x_t = 1$$
$$
After
"""

        self.assertEqual(
            MODULE.source_issues(text),
            [
                (
                    3,
                    "redundant/nested $$ delimiter inside display math block "
                    "opened on line 2",
                )
            ],
        )

    def test_source_issues_reject_unclosed_display_math_block(self) -> None:
        self.assertEqual(
            MODULE.source_issues("Before\n$$\nx_t = 1\n"),
            [(2, "unclosed display math block opened with $$")],
        )

    def test_rendered_issues_detect_markdown_inside_math(self) -> None:
        html = "\n".join(
            [
                r"<p>Good: $\underset{t}{\mathbf{h}}$</p>",
                r"<p>Bad: $\mathbf{q}<em>{t,j}^I$, $w</em>{t,j}^I$</p>",
                r"<p>Also good: \(x_t\)</p>",
            ]
        )

        self.assertEqual(MODULE.rendered_issues(html), [2])

    def test_corrupted_rendered_math_maps_back_to_source_line(self) -> None:
        markdown_path = Path("docs/example.md")
        sources = {
            markdown_path: (
                "Before\n"
                r"Bad: $\mathbf{q}_{t,j}^I$, $w_{t,j}^I$"
                "\nAfter\n"
            )
        }
        fragment = r"$\mathbf{q}<em>{t,j}^I$"

        self.assertEqual(
            MODULE.source_location_for_fragment(fragment, sources),
            (markdown_path, 2),
        )

    def test_ambiguous_rendered_math_keeps_generated_location(self) -> None:
        fragment = r"$x<em>t$"
        sources = {
            Path("docs/one.md"): r"$x_t$",
            Path("docs/two.md"): r"$x_t$",
        }

        self.assertIsNone(
            MODULE.source_location_for_fragment(fragment, sources)
        )

    def test_main_reports_source_and_generated_locations(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repository_root = Path(temp_dir)
            docs_root = repository_root / "docs"
            markdown_path = docs_root / "example" / "index.md"
            markdown_path.parent.mkdir(parents=True)
            markdown_path.write_text(r"Bad: $x_t$, $y_t$" + "\n")

            def fake_build(destination: Path) -> None:
                html_path = destination / "example" / "index.html"
                html_path.parent.mkdir(parents=True)
                html_path.write_text(r"<p>Bad: $x<em>t$, $y</em>t$</p>" + "\n")

            output = StringIO()
            with (
                patch.object(MODULE, "REPO_ROOT", repository_root),
                patch.object(MODULE, "DOCS_ROOT", docs_root),
                patch.object(MODULE, "build_site", side_effect=fake_build),
                redirect_stdout(output),
            ):
                self.assertEqual(MODULE.main(), 1)

            self.assertIn(
                "math formulation: docs/example/index.md:1 "
                "(generated/example/index.html:1): "
                "Markdown emphasis markup appeared inside a math delimiter",
                output.getvalue(),
            )

    def test_main_fails_when_jekyll_is_unavailable(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with (
                patch.object(MODULE, "DOCS_ROOT", Path(temp_dir)),
                patch.object(
                    MODULE,
                    "build_site",
                    side_effect=FileNotFoundError("bundle is unavailable"),
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(MODULE.main(), 1)
