from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "checks" / "math_rendering.py"
SPEC = spec_from_file_location("check_math_rendering", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_source_issues_reject_cross_renderer_workarounds() -> None:
    text = r"""
Good: $\underset{t}{\mathbf{h}}$
Bad for preview: $\mathbf{h}\_t$
Bad for KaTeX: $\mathbf{h}\sb{t}$
```text
$\mathbf{h}\sb{ignored}$
```
"""

    assert MODULE.source_issues(text) == [
        (3, r"escaped underscore \_ renders literally in VS Code"),
        (4, r"unsupported KaTeX command \sb"),
    ]


def test_rendered_issues_detect_markdown_inside_math() -> None:
    html = "\n".join(
        [
            r"<p>Good: $\underset{t}{\mathbf{h}}$</p>",
            r"<p>Bad: $\mathbf{q}<em>{t,j}^I$, $w</em>{t,j}^I$</p>",
            r"<p>Also good: \(x_t\)</p>",
        ]
    )

    assert MODULE.rendered_issues(html) == [2]
