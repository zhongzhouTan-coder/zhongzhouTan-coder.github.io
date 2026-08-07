from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_code_links", ROOT / "scripts" / "checks" / "code_links.py"
)
assert SPEC is not None and SPEC.loader is not None
CHECK_CODE_LINKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK_CODE_LINKS)


class CheckCodeLinksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.markdown_path = ROOT / "docs" / "example.md"
        self.registry = {
            "example-deadbeef0000": {
                "local_checkout": "external-repos/example",
                "provider": "github",
                "repository_url": "https://github.com/example/example",
                "revision": "deadbeef00000000000000000000000000000000",
            }
        }

    def write_strict_page(self, content: str) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        markdown_path = Path(temporary_directory.name) / "example.md"
        markdown_path.write_text(content, encoding="utf-8")
        return markdown_path

    def test_missing_line_is_reported_without_crashing(self) -> None:
        errors = CHECK_CODE_LINKS.validate_include(
            self.markdown_path,
            {
                "href": "../external-repos/example/src/example.py",
                "data-code-repo": "example-deadbeef0000",
                "data-code-path": "src/example.py",
            },
            self.registry,
        )

        self.assertEqual(
            errors, ["docs/example.md: code link is missing: data-code-line"]
        )

    def test_repository_path_cannot_escape_checkout(self) -> None:
        errors = CHECK_CODE_LINKS.validate_include(
            self.markdown_path,
            {
                "href": "../external-repos/example/secret.py#L4",
                "data-code-repo": "example-deadbeef0000",
                "data-code-path": "../secret.py",
                "data-code-line": "4",
            },
            self.registry,
        )

        self.assertIn("docs/example.md: code path must be repository-relative", errors)

    def test_end_line_cannot_precede_start(self) -> None:
        errors = CHECK_CODE_LINKS.validate_include(
            self.markdown_path,
            {
                "href": "../external-repos/example/src/example.py#L20",
                "data-code-repo": "example-deadbeef0000",
                "data-code-path": "src/example.py",
                "data-code-line": "20",
                "data-code-end-line": "10",
            },
            self.registry,
        )

        self.assertIn(
            "docs/example.md: data-code-end-line cannot precede data-code-line",
            errors,
        )

    def test_unclosed_code_link_is_reported_at_opening_line(self) -> None:
        markdown_path = self.write_strict_page(
            """# Example

<a class="code-link" href="../external-repos/example/worker.py#L1"
data-code-repo="example" data-code-path="worker.py"
data-code-line="1"><code>worker.py</code> continues as linked text.
"""
        )

        links, errors = CHECK_CODE_LINKS.parse_code_links(markdown_path)

        self.assertEqual(len(links), 1)
        self.assertEqual(len(errors), 1)
        self.assertIn(":3: code-link anchor is missing </a>", errors[0])

    def test_closed_code_link_has_no_parse_error(self) -> None:
        markdown_path = self.write_strict_page(
            """# Example

<a class="code-link" href="../external-repos/example/worker.py#L1"
data-code-repo="example" data-code-path="worker.py"
data-code-line="1"><code>worker.py</code></a> continues as plain text.
"""
        )

        links, errors = CHECK_CODE_LINKS.parse_code_links(markdown_path)

        self.assertEqual(len(links), 1)
        self.assertEqual(errors, [])

    def test_checkout_cannot_represent_two_revisions(self) -> None:
        registry = {
            **self.registry,
            "example-cafebabe0000": {
                "local_checkout": "external-repos/example",
                "provider": "github",
                "repository_url": "https://github.com/example/example",
                "revision": "cafebabe00000000000000000000000000000000",
            },
        }
        sources = {
            (entry["provider"], entry["repository_url"], entry["revision"])
            for entry in registry.values()
        }

        errors = CHECK_CODE_LINKS.validate_registry(registry, sources)

        self.assertTrue(
            any("reuses 'external-repos/example'" in error for error in errors)
        )

    def test_strict_page_reports_unlinked_repository_filename(self) -> None:
        markdown_path = self.write_strict_page(
            """---
code_links: strict
---

Use `worker.py`, but keep `torch.empty` as ordinary code.
"""
        )

        findings = CHECK_CODE_LINKS.find_unlinked_repository_paths(markdown_path)

        self.assertEqual(findings, [(5, "worker.py")])

    def test_strict_page_ignores_linked_paths_and_code_fences(self) -> None:
        markdown_path = self.write_strict_page(
            """---
code_links: strict
---

<a class="code-link" href="../external-repos/example/worker.py#L1"
data-code-repo="example" data-code-path="worker.py"
data-code-line="1"><code>worker.py</code></a>

```text
generated.py
```

Still report `unlinked.py` on its original line.
"""
        )

        findings = CHECK_CODE_LINKS.find_unlinked_repository_paths(markdown_path)

        self.assertEqual(findings, [(13, "unlinked.py")])

    def test_direct_markdown_checkout_link_is_reported(self) -> None:
        markdown_path = self.write_strict_page(
            """# Analysis

[worker source](../external-repos/example/worker.py#L1)
"""
        )

        findings = CHECK_CODE_LINKS.find_direct_checkout_links(markdown_path)

        self.assertEqual(
            findings, [(3, "../external-repos/example/worker.py#L1")]
        )

    def test_checkout_link_in_code_fence_is_ignored(self) -> None:
        markdown_path = self.write_strict_page(
            """# Example

```markdown
[example](../external-repos/example/worker.py#L1)
```
"""
        )

        findings = CHECK_CODE_LINKS.find_direct_checkout_links(markdown_path)

        self.assertEqual(findings, [])

    def test_checkout_href_without_code_link_metadata_is_reported(self) -> None:
        markdown_path = self.write_strict_page(
            """# Example

<a href="../external-repos/example/worker.py#L1">worker</a>
"""
        )

        findings = CHECK_CODE_LINKS.find_direct_checkout_links(markdown_path)

        self.assertEqual(
            findings, [(3, "../external-repos/example/worker.py#L1")]
        )

    def test_parses_machine_checkable_evidence_table(self) -> None:
        markdown_path = self.write_strict_page(
            """# Analysis

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/example.md` | allocation | `src/pool.py` | `Pool.allocate` | 20 | 35 |
"""
        )

        evidence, errors = CHECK_CODE_LINKS.parse_code_evidence(markdown_path)

        self.assertEqual(errors, [])
        self.assertEqual(len(evidence), 1)
        self.assertEqual(str(evidence[0].docs_path), "docs/example.md")
        self.assertEqual(str(evidence[0].code_path), "src/pool.py")
        self.assertEqual(evidence[0].start_line, 20)
        self.assertEqual(evidence[0].end_line, 35)

    def test_declared_evidence_requires_matching_link(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        temporary_root = Path(temporary_directory.name)
        docs_path = temporary_root / "docs" / "example.md"
        docs_path.parent.mkdir(parents=True)
        docs_path.write_text("# Example\n", encoding="utf-8")
        evidence = CHECK_CODE_LINKS.CodeEvidence(
            source_path=temporary_root / "derived" / "analysis.md",
            source_line=8,
            docs_path=CHECK_CODE_LINKS.PurePosixPath("docs/example.md"),
            finding="allocation",
            code_path=CHECK_CODE_LINKS.PurePosixPath("src/pool.py"),
            symbol="Pool.allocate",
            start_line=20,
            end_line=None,
        )

        errors = CHECK_CODE_LINKS.validate_evidence_coverage(
            [evidence], [], root=temporary_root
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("missing declared code evidence 'allocation'", errors[0])

    def test_declared_evidence_accepts_matching_link(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        temporary_root = Path(temporary_directory.name)
        docs_path = temporary_root / "docs" / "example.md"
        docs_path.parent.mkdir(parents=True)
        docs_path.write_text("# Example\n", encoding="utf-8")
        evidence = CHECK_CODE_LINKS.CodeEvidence(
            source_path=temporary_root / "derived" / "analysis.md",
            source_line=8,
            docs_path=CHECK_CODE_LINKS.PurePosixPath("docs/example.md"),
            finding="allocation",
            code_path=CHECK_CODE_LINKS.PurePosixPath("src/pool.py"),
            symbol="Pool.allocate",
            start_line=20,
            end_line=35,
        )
        links = [
            (
                docs_path,
                {
                    "data-code-path": "src/pool.py",
                    "data-code-line": "20",
                    "data-code-end-line": "35",
                },
            )
        ]

        errors = CHECK_CODE_LINKS.validate_evidence_coverage(
            [evidence], links, root=temporary_root
        )

        self.assertEqual(errors, [])

    def test_strict_evidence_page_requires_a_declaration(self) -> None:
        errors = CHECK_CODE_LINKS.validate_required_evidence_pages(
            [self.markdown_path], []
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("code_evidence: strict requires", errors[0])


if __name__ == "__main__":
    unittest.main()
