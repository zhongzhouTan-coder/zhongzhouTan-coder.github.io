from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "term_links", ROOT / "scripts" / "checks" / "term_links.py"
)
assert SPEC is not None and SPEC.loader is not None
TERM_LINKS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TERM_LINKS
SPEC.loader.exec_module(TERM_LINKS)


class TermLinkTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        (self.root / "docs" / "terms").mkdir(parents=True)

    def write(self, relative_path: str, content: str) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def write_term(self, appears_in: str = "docs/topic/page.md") -> None:
        appears = f"\n  - {appears_in}" if appears_in else ""
        where = (
            "\n- [Topic](../topic/page.md) — Usage."
            if appears_in
            else "\n_No local pages yet._"
        )
        self.write(
            "docs/terms/kv-cache.md",
            f"""---
title: "KV Cache"
summary: "Stored attention state."
category: algorithms
aliases:
  - key-value cache
appears_in:{appears}
---

# KV Cache

**KV Cache** is stored attention state.

## Where It Appears
{where}
""",
        )
        self.write(
            "docs/terms/index.md",
            "# Terms\n\n- [KV Cache](kv-cache.md) — Stored attention state.\n",
        )

    def kinds(self, **kwargs: bool) -> list[str]:
        return [issue.kind for issue in TERM_LINKS.validate(self.root, **kwargs)]

    def test_valid_bidirectional_links_pass(self) -> None:
        self.write_term()
        self.write(
            "docs/topic/page.md",
            "---\ntitle: Topic\n---\n\nUses a [KV cache](../terms/kv-cache.md).\n",
        )

        self.assertEqual(self.kinds(), [])

    def test_appears_in_requires_document_link(self) -> None:
        self.write_term()
        self.write(
            "docs/topic/page.md",
            "---\ntitle: KV Cache in metadata\n---\n\n# KV Cache heading\n\n```text\nKV cache in code\n```\n",
        )

        kinds = self.kinds()
        self.assertIn("missing-term-link", kinds)
        self.assertNotIn("unlinked-term-mention", kinds)

    def test_document_link_requires_appears_in_registration(self) -> None:
        self.write_term(appears_in="")
        self.write(
            "docs/topic/page.md",
            "A [key-value cache](../terms/kv-cache.md) stores attention state.\n",
        )

        self.assertIn("unregistered-term-link", self.kinds())

    def test_plain_mention_warns_and_strict_mode_fails(self) -> None:
        self.write_term(appears_in="")
        self.write("docs/topic/page.md", "A key-value cache stores state.\n")

        issues = TERM_LINKS.validate(self.root)
        mention = next(
            issue for issue in issues if issue.kind == "unlinked-term-mention"
        )
        self.assertEqual(mention.severity, "warning")

        strict_issues = TERM_LINKS.validate(self.root, strict_mentions=True)
        strict_mention = next(
            issue
            for issue in strict_issues
            if issue.kind == "unlinked-term-mention"
        )
        self.assertEqual(strict_mention.severity, "error")

    def test_where_it_appears_matches_front_matter(self) -> None:
        self.write_term()
        self.write(
            "docs/topic/page.md",
            "A [KV cache](../terms/kv-cache.md) stores state.\n",
        )
        term_path = self.root / "docs" / "terms" / "kv-cache.md"
        term_path.write_text(
            term_path.read_text(encoding="utf-8").replace(
                "- [Topic](../topic/page.md) — Usage.", "_Missing backlink._"
            ),
            encoding="utf-8",
        )

        self.assertIn("missing-where-it-appears-link", self.kinds())

    def test_alias_collisions_and_missing_index_entries_fail(self) -> None:
        self.write_term(appears_in="")
        self.write(
            "docs/terms/attention-state.md",
            """---
title: "Attention State"
summary: "Stored state."
category: algorithms
aliases:
  - KV Cache
appears_in:
---

# Attention State

**Attention State** is stored state.

## Where It Appears

*No local pages yet.*
""",
        )

        kinds = self.kinds()
        self.assertIn("name-collision", kinds)
        self.assertIn("missing-index-entry", kinds)


if __name__ == "__main__":
    unittest.main()
