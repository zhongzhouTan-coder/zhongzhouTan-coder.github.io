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

    def write_term(
        self,
        appears_in: str = "docs/topic/page.md",
        mention_aliases: tuple[str, ...] = (),
        mention_lint: str = "canonical",
    ) -> None:
        appears = f"\n  - {appears_in}" if appears_in else ""
        detectable_aliases = "".join(
            f"\n  - {alias}" for alias in mention_aliases
        )
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
mention_aliases:{detectable_aliases}
mention_lint: {mention_lint}
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

    def kinds(self, **kwargs: object) -> list[str]:
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

    def test_consumer_link_does_not_require_curated_registration(self) -> None:
        self.write_term(appears_in="")
        self.write(
            "docs/topic/page.md",
            "A [key-value cache](../terms/kv-cache.md) stores attention state.\n",
        )

        self.assertEqual(self.kinds(), [])

    def test_fix_repairs_where_it_appears_for_a_curated_page(self) -> None:
        self.write_term()
        self.write(
            "docs/topic/page.md",
            "---\ntitle: Topic Page\n---\n\n"
            "A [key-value cache](../terms/kv-cache.md) stores state.\n",
        )
        term_path = self.root / "docs/terms/kv-cache.md"
        term_path.write_text(
            term_path.read_text(encoding="utf-8").replace(
                "- [Topic](../topic/page.md) — Usage.", "_Missing backlink._"
            ),
            encoding="utf-8",
        )

        fixes = TERM_LINKS.apply_safe_fixes(self.root, updated="2026-08-10")

        self.assertEqual(len(fixes), 1)
        self.assertTrue(fixes[0].added_to_where_it_appears)
        term_text = term_path.read_text(encoding="utf-8")
        self.assertIn("- [Topic Page](../topic/page.md)", term_text)
        self.assertEqual(self.kinds(), [])

    def test_fix_does_not_promote_a_consumer_into_appears_in(self) -> None:
        self.write_term(appears_in="")
        self.write(
            "docs/topic/page.md",
            "A [key-value cache](../terms/kv-cache.md) stores state.\n",
        )

        fixes = TERM_LINKS.apply_safe_fixes(self.root, updated="2026-08-10")

        self.assertEqual(fixes, [])
        term_text = (self.root / "docs/terms/kv-cache.md").read_text(encoding="utf-8")
        self.assertNotIn("docs/topic/page.md", term_text)

    def test_fix_leaves_plain_text_mentions_for_agent_judgment(self) -> None:
        self.write_term(
            appears_in="",
            mention_aliases=("key-value cache",),
            mention_lint="aliases",
        )
        self.write("docs/topic/page.md", "A key-value cache stores state.\n")

        fixes = TERM_LINKS.apply_safe_fixes(self.root, updated="2026-08-10")

        self.assertEqual(fixes, [])
        self.assertIn(
            "unlinked-term-mention", self.kinds(review_paths=())
        )

    def test_plain_mention_warns_and_strict_mode_fails(self) -> None:
        self.write_term(
            appears_in="",
            mention_aliases=("key-value cache",),
            mention_lint="aliases",
        )
        self.write("docs/topic/page.md", "A key-value cache stores state.\n")

        self.assertNotIn("unlinked-term-mention", self.kinds())

        issues = TERM_LINKS.validate(self.root, review_paths=())
        mention = next(
            issue for issue in issues if issue.kind == "unlinked-term-mention"
        )
        self.assertEqual(mention.severity, "warning")

        strict_issues = TERM_LINKS.validate(self.root, strict_mentions=True)
        strict_mention = next(
            issue for issue in strict_issues if issue.kind == "unlinked-term-mention"
        )
        self.assertEqual(strict_mention.severity, "error")

    def test_aliases_are_not_detected_unless_opted_in(self) -> None:
        self.write_term(appears_in="")
        self.write("docs/topic/page.md", "A key-value cache stores state.\n")

        self.assertNotIn(
            "unlinked-term-mention", self.kinds(review_paths=())
        )

        self.write_term(appears_in="", mention_aliases=("key-value cache",))
        self.assertNotIn(
            "unlinked-term-mention", self.kinds(review_paths=())
        )

        self.write_term(
            appears_in="",
            mention_aliases=("key-value cache",),
            mention_lint="aliases",
        )
        self.assertIn(
            "unlinked-term-mention", self.kinds(review_paths=())
        )

    def test_mention_lint_off_disables_discovery(self) -> None:
        self.write_term(appears_in="", mention_lint="off")
        self.write("docs/topic/page.md", "A KV Cache stores state.\n")

        self.assertNotIn(
            "unlinked-term-mention", self.kinds(review_paths=())
        )

    def test_invalid_mention_lint_mode_fails(self) -> None:
        self.write_term(appears_in="", mention_lint="sometimes")

        self.assertIn("invalid-mention-lint", self.kinds())

    def test_existing_navigation_link_is_not_an_unlinked_mention(self) -> None:
        self.write_term(appears_in="")
        self.write(
            "docs/topic/page.md",
            "See [KV Cache design](another-page.md) for the detailed topic.\n",
        )

        self.assertNotIn(
            "unlinked-term-mention", self.kinds(review_paths=())
        )

    def test_mention_review_can_be_scoped_to_a_path(self) -> None:
        self.write_term(appears_in="")
        self.write("docs/topic/page.md", "A KV Cache stores state.\n")
        self.write("docs/other/page.md", "Another KV Cache stores state.\n")

        issues = TERM_LINKS.validate(
            self.root, review_paths=(Path("docs/topic"),)
        )
        mentions = [
            issue for issue in issues if issue.kind == "unlinked-term-mention"
        ]
        self.assertEqual(len(mentions), 1)
        self.assertEqual(mentions[0].path, "docs/topic/page.md")

    def test_mention_review_rejects_paths_outside_docs(self) -> None:
        self.write_term(appears_in="")

        with self.assertRaisesRegex(ValueError, "outside docs"):
            TERM_LINKS.validate(
                self.root, review_paths=(self.root.parent,)
            )

    def test_detectable_alias_must_also_be_registered_as_an_alias(self) -> None:
        self.write_term(appears_in="")
        term_path = self.root / "docs/terms/kv-cache.md"
        term_path.write_text(
            term_path.read_text(encoding="utf-8").replace(
                "mention_aliases:", "mention_aliases:\n  - attention memory"
            ),
            encoding="utf-8",
        )

        self.assertIn("unregistered-mention-alias", self.kinds())

    def test_reviewed_inline_ignore_suppresses_only_its_line(self) -> None:
        self.write_term(appears_in="")
        self.write(
            "docs/topic/page.md",
            "KV Cache names an unrelated fixture. "
            "<!-- termlint-ignore: kv-cache -- Fixture name, not the concept. -->\n",
        )

        self.assertNotIn(
            "unlinked-term-mention", self.kinds(review_paths=())
        )

        self.write(
            "docs/topic/page.md",
            "KV Cache names an unrelated fixture. "
            "<!-- termlint-ignore: kv-cache -- Fixture name, not the concept. -->\n"
            "A KV Cache stores attention state.\n",
        )
        self.assertIn(
            "unlinked-term-mention", self.kinds(review_paths=())
        )

    def test_comment_only_ignore_applies_to_previous_prose_line(self) -> None:
        self.write_term(appears_in="")
        self.write(
            "docs/topic/page.md",
            "KV Cache names an unrelated fixture.\n"
            "<!-- termlint-ignore: kv-cache -- Fixture name, not the concept. -->\n",
        )

        self.assertNotIn("unlinked-term-mention", self.kinds())

    def test_invalid_unknown_and_stale_ignores_fail(self) -> None:
        self.write_term(appears_in="")
        self.write(
            "docs/topic/page.md",
            "KV Cache fixture. <!-- termlint-ignore: kv-cache -->\n"
            "Text. <!-- termlint-ignore: missing-term -- Reviewed false positive. -->\n"
            "Text. <!-- termlint-ignore: kv-cache -- No matching mention here. -->\n",
        )

        kinds = self.kinds()
        self.assertIn("invalid-termlint-ignore", kinds)
        self.assertIn("unknown-termlint-ignore", kinds)
        self.assertIn("stale-termlint-ignore", kinds)

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
