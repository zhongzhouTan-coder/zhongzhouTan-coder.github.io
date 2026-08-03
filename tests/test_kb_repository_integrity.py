from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INTEGRITY_SCRIPT = REPOSITORY_ROOT / "scripts/kb-check-integrity.py"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
SOURCE_ID = f"github:owner/repo@{COMMIT}"
RAW_PATH = "raw/frameworks/repo-codebase--github-0123456789ab.md"
DERIVED_PATH = f"derived/repo-analysis/frameworks/repo/{COMMIT}/"
ANALYSIS_PATH = f"{DERIVED_PATH}important-files.md"


class RepositoryIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        categories = {
            "categories": {
                "frameworks": {
                    "raw_prefix": "raw/frameworks/",
                    "derived_prefix": "derived/pdf-markdown/frameworks/",
                    "repo_analysis_prefix": "derived/repo-analysis/frameworks/",
                    "docs_prefix": "docs/frameworks/",
                },
                "algorithms": {
                    "raw_prefix": "raw/algorithms/",
                    "derived_prefix": "derived/pdf-markdown/algorithms/",
                    "repo_analysis_prefix": "derived/repo-analysis/algorithms/",
                    "docs_prefix": "docs/algorithms/",
                },
            }
        }
        self.write_json("kb-categories.json", categories)
        self.entry = {
            "id": SOURCE_ID,
            "title": "Repo Codebase",
            "slug": "repo-codebase",
            "repo_slug": "repo",
            "revision": COMMIT,
            "category": "frameworks",
            "kind": "repository",
            "raw_paths": [RAW_PATH],
            "derived_path": DERIVED_PATH,
            "docs_paths": [
                "docs/frameworks/repo.md",
                "docs/algorithms/repo-kernel.md",
            ],
            "status": "ingested",
        }
        self.write_manifest()
        self.write(
            RAW_PATH,
            f"""---
kind: repository-source
repository_url: https://github.com/owner/repo
local_checkout: external-repos/repo/
commit: {COMMIT}
ref: main
inspected: 2026-07-28
checkout_state: clean
---

# Repo Source Record

## Reading Scope

- Runtime.

## Important Entry Files

- `src/main.py` — entry point.

## Limitations

- Static reading.
""",
        )
        self.write(
            ANALYSIS_PATH,
            f"""---
kind: repository-analysis
repository_id: {SOURCE_ID}
commit: {COMMIT}
source_record: {RAW_PATH}
generated: 2026-07-28
---

# Important Files

- `src/main.py` — entry point.
""",
        )
        for docs_path in self.entry["docs_paths"]:
            self.write_docs(docs_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write(self, relative_path: str, content: str) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def write_json(self, relative_path: str, value: object) -> None:
        self.write(relative_path, json.dumps(value, indent=2))

    def write_manifest(self) -> None:
        self.write_json(
            "sources.json",
            {"schema_version": 1, "sources": [self.entry]},
        )

    def write_docs(
        self,
        docs_path: str,
        *,
        confidence: str = "high",
        include_raw: bool = True,
        include_commit: bool = True,
    ) -> None:
        sources = [ANALYSIS_PATH]
        if include_raw:
            sources.insert(0, RAW_PATH)
        source_lines = "\n".join(f"  - {source}" for source in sources)
        commit_line = f"**Commit:** `{COMMIT}`" if include_commit else "No revision."
        self.write(
            docs_path,
            f"""---
title: "Repo"
summary: "Repository-backed test page."
layout: default
confidence: {confidence}
sources:
{source_lines}
updated: 2026-07-28
---

# Repo

{commit_line}
""",
        )

    def run_integrity(
        self, *, json_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        command = [
            "python3",
            str(INTEGRITY_SCRIPT),
            "--root",
            str(self.root),
        ]
        if json_output:
            command.append("--json")
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )

    def assert_integrity_error(self, expected: str) -> None:
        result = self.run_integrity()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(expected, result.stderr)

    def test_valid_revision_supports_multiple_categories(self) -> None:
        result = self.run_integrity()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_json_output_is_structured_for_errors(self) -> None:
        (self.root / ANALYSIS_PATH).unlink()
        result = self.run_integrity(json_output=True)
        payload = json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_count"], len(payload["errors"]))
        self.assertTrue(
            any("missing required repo analysis file" in error for error in payload["errors"])
        )
        self.assertEqual(result.stderr, "")

    def test_raw_commit_must_match_manifest_id(self) -> None:
        raw_file = self.root / RAW_PATH
        raw_file.write_text(
            raw_file.read_text(encoding="utf-8").replace(
                f"commit: {COMMIT}",
                f"commit: {'f' * 40}",
            ),
            encoding="utf-8",
        )
        self.assert_integrity_error("raw metadata commit must be")

    def test_important_files_is_required(self) -> None:
        (self.root / ANALYSIS_PATH).unlink()
        self.assert_integrity_error("missing required repo analysis file")

    def test_every_consumer_must_cite_raw_revision(self) -> None:
        self.write_docs(
            "docs/algorithms/repo-kernel.md",
            include_raw=False,
        )
        self.assert_integrity_error("front matter missing source")

    def test_citing_page_must_be_listed_as_consumer(self) -> None:
        self.write_docs("docs/frameworks/unlisted.md")
        self.assert_integrity_error("but is missing from docs_paths")

    def test_full_commit_must_appear_near_top(self) -> None:
        self.write_docs(
            "docs/frameworks/repo.md",
            include_commit=False,
        )
        self.assert_integrity_error("must state full commit near the top")

    def test_dirty_revision_requires_low_confidence(self) -> None:
        raw_file = self.root / RAW_PATH
        raw_file.write_text(
            raw_file.read_text(encoding="utf-8").replace(
                "checkout_state: clean",
                "checkout_state: dirty",
            ),
            encoding="utf-8",
        )
        self.assert_integrity_error("dirty checkout requires confidence low")


if __name__ == "__main__":
    unittest.main()
