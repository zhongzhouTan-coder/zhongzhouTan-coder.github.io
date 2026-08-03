from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEARCH_SCRIPT = REPOSITORY_ROOT / "scripts/kb-search.py"


class KnowledgeBaseSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.write_doc(
            "docs/algorithms/paged-attention.md",
            "PagedAttention",
            "Paged KV-cache memory management.",
            "PagedAttention stores KV cache blocks in non-contiguous memory.",
        )
        self.write_doc(
            "docs/training/pipeline.md",
            "Pipeline Parallelism",
            "Training stages and microbatches.",
            "A pipeline schedule reduces idle bubbles during training.",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_doc(self, path: str, title: str, summary: str, body: str) -> None:
        full_path = self.root / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(
            f'---\ntitle: "{title}"\nsummary: "{summary}"\n---\n\n# {title}\n\n{body}\n',
            encoding="utf-8",
        )

    def run_search(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "python3",
                str(SEARCH_SCRIPT),
                *arguments,
                "--root",
                str(self.root),
                "--json",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_title_and_body_search_returns_structured_hit(self) -> None:
        result = self.run_search("paged kv cache")
        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["hits"][0]["path"], "docs/algorithms/paged-attention.md")
        self.assertIn("non-contiguous", payload["hits"][0]["snippet"])

    def test_category_filter_limits_the_corpus(self) -> None:
        result = self.run_search("pipeline", "--category", "training")
        payload = json.loads(result.stdout)
        self.assertEqual(payload["document_count"], 1)
        self.assertEqual(payload["hits"][0]["category"], "training")


if __name__ == "__main__":
    unittest.main()
