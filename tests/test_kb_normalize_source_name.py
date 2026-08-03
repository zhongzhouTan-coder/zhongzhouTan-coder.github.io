from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NORMALIZE_SCRIPT = REPOSITORY_ROOT / "scripts/kb-normalize-source-name.py"


class NormalizeSourceNameTests(unittest.TestCase):
    def test_json_output_reports_expected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "sources": [
                    {
                        "id": "arxiv:1234.5678v1",
                        "slug": "example-paper",
                        "category": "algorithms",
                        "kind": "paper",
                        "raw_paths": ["raw/algorithms/wrong.pdf"],
                        "derived_path": "derived/pdf-markdown/algorithms/wrong.md",
                    }
                ]
            }
            (root / "sources.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    "python3",
                    str(NORMALIZE_SCRIPT),
                    "--root",
                    str(root),
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["mismatch_count"], 2)
        self.assertEqual(
            payload["mismatches"][0],
            {
                "kind": "raw",
                "actual": "raw/algorithms/wrong.pdf",
                "expected": (
                    "raw/algorithms/example-paper--arxiv-1234.5678v1.pdf"
                ),
            },
        )
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
