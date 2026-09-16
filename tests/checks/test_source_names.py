from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NORMALIZE_SCRIPT = REPOSITORY_ROOT / "scripts/checks/source_names.py"


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
                        "topic_path": "foundations",
                        "kind": "paper",
                        "raw_paths": ["raw/algorithms/foundations/wrong.pdf"],
                        "derived_path": "derived/pdf-markdown/algorithms/foundations/wrong.md",
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
                "actual": "raw/algorithms/foundations/wrong.pdf",
                "expected": (
                    "raw/algorithms/foundations/example-paper--arxiv-1234.5678v1.pdf"
                ),
            },
        )
        self.assertEqual(result.stderr, "")

    def test_gitcode_repository_uses_provider_suffix(self) -> None:
        commit = "0123456789abcdef0123456789abcdef01234567"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "sources": [
                    {
                        "id": f"gitcode:owner/repo@{commit}",
                        "slug": "repo-codebase",
                        "repo_slug": "repo",
                        "provider": "gitcode",
                        "revision": commit,
                        "category": "frameworks",
                        "topic_path": "serving/repo",
                        "kind": "repository",
                        "raw_paths": ["raw/frameworks/serving/repo/wrong.md"],
                        "derived_path": (
                            f"derived/repo-analysis/frameworks/serving/repo/repo/{commit}/"
                        ),
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
        self.assertEqual(
            payload["mismatches"][0]["expected"],
            "raw/frameworks/serving/repo/repo-codebase--gitcode-0123456789ab.md",
        )


if __name__ == "__main__":
    unittest.main()
