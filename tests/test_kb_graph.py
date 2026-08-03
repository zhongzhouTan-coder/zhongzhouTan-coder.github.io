from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GRAPH_SCRIPT = REPOSITORY_ROOT / "scripts/kb-graph.py"


class KnowledgeBaseGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.write("docs/logs/index.md", "[Algorithms](../algorithms/index.md)\n")
        self.write(
            "docs/algorithms/index.md",
            "[Alpha](alpha.md)\n[Beta](<beta.md#details>)\n",
        )
        self.write("docs/algorithms/alpha.md", "[Beta](beta.md)\n")
        self.write("docs/algorithms/beta.md", "# Beta\n")
        self.write("docs/training/orphan.md", "# Orphan\n")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write(self, path: str, content: str) -> None:
        full_path = self.root / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    def test_graph_reports_navigation_and_disconnected_orphan(self) -> None:
        result = subprocess.run(
            [
                "python3",
                str(GRAPH_SCRIPT),
                "--root",
                str(self.root),
                "--json",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["total_pages"], 5)
        self.assertEqual(payload["total_edges"], 4)
        self.assertEqual(payload["component_count"], 2)
        self.assertEqual(payload["orphans"], ["docs/training/orphan.md"])
        self.assertEqual(
            payload["sinks"],
            ["docs/algorithms/beta.md", "docs/training/orphan.md"],
        )


if __name__ == "__main__":
    unittest.main()
