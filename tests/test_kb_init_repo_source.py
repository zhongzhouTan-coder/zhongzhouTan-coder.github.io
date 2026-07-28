from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD_SCRIPT = REPOSITORY_ROOT / "scripts/kb-init-repo-source.py"


class RepositorySourceScaffoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.checkout = self.root / "external-repos/repo"
        self.checkout.mkdir(parents=True)
        (self.root / ".gitignore").write_text(
            "external-repos/\n",
            encoding="utf-8",
        )
        subprocess.run(
            ["git", "init", "-q", str(self.root)],
            check=True,
        )
        subprocess.run(
            ["git", "init", "-q", str(self.checkout)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.checkout), "config", "user.name", "Test"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.checkout),
                "config",
                "user.email",
                "test@example.com",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.checkout),
                "remote",
                "add",
                "origin",
                "git@github.com:owner/repo.git",
            ],
            check=True,
        )
        (self.checkout / "main.py").write_text("print('ok')\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(self.checkout), "add", "main.py"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.checkout), "commit", "-qm", "initial"],
            check=True,
        )
        self.commit = subprocess.run(
            ["git", "-C", str(self.checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        categories = {
            "categories": {
                "frameworks": {
                    "raw_prefix": "raw/frameworks/",
                    "derived_prefix": "derived/pdf-markdown/frameworks/",
                    "repo_analysis_prefix": "derived/repo-analysis/frameworks/",
                    "docs_prefix": "docs/frameworks/",
                }
            }
        }
        (self.root / "kb-categories.json").write_text(
            json.dumps(categories),
            encoding="utf-8",
        )
        (self.root / "sources.json").write_text(
            json.dumps({"schema_version": 1, "sources": []}, indent=2),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_scaffold(
        self, docs_path: str = "docs/frameworks/repo.md"
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "python3",
                str(SCAFFOLD_SCRIPT),
                "external-repos/repo",
                "--category",
                "frameworks",
                "--docs-path",
                docs_path,
                "--scope",
                "Runtime entry point",
                "--important-file",
                "main.py::Executable entry point",
                "--root",
                str(self.root),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_create_then_reuse_revision_for_another_page(self) -> None:
        first = self.run_scaffold()
        self.assertEqual(first.returncode, 0, first.stderr)
        short_sha = self.commit[:12]
        raw_path = (
            self.root
            / f"raw/frameworks/repo-codebase--github-{short_sha}.md"
        )
        analysis_path = (
            self.root
            / "derived/repo-analysis/frameworks/repo"
            / self.commit
            / "important-files.md"
        )
        self.assertTrue(raw_path.is_file())
        self.assertTrue(analysis_path.is_file())
        original_raw = raw_path.read_text(encoding="utf-8")

        second = self.run_scaffold("docs/algorithms/repo-kernel.md")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("reused github:owner/repo@", second.stdout)
        self.assertEqual(raw_path.read_text(encoding="utf-8"), original_raw)

        manifest = json.loads(
            (self.root / "sources.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest["sources"]), 1)
        self.assertEqual(
            manifest["sources"][0]["docs_paths"],
            [
                "docs/frameworks/repo.md",
                "docs/algorithms/repo-kernel.md",
            ],
        )

    def test_dirty_checkout_is_rejected_by_default(self) -> None:
        (self.checkout / "untracked.py").write_text("", encoding="utf-8")
        result = self.run_scaffold()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checkout is dirty", result.stderr)


if __name__ == "__main__":
    unittest.main()
