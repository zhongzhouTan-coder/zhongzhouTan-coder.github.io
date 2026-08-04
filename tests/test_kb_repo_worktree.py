from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKTREE_SCRIPT = REPOSITORY_ROOT / "scripts/kb-repo-worktree.py"


class RepositoryWorktreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary_root = Path(self.temporary_directory.name)
        self.root = temporary_root / "wiki"
        self.author = temporary_root / "author"
        self.remote = temporary_root / "remote.git"
        self.root.mkdir()
        self.author.mkdir()

        self.git("init", "-q", "-b", "main", str(self.author))
        self.git("-C", str(self.author), "config", "user.name", "Test")
        self.git(
            "-C", str(self.author), "config", "user.email", "test@example.com"
        )
        (self.author / "src").mkdir()
        (self.author / "docs").mkdir()
        (self.author / "src/core.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.author / "docs/readme.md").write_text("initial\n", encoding="utf-8")
        self.commit("initial")
        self.pinned = self.revision(self.author)

        self.git("clone", "-q", "--bare", str(self.author), str(self.remote))
        self.git("-C", str(self.author), "remote", "add", "origin", str(self.remote))

        registry = {
            f"repo-{self.pinned[:12]}": {
                "local_checkout": f"external-repos/repo-{self.pinned[:12]}",
                "provider": "github",
                "repository_url": "https://github.com/owner/repo",
                "revision": self.pinned,
            }
        }
        sources = {
            "schema_version": 1,
            "sources": [
                {
                    "kind": "repository",
                    "provider": "github",
                    "repository_url": "https://github.com/owner/repo",
                    "revision": self.pinned,
                    "repo_slug": "repo",
                    "raw_paths": [
                        f"raw/frameworks/repo-codebase--github-{self.pinned[:12]}.md"
                    ],
                }
            ],
        }
        (self.root / "docs/_data").mkdir(parents=True)
        self.raw_path = (
            self.root
            / f"raw/frameworks/repo-codebase--github-{self.pinned[:12]}.md"
        )
        self.raw_path.parent.mkdir(parents=True)
        self.raw_path.write_text(
            f"---\ninspected: {date.today().isoformat()}\n---\n",
            encoding="utf-8",
        )
        (self.root / "docs/_data/code_repositories.json").write_text(
            json.dumps(registry), encoding="utf-8"
        )
        (self.root / "sources.json").write_text(
            json.dumps(sources), encoding="utf-8"
        )
        self.repository_key = f"repo-{self.pinned[:12]}"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )

    def revision(self, repository: Path) -> str:
        return self.git("-C", str(repository), "rev-parse", "HEAD").stdout.strip()

    def commit(self, message: str) -> None:
        self.git("-C", str(self.author), "add", ".")
        self.git("-C", str(self.author), "commit", "-qm", message)

    def push(self) -> None:
        self.git("-C", str(self.author), "push", "-q", "origin", "main")

    def set_inspected_days_ago(self, days: int) -> None:
        inspected = date.today() - timedelta(days=days)
        self.raw_path.write_text(
            f"---\ninspected: {inspected.isoformat()}\n---\n",
            encoding="utf-8",
        )

    def run_script(self, command: str, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "python3",
                str(WORKTREE_SCRIPT),
                command,
                self.repository_key,
                "--remote-url",
                str(self.remote),
                "--root",
                str(self.root),
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def run_batch_script(
        self, command: str, *extra: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "python3",
                str(WORKTREE_SCRIPT),
                command,
                "--root",
                str(self.root),
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_sync_reuses_revision_when_only_unrelated_paths_changed(self) -> None:
        (self.author / "docs/readme.md").write_text("updated\n", encoding="utf-8")
        self.commit("docs only")
        self.push()
        latest = self.revision(self.author)

        result = self.run_script("sync", "--path", "src/core.py")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("decision: reuse", result.stdout)
        self.assertFalse(
            (self.root / f"external-repos/repo-{latest[:12]}").exists()
        )

    def test_sync_materializes_latest_when_relevant_paths_changed(self) -> None:
        self.set_inspected_days_ago(14)
        (self.author / "src/core.py").write_text("VALUE = 2\n", encoding="utf-8")
        self.commit("change core")
        self.push()
        latest = self.revision(self.author)

        result = self.run_script("sync", "--path", "src/core.py")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("decision: new revision", result.stdout)
        checkout = self.root / f"external-repos/repo-{latest[:12]}"
        self.assertTrue(checkout.is_dir())
        self.assertEqual(self.revision(checkout), latest)

    def test_force_materializes_relevant_change_during_interval(self) -> None:
        (self.author / "src/core.py").write_text("VALUE = 2\n", encoding="utf-8")
        self.commit("change core")
        self.push()
        latest = self.revision(self.author)

        result = self.run_script(
            "sync", "--path", "src/core.py", "--force-new-revision"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("decision: new revision", result.stdout)
        checkout = self.root / f"external-repos/repo-{latest[:12]}"
        self.assertEqual(self.revision(checkout), latest)

    def test_sync_defers_relevant_change_during_revision_interval(self) -> None:
        (self.author / "src/core.py").write_text("VALUE = 2\n", encoding="utf-8")
        self.commit("change core")
        self.push()
        latest = self.revision(self.author)

        result = self.run_script("sync", "--path", "src/core.py")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("decision: defer", result.stdout)
        self.assertIn("minimum revision interval: 14 days", result.stdout)
        self.assertFalse(
            (self.root / f"external-repos/repo-{latest[:12]}").exists()
        )

    def test_zero_revision_interval_materializes_relevant_change(self) -> None:
        (self.author / "src/core.py").write_text("VALUE = 2\n", encoding="utf-8")
        self.commit("change core")
        self.push()
        latest = self.revision(self.author)

        result = self.run_script(
            "sync",
            "--path",
            "src/core.py",
            "--min-revision-interval-days",
            "0",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("decision: new revision", result.stdout)
        checkout = self.root / f"external-repos/repo-{latest[:12]}"
        self.assertEqual(self.revision(checkout), latest)

    def test_registered_revision_can_be_retired_and_restored(self) -> None:
        materialized = self.run_script("materialize")
        checkout = self.root / f"external-repos/repo-{self.pinned[:12]}"
        self.assertEqual(materialized.returncode, 0, materialized.stderr)
        self.assertTrue(checkout.is_dir())

        retired = self.run_script("retire")
        self.assertEqual(retired.returncode, 0, retired.stderr)
        self.assertFalse(checkout.exists())

        restored = self.run_script("materialize")
        self.assertEqual(restored.returncode, 0, restored.stderr)
        self.assertEqual(self.revision(checkout), self.pinned)

    def test_status_reports_workspace_readiness_without_network(self) -> None:
        missing = self.run_batch_script("status", "--json")
        self.assertEqual(missing.returncode, 0, missing.stderr)
        missing_rows = json.loads(missing.stdout)
        self.assertEqual(missing_rows[0]["status"], "not-materialized")
        self.assertEqual(missing_rows[0]["cache"], "missing")

        restored = self.run_script("materialize")
        self.assertEqual(restored.returncode, 0, restored.stderr)

        ready = self.run_batch_script("status", "--json")
        self.assertEqual(ready.returncode, 0, ready.stderr)
        ready_rows = json.loads(ready.stdout)
        self.assertEqual(ready_rows[0]["status"], "ready")
        self.assertEqual(ready_rows[0]["cache"], "ready")

    def test_materialize_all_restores_every_registered_revision(self) -> None:
        # Seed the cache through the test-only local remote, then retire the
        # worktree so materialize-all can run without network access.
        restored = self.run_script("materialize")
        self.assertEqual(restored.returncode, 0, restored.stderr)
        retired = self.run_script("retire")
        self.assertEqual(retired.returncode, 0, retired.stderr)

        result = self.run_batch_script(
            "materialize-all",
            self.repository_key,
            "--remote-url",
            str(self.remote),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        checkout = self.root / f"external-repos/repo-{self.pinned[:12]}"
        self.assertEqual(self.revision(checkout), self.pinned)


if __name__ == "__main__":
    unittest.main()
