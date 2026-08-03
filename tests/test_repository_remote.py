from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from repository_remote import parse_repository_remote  # noqa: E402


class RepositoryRemoteTests(unittest.TestCase):
    def test_normalizes_gitcode_scp_origin(self) -> None:
        remote = parse_repository_remote(
            "git@gitcode.com:cann/cannbot-skills.git"
        )

        self.assertEqual(remote.provider, "gitcode")
        self.assertEqual(remote.repository_path, "cann/cannbot-skills")
        self.assertEqual(
            remote.repository_url,
            "https://gitcode.com/cann/cannbot-skills",
        )

    def test_normalizes_github_https_origin(self) -> None:
        remote = parse_repository_remote(
            "https://github.com/vllm-project/vllm.git"
        )

        self.assertEqual(remote.provider, "github")
        self.assertEqual(
            remote.repository_url,
            "https://github.com/vllm-project/vllm",
        )

    def test_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported repository host"):
            parse_repository_remote("git@example.com:owner/repo.git")


if __name__ == "__main__":
    unittest.main()
