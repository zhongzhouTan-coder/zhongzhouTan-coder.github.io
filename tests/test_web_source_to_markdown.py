from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INGEST_SCRIPT = REPOSITORY_ROOT / "scripts/web-source-to-markdown.mjs"
INTEGRITY_SCRIPT = REPOSITORY_ROOT / "scripts/kb-check-integrity.py"
NORMALIZE_SCRIPT = REPOSITORY_ROOT / "scripts/kb-normalize-source-name.py"
CAPTURED_AT = "2026-07-28T12:34:56.000Z"
ARTICLE_HTML = """<!doctype html>
<html>
  <head>
    <title>Automatic Web Capture</title>
    <link rel="canonical" href="/canonical-article">
    <meta name="author" content="Docs Author">
  </head>
  <body>
    <nav>Navigation that should not be primary content.</nav>
    <main>
      <article>
        <h1>Automatic Web Capture</h1>
        <p>This article explains a reproducible ingestion workflow.</p>
        <p>Read the <a href="/guide">related guide</a>.</p>
        <svg viewBox="0 0 200 100">
          <defs><style>.box { fill: red; }</style></defs>
          <title>Capture pipeline diagram</title>
          <rect class="box" x="5" y="5" width="80" height="40"></rect>
          <path d="M85 25 L115 25"></path>
          <rect x="115" y="5" width="80" height="40"></rect>
        </svg>
        <p><em>Figure 1. Capture pipeline diagram.</em></p>
        <img
          src="data:image/svg+xml,%3Csvg%3E%3C/svg%3E"
          data-src="/diagram.png"
          alt="A lazily loaded diagram"
        >
      </article>
    </main>
  </body>
</html>
"""


class WebSourceIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.write_json(
            "kb-categories.json",
            {
                "categories": {
                    "frameworks": {
                        "raw_prefix": "raw/frameworks/",
                        "derived_prefix": "derived/pdf-markdown/frameworks/",
                        "web_derived_prefix": "derived/web-markdown/frameworks/",
                        "repo_analysis_prefix": "derived/repo-analysis/frameworks/",
                        "docs_prefix": "docs/frameworks/",
                    }
                }
            },
        )
        self.write_json(
            "sources.json",
            {
                "schema_version": 1,
                "naming_policy": {},
                "sources": [],
            },
        )
        self.fixture = self.root / "article.html"
        self.fixture.write_text(ARTICLE_HTML, encoding="utf-8")
        self.url = "https://example.test/article"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_json(self, relative_path: str, value: object) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{json.dumps(value, indent=2)}\n", encoding="utf-8")

    def run_ingest(
        self, *extra_args: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "node",
                str(INGEST_SCRIPT),
                "--url",
                self.url,
                "--category",
                "frameworks",
                "--slug",
                "automatic-web-capture",
                "--renderer",
                "http",
                "--input-html",
                str(self.fixture),
                "--captured-at",
                CAPTURED_AT,
                "--root",
                str(self.root),
                *extra_args,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_capture_creates_immutable_bundle_and_manifest_entry(self) -> None:
        result = self.run_ingest()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "captured")

        raw_html = self.root / report["raw_html"]
        raw_metadata = self.root / report["raw_metadata"]
        derived_markdown = self.root / report["derived_markdown"]
        self.assertTrue(raw_html.is_file())
        self.assertTrue(raw_metadata.is_file())
        self.assertTrue(derived_markdown.is_file())
        self.assertEqual(len(report["derived_assets"]), 1)
        derived_asset = self.root / report["derived_assets"][0]
        self.assertTrue(derived_asset.is_file())
        asset_text = derived_asset.read_text(encoding="utf-8")
        self.assertIn("<svg", asset_text)
        self.assertIn("<style>", asset_text)
        self.assertIn("fill: red", asset_text)

        metadata = json.loads(raw_metadata.read_text(encoding="utf-8"))
        self.assertEqual(metadata["requested_url"], self.url)
        self.assertEqual(metadata["renderer"], "local-html")
        self.assertEqual(
            metadata["canonical_url"],
            "https://example.test/canonical-article",
        )
        markdown = derived_markdown.read_text(encoding="utf-8")
        self.assertIn("Automatic Web Capture", markdown)
        self.assertIn(
            "](https://example.test/guide)",
            markdown,
        )
        self.assertIn(
            f"]({derived_asset.parent.name}/{derived_asset.name})",
            markdown,
        )
        self.assertIn(
            "![A lazily loaded diagram](https://example.test/diagram.png)",
            markdown,
        )

        manifest = json.loads(
            (self.root / "sources.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest["sources"]), 1)
        entry = manifest["sources"][0]
        self.assertEqual(entry["kind"], "web")
        self.assertEqual(entry["status"], "captured")
        self.assertEqual(entry["revision"], metadata["content_sha256"])

        integrity = subprocess.run(
            ["python3", str(INTEGRITY_SCRIPT), "--root", str(self.root)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(integrity.returncode, 0, integrity.stderr)
        normalization = subprocess.run(
            ["python3", str(NORMALIZE_SCRIPT), "--root", str(self.root)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(normalization.returncode, 0, normalization.stdout)

    def test_same_capture_is_reused_without_overwrite(self) -> None:
        first = self.run_ingest()
        self.assertEqual(first.returncode, 0, first.stderr)
        first_report = json.loads(first.stdout)
        raw_html = self.root / first_report["raw_html"]
        original_mtime = raw_html.stat().st_mtime_ns

        second = self.run_ingest(
            "--captured-at",
            "2026-07-28T13:00:00.000Z",
        )
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(json.loads(second.stdout)["status"], "reused")
        self.assertEqual(raw_html.stat().st_mtime_ns, original_mtime)
        manifest = json.loads(
            (self.root / "sources.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest["sources"]), 1)

    def test_existing_snapshot_can_regenerate_derived_assets(self) -> None:
        first = self.run_ingest()
        self.assertEqual(first.returncode, 0, first.stderr)
        report = json.loads(first.stdout)
        derived_markdown = self.root / report["derived_markdown"]
        derived_markdown.write_text("outdated\n", encoding="utf-8")

        regeneration = subprocess.run(
            [
                "node",
                str(INGEST_SCRIPT),
                "--url",
                self.url,
                "--category",
                "frameworks",
                "--slug",
                "automatic-web-capture",
                "--renderer",
                "http",
                "--input-html",
                str(self.root / report["raw_html"]),
                "--captured-at",
                CAPTURED_AT,
                "--root",
                str(self.root),
                "--regenerate-derived",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(regeneration.returncode, 0, regeneration.stderr)
        self.assertEqual(json.loads(regeneration.stdout)["status"], "regenerated")
        regenerated = derived_markdown.read_text(encoding="utf-8")
        self.assertIn("Capture pipeline diagram", regenerated)
        self.assertIn(".assets/inline-01-", regenerated)

    def test_private_network_requires_explicit_opt_in(self) -> None:
        result = subprocess.run(
            [
                "node",
                str(INGEST_SCRIPT),
                "--url",
                "http://127.0.0.1/article",
                "--category",
                "frameworks",
                "--slug",
                "automatic-web-capture",
                "--renderer",
                "http",
                "--root",
                str(self.root),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("private IP URLs are blocked", result.stderr)

    def test_integrity_detects_modified_raw_html(self) -> None:
        result = self.run_ingest()
        self.assertEqual(result.returncode, 0, result.stderr)
        raw_html = self.root / json.loads(result.stdout)["raw_html"]
        raw_html.write_text(
            f"{raw_html.read_text(encoding='utf-8')}\n<!-- changed -->\n",
            encoding="utf-8",
        )
        integrity = subprocess.run(
            ["python3", str(INTEGRITY_SCRIPT), "--root", str(self.root)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(integrity.returncode, 0)
        self.assertIn("revision does not match raw HTML SHA-256", integrity.stderr)

    def test_ingested_source_requires_docs_to_cite_complete_bundle(self) -> None:
        result = self.run_ingest()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        manifest_path = self.root / "sources.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest["sources"][0]
        entry["docs_path"] = "docs/frameworks/automatic-web-capture.md"
        entry["status"] = "ingested"
        self.write_json("sources.json", manifest)

        sources = [
            report["raw_html"],
            report["raw_metadata"],
            report["derived_markdown"],
        ]
        source_lines = "\n".join(f"  - {source}" for source in sources)
        docs_path = self.root / entry["docs_path"]
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.write_text(
            f"""---
title: "Automatic Web Capture"
summary: "Test page backed by an immutable web snapshot."
layout: default
confidence: high
sources:
{source_lines}
updated: 2026-07-28
---

# Automatic Web Capture

Captured evidence.
""",
            encoding="utf-8",
        )
        integrity = subprocess.run(
            ["python3", str(INTEGRITY_SCRIPT), "--root", str(self.root)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(integrity.returncode, 0, integrity.stderr)

        docs_path.write_text(
            docs_path.read_text(encoding="utf-8").replace(
                f"  - {report['raw_metadata']}\n",
                "",
            ),
            encoding="utf-8",
        )
        integrity = subprocess.run(
            ["python3", str(INTEGRITY_SCRIPT), "--root", str(self.root)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(integrity.returncode, 0)
        self.assertIn("front matter missing source", integrity.stderr)


if __name__ == "__main__":
    unittest.main()
