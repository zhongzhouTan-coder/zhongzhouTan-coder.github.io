---
description: "Use when capturing an HTML page as immutable evidence or synthesizing docs from a web snapshot."
applyTo: "raw/**/*.html, raw/**/*.metadata.json, derived/web-markdown/**/*, scripts/ingestion/web_to_markdown.mjs, sources.json"
---

# Web Source Instructions

Apply this workflow to HTML snapshots. A PDF uses the PDF workflow; a code
repository uses the repository-reading workflow.

- [Capture](references/web-capture.md): renderer, immutable raw bundle,
  regeneration of derived Markdown.
- [Synthesis and validation](references/web-synthesis.md): provenance,
  figures, manifest transition, checks.

Read capture before saving a new web revision. Read synthesis and validation
before writing its docs page or changing its manifest status to `ingested`.
