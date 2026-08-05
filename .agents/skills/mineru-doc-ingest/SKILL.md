---
name: mineru-doc-ingest
description: Use when ingesting new PDFs or other document sources into this repository's markdown knowledge base with scripts/ingestion/mineru_to_markdown.py, especially when converting raw/ files to derived/pdf-markdown/, synthesizing docs pages, updating docs/logs/index.md and docs/logs/log.md, or handling MinerU conversion failures.
---

# MinerU Doc Ingest

Use this workflow to turn a new source document into maintained knowledge-base pages.

## Required Repository Rules

Read these instruction files before writing docs or logs:

- `.github/instructions/docs-front-matter.instructions.md` for `docs/**/*.md` front matter, source paths, confidence, and topic placement.
- `.github/instructions/docs-content-structure.instructions.md` for paper-insight page body structure.
- `.github/instructions/logs-maintenance.instructions.md` before editing `docs/logs/index.md` or `docs/logs/log.md`.

## Ingest Workflow

1. Preflight the repository state:
   - Inspect `docs/logs/index.md` first to avoid duplicate topic pages.
   - Check the target topic folder under `docs/` for an existing page to update.
   - Check whether generated Markdown already exists under `derived/pdf-markdown/<topic-category>/<source>/`.
2. Read the new source under `raw/`. Never modify files under `raw/`.
3. If the source is a PDF or document that needs extraction and no complete generated Markdown exists, convert it with:

```bash
scripts/ingestion/mineru_to_markdown.py --mode precise raw/<source-file>.pdf --output-dir derived/pdf-markdown/<topic-category>
```

Choose `<topic-category>` from the source topic or likely target docs topic, such as `ai`, `algorithms`, `benchmarks`, `hardware`, or `frameworks`. Do not write generated Markdown directly at the top level of `derived/pdf-markdown/`.

4. Treat the generated Markdown as the primary source for synthesis. Read it before writing or updating the docs page.
5. **Handle useful images from the derived markdown.** When MinerU extracts figures that should appear in the docs page, follow this workflow:
   - **Select:** Review the images extracted by MinerU under `derived/pdf-markdown/<topic-category>/<source>/images/`. Choose figures that explain key mechanisms, architectures, or results referenced in the docs page. Prefer the paper's own architectural diagrams, pipeline figures, and headline result charts. Skip decorative or redundant images.
   - **Copy:** Copy selected images to a local `assets/` folder under the docs page directory (e.g., `docs/algorithms/deepseek-v3.2/assets/`). Use folder-backed pages (`topic/index.md`) when a page needs local assets.
   - **Rename:** Rename every copied image to a descriptive, human-readable kebab-case filename before referencing it (e.g., `dsa-architecture.jpg`, `thinking-retention-chart.jpg`). Do not keep generated names such as `image_1.jpg` or `fig_3.png`.
   - **Reference:** Embed each image in the docs page near the prose that explains it. Always add an italic caption below the image describing what the reader should see. Images without a prose reference and caption are orphaned — do not leave them unused in the assets folder.
   - **Do not** reference images directly from `derived/pdf-markdown/` paths. Always copy to the `docs/` tree first so the page is self-contained and lint-able.
   - **Skip cleanly:** If no extracted image is useful, do not copy images just to satisfy the workflow. Use Mermaid, Draw.io, or prose instead when that better explains the material.
6. Create or update the best existing page under `docs/`; prefer updating an existing related page over creating a duplicate.
7. Cite the original `raw/` source path in docs front matter, not the generated Markdown path. Mention extraction limitations in the page body when they affect confidence.
8. Link related existing docs pages with relative internal links.
9. Update related pages only when the new material changes a claim, adds a needed comparison, supersedes older context, or deserves a short cross-link. Do not rewrite related pages just because a new page exists.
10. Update `docs/logs/index.md` so the page is discoverable.
11. Append a concise chronological entry to `docs/logs/log.md`.
12. Run `./scripts/lint-docs.sh` after docs or logs changes and fix safe findings (including orphan image warnings).

## MinerU Command Notes

Use `--mode precise` by default for new PDF ingest. It requires `MINERU_API_TOKEN` or `--token`.

Useful flags:

- `--language en` for English papers.
- `--language ch` for Chinese sources or when language is unknown.
- `--ocr` for scanned documents.
- `--page-range 1-10` for partial extraction.
- `--model-version vlm` unless a source clearly needs another supported MinerU model.
- `--use-proxy` only when the environment requires proxy variables.

If MinerU is unavailable, rate-limited, blocked by SSL/network issues, or clearly incomplete:

1. Check whether a generated Markdown file already exists under `derived/pdf-markdown/`.
2. Use the existing generated Markdown if it is complete enough.
3. Use a fallback extractor only after checking for generated Markdown.
4. Record the conversion limitation in the resulting docs page and set confidence accordingly.

## Output Expectations

When finishing an ingest task, report:

- the source file processed
- the generated Markdown path, if any
- the docs page created or updated
- the logs updated
- whether `./scripts/lint-docs.sh` passed or what remains
