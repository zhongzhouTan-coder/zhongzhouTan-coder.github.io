---
name: mineru-doc-ingest
description: Use when ingesting new PDFs or other document sources into this repository's markdown knowledge base with scripts/mineru-agent-to-markdown.py, especially when converting raw/ files to derived/pdf-markdown/, synthesizing docs pages, updating docs/logs/index.md and docs/logs/log.md, or handling MinerU conversion failures.
---

# MinerU Doc Ingest

Use this workflow to turn a new source document into maintained knowledge-base pages.

## Required Repository Rules

Read these instruction files before writing docs or logs:

- `.github/instructions/docs-front-matter.instructions.md` for `docs/**/*.md` front matter, source paths, confidence, and topic placement.
- `.github/instructions/docs-content-structure.instructions.md` for paper-insight page body structure.
- `.github/instructions/logs-maintenance.instructions.md` before editing `docs/logs/index.md` or `docs/logs/log.md`.

## Ingest Workflow

1. Inspect `docs/logs/index.md` first to avoid duplicate topic pages.
2. Read the new source under `raw/`. Never modify files under `raw/`.
3. If the source is a PDF or document that needs extraction, convert it with:

```bash
scripts/mineru-agent-to-markdown.py --mode precise raw/<source-file>.pdf --output-dir derived/pdf-markdown/<topic-category>
```

Choose `<topic-category>` from the source topic or likely target docs topic, such as `ai`, `algorithms`, `benchmarks`, `hardware`, or `frameworks`. Do not write generated Markdown directly at the top level of `derived/pdf-markdown/`.

4. Treat the generated Markdown as the primary source for synthesis. Read it before writing or updating the docs page.
5. Create or update the best existing page under `docs/`; prefer updating an existing related page over creating a duplicate.
6. Cite the original `raw/` source path in docs front matter, not the generated Markdown path. Mention extraction limitations in the page body when they affect confidence.
7. Link related existing docs pages with relative internal links.
8. Update `docs/logs/index.md` so the page is discoverable.
9. Append a concise chronological entry to `docs/logs/log.md`.
10. Run `./scripts/lint-docs.sh` after docs or logs changes and fix safe findings.

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
