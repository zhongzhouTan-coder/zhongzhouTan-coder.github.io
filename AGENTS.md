
# Personal docs Rules

## Directory structure

- raw/ stores original source files. Never modify them.
- docs/ stores AI-maintained markdown knowledge pages.
- logs/index.md stores the docs page index.
- logs/log.md stores chronological change logs.

## Writing rules

- Use markdown only.
- Keep each page focused on one topic.
- Add internal links whenever related pages already exist.
- Update existing pages instead of creating duplicates.
- If new information conflicts with old information, note the contradiction explicitly.
- We can draw some images with mermaid syntax, and prefer images when they clarify complex relationships or processes better than text alone.
- If a docs page references local sibling assets such as images or draw.io files, prefer a folder-backed page at `topic/index.md` so relative asset links render in both VS Code Markdown preview and GitHub Pages.

## Docs page requirements

- Apply these rules to every page under `docs/`.
- Every docs page must begin with Jekyll front matter using this schema:

```yaml
---
title: "Short page title"
summary: "One-sentence description of what the page covers."
layout: default
doc_layer: layer_0
confidence: high
sources:
  - raw/example-source.md
updated: 2026-05-28
---
```

- Required fields: `title`, `summary`, `layout`, `doc_layer`, `confidence`, `sources`, `updated`.
- `doc_layer` must match the containing folder: `layer_0`, `layer_1`, or `layer_2`.
- `confidence` must match the layer: `high` for `layer_0`, `medium` for `layer_1`, `low` for `layer_2`.
- `sources` must list the relevant `raw/` source files using repo-relative paths.
- `updated` must use `YYYY-MM-DD`.
- After the front matter, start the page body with one `#` heading that matches the page title in meaning.

## Logs requirements

- `logs/index.md` must keep the top-level title `# Wiki Index`.
- Keep `logs/index.md` grouped by category headings with concise bullets linking to pages in `docs/` using relative links.
- `logs/log.md` must keep the top-level title `# Wiki Log`.
- Append changes to `logs/log.md` in chronological order, using `## YYYY-MM-DD` headings and flat bullet lists under each date.
- Each log bullet should briefly state what changed, which source or topic it came from, and which docs page or layer was updated.

## Ingest workflow

When a new source is added:

1. Read the source from raw/
2. Create or update relevant pages in docs/
3. Update logs/index.md
4. Append a new entry to logs/log.md

## Query workflow

When answering questions:

1. Read logs/index.md first
2. Find relevant docs pages
3. Synthesize the answer from docs/
4. If the answer is valuable, save it as a new page in docs/
