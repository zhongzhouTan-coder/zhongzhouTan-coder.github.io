---
description: "Use when creating or updating docs pages under docs/. Enforces the standard Jekyll front matter schema, readable confidence metadata, topic-folder organization, and GitHub Pages markdown conventions for this repository."
applyTo: "docs/**/*.md"
---

# Docs Front Matter Rules

Use this instruction for every page under docs/.

## Required Front Matter

Every docs page must begin with Jekyll front matter using this schema:

```yaml
---
title: "Short page title"
summary: "One-sentence description of what the page covers."
layout: default
confidence: high
sources:
  - raw/example-source.md
updated: 2026-05-28
---
```

## Field Rules

- `title`: required. Keep it short, specific, and aligned with the page heading.
- `summary`: required. One sentence that explains the page's purpose.
- `layout`: required. Use `default` unless the repository later defines a docs-specific layout.
- `confidence`: required. Use `high`, `medium`, or `low` to record evidentiary confidence directly.
- `sources`: required. List the relevant `raw/` source files as relative paths.
- `updated`: required. Use the current date in `YYYY-MM-DD` format.

## Consistency Rules

- Keep `confidence` consistent with the evidence quality in the page body.
- Organize pages by topic folder such as `docs/benchmarks/`, `docs/frameworks/`, `docs/algorithms/`, or `docs/hardware/`; do not create topic-agnostic `docs/layer_*` folders.
- Within a category, group related pages into subcategory folders by project or theme (for example `docs/frameworks/vllm/`, `docs/algorithms/flashattention/`, `docs/benchmarks/agent-eval/`), each with a hub `index.md` that lists its pages. A page path must stay under the category prefix that matches its `sources.json` `docs_path`; do not relocate a page to a different category without also moving its `raw/` and `derived/` sources and updating the manifest.
- If a page's confidence changes, update the front matter and move the file only when its topic/category path also needs to change.
- If multiple raw files support the page, list all of them in `sources`.
- Do not invent source paths. Read the source files first.
- If a page has no direct raw source and only records an open question or contradiction, still include a `sources` list with the most relevant upstream material that motivated the note.

## Page Structure

- After the front matter, start the body with a single `#` heading that matches the `title` in meaning.
- Use relative markdown links for internal references.
- Keep markdown GitHub Pages friendly: no embedded scripts and no unnecessary
  raw HTML. Revision-aware repository code anchors are the standard exception;
  use the exact `code-link` structure from `repo-reading.instructions.md`.
- **Body content structure is defined in [`docs-content-structure.instructions.md`](docs-content-structure.instructions.md).** Follow that file for the required section order, cognitive principles, and self-test checklist when creating or updating paper-insight pages.

## Confidence Mapping

- Use `confidence: high` for direct, stable source-backed facts.
- Use `confidence: medium` for source-backed synthesis or interpretation.
- Use `confidence: low` for tentative notes, contradictions, or open questions.

## Exceptions

- Do not apply this schema to files outside `docs/`.
- If the repository later adds a different required Jekyll key, preserve this schema and extend it rather than replacing fields ad hoc.
