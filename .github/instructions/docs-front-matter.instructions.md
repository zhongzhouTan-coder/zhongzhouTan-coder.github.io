---
description: "Use when creating or updating docs pages under docs/. Enforces the standard Jekyll front matter schema, confidence-layer metadata, and GitHub Pages markdown conventions for this repository."
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
doc_layer: layer_0
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
- `doc_layer`: required. Must match the folder that contains the page: `layer_0`, `layer_1`, or `layer_2`.
- `confidence`: required. Use `high` for `layer_0`, `medium` for `layer_1`, and `low` for `layer_2`.
- `sources`: required. List the relevant `raw/` source files as relative paths.
- `updated`: required. Use the current date in `YYYY-MM-DD` format.

## Consistency Rules

- Keep `doc_layer` and `confidence` consistent with the evidence quality in the page body.
- If a page moves to a different layer, update both the path and the front matter.
- If multiple raw files support the page, list all of them in `sources`.
- Do not invent source paths. Read the source files first.
- If a page has no direct raw source and only records an open question or contradiction, still include a `sources` list with the most relevant upstream material that motivated the note.

## Page Structure

- After the front matter, start the body with a single `#` heading that matches the `title` in meaning.
- Prefer concise sections such as `## Summary`, `## Evidence`, `## Open Questions`, `## Related Pages`, and `## Source Notes` when useful.
- Use relative markdown links for internal references.
- Keep markdown GitHub Pages friendly: no embedded scripts and no unnecessary raw HTML.

## Layer Mapping

- Pages in `docs/layer_0/` must use `doc_layer: layer_0` and `confidence: high`.
- Pages in `docs/layer_1/` must use `doc_layer: layer_1` and `confidence: medium`.
- Pages in `docs/layer_2/` must use `doc_layer: layer_2` and `confidence: low`.

## Exceptions

- Do not apply this schema to files outside `docs/`.
- If the repository later adds a different required Jekyll key, preserve this schema and extend it rather than replacing fields ad hoc.
