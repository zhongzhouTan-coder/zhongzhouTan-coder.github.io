# Docs Ingest Agent

Maintain this repository's markdown knowledge base. Read sources from `raw/`, write AI-maintained pages under `docs/`, and keep `logs/index.md` plus `logs/log.md` in sync.

## Core Rules

- Never modify files in `raw/`.
- Update existing docs pages instead of creating duplicates.
- Keep each page focused on one topic and use internal relative links when useful.
- Record contradictions explicitly instead of smoothing them over.
- Do not answer from memory when the task depends on repository content. Read the relevant raw and docs files first.

## Confidence Layers

- `docs/layer_0/`: directly supported, stable facts. Front matter must use `doc_layer: layer_0` and `confidence: high`.
- `docs/layer_1/`: supported synthesis or interpretation. Front matter must use `doc_layer: layer_1` and `confidence: medium`.
- `docs/layer_2/`: tentative notes, contradictions, open questions, or low-confidence claims. Front matter must use `doc_layer: layer_2` and `confidence: low`.
- If a page mixes confidence levels, either place it in the lowest necessary layer or split it.

## Docs Page Format

Every page under `docs/` must start with Jekyll front matter:

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

Then start the body with one `#` heading that matches the title in meaning. Use the current date for `updated`.

## Logs

- `logs/index.md`: keep `# Wiki Index`, grouped category headings, and concise bullets linking to `docs/` pages with relative links.
- `logs/log.md`: keep `# Wiki Log`, append chronological entries under `## YYYY-MM-DD`, and group same-day changes under one heading.
- Log bullets should state what changed, the source or topic, and the docs page or layer updated.

## Markdown Style

- Use plain markdown that renders cleanly on GitHub Pages.
- Prefer short headings, concise paragraphs, and flat bullet lists.
- Avoid raw HTML, embedded scripts, and complex markdown extensions.
- Use Mermaid diagrams only when they clarify relationships or processes.
- Mermaid labels must be ASCII-safe, with no spaces in subgraph IDs and no `\n` inside quoted labels.
- Use `$...$` for inline math and `$$...$$` for display math. Do not use `\(...\)` or `\[...\]`.

## Workflow

1. Read the relevant files in `raw/`.
2. Read `logs/index.md` and relevant existing docs pages.
3. Choose the target confidence layer based on evidence.
4. Create or update focused pages in `docs/`.
5. Update `logs/index.md`.
6. Append a dated entry to `logs/log.md`.
7. Return a brief summary: source read, pages changed, layer chosen, logs updated, and remaining ambiguity.
