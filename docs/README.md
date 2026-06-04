---
title: "Knowledge Base Introduction"
summary: "Overview of this repository's documentation structure, confidence layers, and how to navigate or contribute to the knowledge base."
layout: default
doc_layer: layer_0
confidence: high
sources:
  - AGENTS.md
updated: 2026-05-29
---

# Knowledge Base Introduction

This repository is an AI-maintained personal knowledge base published on GitHub Pages. Source material lives in `raw/`. Processed knowledge pages live in `docs/`. A searchable index and a chronological change log live in `logs/`.

## Directory Structure

- `raw/` — Source files. Never modified by the agent.
- `docs/` — AI-maintained markdown knowledge pages, divided into confidence layers.
  - `layer_0/` — High-confidence facts directly supported by source material.
  - `layer_1/` — Medium-confidence synthesis and summaries that involve interpretation.
  - `layer_2/` — Low-confidence notes, open questions, contradictions, and tentative conclusions.
- `logs/`
  - `index.md` — Categorised index of all docs pages.
  - `log.md` — Chronological record of every ingest and edit.

## Confidence Layers

| Layer | Confidence | When to use |
|---|---|---|
| `layer_0` | High | Fact is directly and unambiguously supported by a raw source. |
| `layer_1` | Medium | Synthesis or summary that is likely correct but involves interpretation. |
| `layer_2` | Low | Open question, contradiction, tentative conclusion, or information that still needs confirmation. |

When evidence quality improves, pages move to a higher layer. When a page mixes confidence levels, it is either kept in the lowest required layer or split into separate pages.

## How to Use

**Finding information**

1. Start with [logs/index.md](../logs/index.md) to browse by topic, person, concept, or source.
2. Follow links to the relevant docs page.
3. Check the `doc_layer` and `confidence` fields in the front matter to understand how much to rely on the content.

**Ingesting a new source**

1. Add the source file to `raw/`.
2. Use the Docs Ingest Agent to read the source, create or update docs pages, and update `logs/index.md` and `logs/log.md`.

**Maintaining quality**

Run the docs lint check at any time:

```bash
./scripts/lint-docs.sh
```

Or invoke the `lint-docs-cleanup` Codex skill to have the agent fix safe issues and report deletion candidates.

## Related Pages

- [Wiki Index](../logs/index.md)
- [Wiki Log](../logs/log.md)
