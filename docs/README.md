---
title: "Knowledge Base Introduction"
summary: "Overview of this repository's documentation structure, category organization, and how to navigate or contribute to the knowledge base."
layout: default
confidence: high
sources:
  - AGENTS.md
updated: 2026-07-09
---

# Knowledge Base Introduction

This repository is an AI-maintained personal knowledge base published on GitHub Pages. Source material lives in `raw/`. Processed knowledge pages live in `docs/`. A searchable index and a chronological change log live in `logs/`.

## Directory Structure

- `raw/` — Source files. Never modified by the agent.
- `docs/` — AI-maintained markdown knowledge pages, divided into topic categories.
  - `benchmarks/` — Benchmark designs, tasks, metrics, and empirical findings.
  - `frameworks/` — LLM serving and language-model programming systems.
  - `algorithms/` — Inference algorithms and kernel-level methods.
  - `hardware/` — Numerics, hardware features, and accelerator-specific notes.
- `logs/`
  - `index.md` — Categorised index of all docs pages.
  - `log.md` — Chronological record of every ingest and edit.

## Categories

| Category | When to use |
|---|---|
| `benchmarks` | Papers or notes about benchmark construction, domains, metrics, task generation, or evaluation results. |
| `frameworks` | Runtime systems, serving engines, programming frameworks, and application orchestration systems. |
| `algorithms` | Inference algorithms, kernel implementations, scheduling methods, and mathematical procedures. |
| `hardware` | Hardware formats, accelerator features, precision recipes, and numerics-focused notes. |

Create a new category only when an existing one would make the page hard to find.

## How to Use

**Finding information**

1. Start with [logs/index.md](../logs/index.md) to browse by topic, person, concept, or source.
2. Follow links to the relevant docs page.
3. Check the page sources in the front matter to understand what evidence supports the content.

**Ingesting a new source**

1. Add the source file to `raw/`.
2. Follow the repo rules in `AGENTS.md` to read the source, create or update docs pages, and update `logs/index.md` and `logs/log.md`.

**Maintaining quality**

Run the docs lint check at any time:

```bash
./scripts/lint-docs.sh
```

Or invoke the `lint-docs-cleanup` Codex skill to have the agent fix safe issues and report deletion candidates.

## Related Pages

- [Wiki Index](../logs/index.md)
- [Wiki Log](../logs/log.md)
