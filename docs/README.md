---
title: "Knowledge Base Introduction"
summary: "Overview of this repository's documentation structure, category organization, and how to navigate or contribute to the knowledge base."
layout: default
confidence: high
sources:
  - AGENTS.md
updated: 2026-07-14
---

# Knowledge Base Introduction

This repository is an AI-maintained personal knowledge base published on GitHub Pages. Source material lives in `raw/`. Processed knowledge pages live in `docs/`. A searchable index and a chronological change log live in `logs/`.

## Directory Structure

- `raw/` — Canonical source files. File content is not modified; filenames and category paths are maintained for retrieval.
- `derived/pdf-markdown/` — Generated Markdown extracted from PDFs, organized under the same category names as `raw/` and `docs/`.
- `docs/` — AI-maintained markdown knowledge pages, divided into topic categories, each with subcategory folders for related projects or themes.
  - `benchmarks/` — Benchmark designs, tasks, metrics, and empirical findings; subfolders `agent-eval/` and `serving-perf/`.
  - `frameworks/` — LLM serving and language-model programming systems; per-project subfolders such as `vllm/`, `vllm-ascend/`, `triton/`, `triton-ascend/`, `sglang/`, `dspark/`, `harbor/`, and `deepseek/`.
  - `algorithms/` — Inference algorithms and kernel-level methods; subfolders `foundations/`, `flashattention/`, `attention-variants/`, and `linear-attention/`.
  - `training/` — Model training, fine-tuning, transfer learning, and generalization; subfolders `parallelism/`, `foundation-models/`, `deepseek/`, `kimi/`, `efficient-attention/`, and `fine-tuning/`.
  - `hardware/` — Numerics, hardware features, and accelerator-specific notes; subfolder `quantization/`.
- `sources.json` — Machine-readable manifest linking raw sources, generated Markdown, and docs pages.
- `kb-categories.json` — Canonical category registry and legacy category aliases.
- `logs/`
  - `index.md` — Categorised index of all docs pages.
  - `log.md` — Chronological record of every ingest and edit.

## Categories

| Category | When to use |
|---|---|
| `benchmarks` | Papers or notes about benchmark construction, domains, metrics, task generation, or evaluation results. |
| `frameworks` | Runtime systems, serving engines, programming frameworks, and application orchestration systems. |
| `algorithms` | Inference algorithms, kernel implementations, scheduling methods, and mathematical procedures. |
| `training` | Training dynamics, fine-tuning methods, transfer learning, and generalization analysis. |
| `hardware` | Hardware formats, accelerator features, precision recipes, and numerics-focused notes. |

Create a new category only when an existing one would make the page hard to find.

## How to Use

### Finding information

1. Start with [logs/index.md](logs/index.md) to browse by topic, person, concept, or source.
2. Follow links to the relevant docs page.
3. Check the page sources in the front matter to understand what evidence supports the content.

### Ingesting a new source

1. Choose one canonical category from `kb-categories.json`.
2. Name the source with the deterministic source naming policy, for example `raw/algorithms/flashattention-io-aware-exact-attention--arxiv-2205.14135v2.pdf`.
3. Add or update the corresponding `sources.json` entry.
4. For PDFs, generate Markdown under the matching `derived/pdf-markdown/{category}/` directory.
5. Follow the repo rules in `AGENTS.md` to read the source, create or update docs pages, and update `logs/index.md` and `logs/log.md`.

### Maintaining quality

Run the docs lint check at any time:

```bash
./scripts/lint-docs.sh
```

For source naming and manifest checks:

```bash
python3 scripts/checks/source_names.py
python3 scripts/checks/repository_integrity.py
```

Or invoke the `lint-docs-cleanup` Codex skill to have the agent fix safe issues and report deletion candidates.

## Related Pages

- [Wiki Index](logs/index.md)
- [Wiki Log](logs/log.md)
