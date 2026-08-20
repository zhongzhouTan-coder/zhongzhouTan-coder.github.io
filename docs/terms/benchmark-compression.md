---
title: "Benchmark Compression"
summary: "Selection of a small evaluation coreset that preserves the full benchmark's scores, rankings, or other target statistics."
tooltip: "Benchmark compression replaces a full evaluation set with a much smaller calibrated coreset. Its central risk is distribution shift: a subset that reconstructs historical models may stop representing future systems."
layout: default
confidence: medium
category: benchmarks
sources:
  - raw/benchmarks/rethinking-llm-evaluation-200x-less-data--arxiv-2510.10457v1.pdf
  - raw/benchmarks/autojudger-agent-driven-mllm-benchmarking--arxiv-2505.21389v1.pdf
aliases:
  - benchmark coreset selection
  - compressed benchmark
mention_aliases:
  - benchmark coreset selection
mention_lint: canonical
appears_in:
  - docs/benchmarks/agent-eval/essencebench/index.md
  - docs/benchmarks/agent-eval/autojudger.md
updated: 2026-08-19
---

# Benchmark Compression

**Benchmark Compression** is the selection of a small evaluation coreset whose results preserve a full benchmark's target statistics, such as model scores or rankings.

## Why It Exists

As benchmark suites add tasks and samples, evaluating every new model becomes expensive in accelerator time, tokens, and judge calls. Many items also carry overlapping semantic or behavioral information, so a carefully calibrated subset can be more cost-effective than repeatedly running the full set.

## How It Works

A compression method first defines what must survive: absolute score, pairwise ordering, top-model retrieval, capability coverage, or a combination. It then selects a fixed subset or adaptively chooses items and validates the reduced evaluation against full-benchmark outcomes from a reference model population.

EssenceBench learns a fixed coreset by removing redundant items and searching for subsets that reconstruct historical full scores. AutoJudger instead conducts a model-specific adaptive interview, selecting each next item from estimated ability, difficulty, and semantic coverage. Both reduce evaluated items, but only the first publishes one common subset for every model.

## Tradeoffs

Compression shifts work from repeated evaluation into calibration and validation. A fixed coreset is cheap and comparable across models but can become stale or overfit; adaptive testing can spend questions where they are most informative but gives different models different item paths. Neither approach establishes validity under a new model family or task format unless that distribution shift is tested explicitly.

## Common Confusions

- **Benchmark compression vs. training-data pruning:** Compression reduces evaluation cost while trying to preserve measurement; pruning changes the data used to train a model.
- **Compression ratio vs. retained fraction:** “200x compression” means retaining $1/200$ of the original items, not removing 200%.
- **Score reconstruction vs. capability coverage:** A subset can predict aggregate leaderboard scores while omitting rare behaviors that matter diagnostically.

## Where It Appears

- [EssenceBench](../benchmarks/agent-eval/essencebench/index.md) — Learns one fixed coreset through redundancy filtering, genetic search, and attribution-guided refinement.
- [AutoJudger](../benchmarks/agent-eval/autojudger.md) — Contrasting adaptive approach that selects a personalized sequence of informative questions for each evaluated model.
