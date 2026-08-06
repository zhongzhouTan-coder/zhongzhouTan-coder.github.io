---
title: "Chunked Prefill"
summary: "Splitting a long LLM prompt into causal chunks processed over multiple iterations so prompt work can share batches with decode work."
tooltip: "Chunked prefill divides one prompt into smaller causal pieces. It preserves the same KV-cache result while letting a scheduler interleave prompt work with decode tokens; the chunk size trades prefill efficiency against scheduling flexibility."
layout: default
confidence: high
category: frameworks
sources:
  - raw/frameworks/sarathi-efficient-llm-inference-with-chunked-prefills-2308.16369v1--arxiv-2308.16369v1.pdf
  - derived/pdf-markdown/frameworks/sarathi-efficient-llm-inference-with-chunked-prefills-2308.16369v1.md
aliases:
  - chunked-prefills
appears_in:
  - docs/frameworks/sarathi/index.md
  - docs/frameworks/vllm/vllm-continuous-batching/index.md
updated: 2026-08-06
---

# Chunked Prefill

**Chunked prefill** splits a long LLM prompt into smaller causal chunks and processes those chunks across multiple model iterations instead of executing the entire prompt in one prefill pass.

## Why It Exists

A full prefill can saturate accelerator compute, while decode work is often memory-bound at practical batch sizes. Chunking creates several prompt-work opportunities that can be mixed with decode tokens, improving utilization and making per-iteration work more predictable.

## How It Works

Each chunk writes KV-cache entries for its prompt tokens. Later chunks attend to all preceding prompt keys and values, but not future tokens. With a correct causal attention mask, the final KV cache and model outputs are equivalent to processing the full prompt at once.

The chunk size is a tradeoff: smaller chunks provide more opportunities to interleave decodes but reduce matrix-multiplication efficiency and reread earlier KV entries more often.

## Common Confusions

- **Chunked prefill vs. continuous batching:** Chunking splits one prompt; continuous batching chooses the complete set of work for each model iteration.
- **Chunked prefill vs. decode:** A chunk still processes prompt tokens in parallel; decode advances one generated token per request.
- **Chunked prefill vs. paged KV memory:** Chunking controls execution order; paged KV memory controls where cached keys and values are stored.

## Where It Appears

- [Sarathi: Chunked Prefills for Efficient LLM Inference](../frameworks/sarathi/index.md) - combines chunked prefills with decode-maximal batching and pipeline scheduling.
- [vLLM Continuous Batching](../frameworks/vllm/vllm-continuous-batching/index.md) - implements prompt chunks under a token-budget scheduler in a modern serving engine.
