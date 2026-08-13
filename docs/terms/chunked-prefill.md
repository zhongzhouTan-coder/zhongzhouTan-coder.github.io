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
  - docs/frameworks/index.md
  - docs/frameworks/sarathi/index.md
  - docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md
  - docs/frameworks/vllm/index.md
  - docs/frameworks/vllm/vllm-code-learning-path.md
  - docs/frameworks/vllm/vllm-continuous-batching/index.md
  - docs/frameworks/vllm/vllm-overview.md
  - docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md
updated: 2026-08-13
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
- [vLLM-Ascend Prefill and Decode Scheduling: Qwen3.5 GQA](../frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md) - follows the prompt-chunk cursor from vLLM scheduling into Ascend mixed FIA execution.
- [Frameworks](../frameworks/index.md) — Framework pages covering LLM serving systems and structured language-model programming runtimes.
- [vLLM](../frameworks/vllm/index.md) — vLLM serving framework pages: PagedAttention paper, request-flow code learning path, continuous batching, and Kimi K3 code.
- [vLLM Code Learning Path and Request Flow](../frameworks/vllm/vllm-code-learning-path.md) — A code-oriented map of the current vLLM serving stack, the request lifecycle, and an achievement-driven path to build a mini.
- [vLLM Architecture and Code Organization Overview](../frameworks/vllm/vllm-overview.md) — A top-down code-reading map of the vLLM repository at commit a0c092ee72c0: how the V1 serving engine, model executor, config.
- [vLLM Prefill/Decode Disaggregated Deployment Path](../frameworks/vllm/prefill-decode-disaggregated-deployment/index.md) — Contrasts prompt chunking inside one mixed scheduler with isolating prefill and decode in separate engine pools.
