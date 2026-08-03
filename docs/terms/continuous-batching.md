---
title: "Continuous Batching"
summary: "An LLM-serving scheduling pattern that rebuilds the active batch at each model iteration so finished requests can leave and newly ready work can enter."
tooltip: "Continuous batching reschedules LLM work at every model iteration instead of waiting for an entire static request group to finish. It improves utilization under variable output lengths, but admission remains bounded by token, sequence, and KV-memory capacity."
layout: default
confidence: high
category: frameworks
sources:
  - raw/frameworks/continuous-batching-llm-inference--web-2026-08-02-083bded3a6af.html
  - raw/frameworks/continuous-batching-llm-inference--web-2026-08-02-083bded3a6af.metadata.json
  - derived/web-markdown/frameworks/continuous-batching-llm-inference--web-2026-08-02-083bded3a6af.md
aliases:
  - iteration-level scheduling
  - in-flight batching
appears_in:
  - docs/frameworks/vllm-framework.md
  - docs/frameworks/vllm-continuous-batching/index.md
updated: 2026-08-02
---

# Continuous Batching

**Continuous Batching** is an LLM-serving scheduling pattern that chooses the
active work again at every model iteration, allowing completed requests to
leave and newly ready requests to enter without a whole-batch barrier.

## Why It Exists

Autoregressive requests generate different numbers of tokens. In a static
batch, short requests leave idle rows until the longest request finishes;
continuous batching turns each decode boundary into another scheduling
opportunity.

## How It Works

The scheduler tracks live request state across iterations. Before each model
forward, it selects runnable requests and assigns per-request token work under
compute and memory limits. After the forward, it advances each request
independently, removes finished work, and repeats.

Modern engines may mix one-token decodes, prompt chunks, and speculative tokens
in one iteration. The mechanism is therefore more precise than the simplified
picture of replacing only a finished decode row.

## Tradeoffs

Continuous batching adds scheduler and batch-reconciliation overhead. Under
high load, prompt work can interfere with decode latency, and KV-cache pressure
can cause preemption or recomputation. It provides little benefit when traffic
is sparse or request lengths are uniform.

## Common Confusions

- **Continuous vs. request-level dynamic batching:** Request-level batching
  chooses a static group before execution; continuous batching can change the
  group between generation iterations.
- **Continuous batching vs. PagedAttention:** Scheduling decides when work
  runs; PagedAttention decides how its KV state is allocated and accessed.
- **Continuous batching vs. chunked prefill:** Chunking splits one prompt over
  iterations; continuous batching is the broader policy that chooses all work
  in each iteration.

## Where It Appears

- [vLLM: PagedAttention Serving Framework](../frameworks/vllm-framework.md) —
  Relates Orca-style iteration-level scheduling to vLLM's paged KV-memory
  management.
- [vLLM Continuous Batching](../frameworks/vllm-continuous-batching/index.md) —
  Traces the current V1 scheduler, KV allocation, worker-batch update, finish,
  and preemption paths.

## Related Terms

- [KV Cache](kv-cache.md) — Request state whose capacity often limits
  continuous-batch concurrency.
