---
title: "Prefix Caching"
summary: "Reusing compatible attention state for a previously processed token prefix to avoid repeated prompt computation."
tooltip: "Reusing compatible attention state for a previously processed token prefix to avoid repeated prompt computation. Its benefit depends on execution dependencies, resource placement, and communication cost."
layout: default
confidence: high
category: frameworks
sources:
  - raw/frameworks/serving/inference-optimization/llm-inference-optimization-practices-2026--caict.pdf
aliases:
  - prefix cache reuse
mention_lint: canonical
appears_in:
  - docs/frameworks/serving/inference-optimization/caict-2026/index.md
updated: 2026-10-31
---

# Prefix Caching

**Prefix Caching** reuses compatible attention state for a previously processed token prefix to avoid computing that same prefix again.

## Why It Exists

Requests often repeat a system instruction, conversation history, or document prefix. Recomputing it consumes prompt-processing capacity and delays the first output token.

## How It Works

The runtime identifies a matching token prefix under a compatible model and execution configuration. It reuses the corresponding [KV cache](kv-cache.md) and computes only the unmatched suffix. For example, a valid 24K-prefix hit in a 32K-token prompt leaves 8K tokens of new prompt computation, though fetching cached state still has a cost.

Caches can be local to one worker or shared across instances. A router can favor a worker with reusable state, while also considering its load.

## Tradeoffs

Cache storage, lookup, eviction, and transfer have costs. Low reuse or remote state behind a slow link can reduce the benefit. In ordinary causal attention, an identical passage at a different position or under different preceding text does not automatically have interchangeable state.

## Common Confusions

- [PagedAttention](pagedattention.md) manages state allocation; prefix reuse avoids recomputation. Either idea can be described independently.
- [KV-cache offloading](kv-cache-offloading.md) changes where state resides; it does not create a prefix match.
- Caching prompt state does not mean reusing a completed answer or skipping generation of the new response.

## Where It Appears

- [CAICT 2026: Inference Optimization as a System](../frameworks/serving/inference-optimization/caict-2026/index.md) — Connects prefix reuse to cache-aware routing, transfer cost, and phase separation (§§III–IV).

## Related Terms

- [KV Cache](kv-cache.md)
- [PagedAttention](pagedattention.md)
- [Prefill-Decode Disaggregation](prefill-decode-disaggregation.md)
