---
title: "Context Parallelism"
summary: "A distributed inference strategy that shards a long sequence and its KV state across ranks while preserving exact attention."
tooltip: "Context parallelism distributes tokens and KV-cache entries across GPUs or hosts, then uses communication during attention so every query still sees the full context. It primarily reduces long-context prefill latency and spreads KV capacity; it is not a replacement for tensor parallelism when model weights do not fit on one host."
layout: default
confidence: high
category: algorithms
sources:
  - raw/algorithms/context-parallelism-scalable-million-token-inference--arxiv-2411.01783v3.pdf
aliases:
  - CP
  - context parallel
appears_in:
  - docs/algorithms/context-parallelism/index.md
  - docs/frameworks/vllm/vllm-context-parallelism.md
  - docs/frameworks/vllm/dcp-attention/index.md
updated: 2026-08-11
---

# Context Parallelism

**Context Parallelism** is a distributed inference strategy that shards a request's sequence tokens and KV-cache entries across ranks while reconstructing exact attention through communication.

## Why It Exists

Long-context prefill has both quadratic attention work and linearly growing KV-cache storage. Context parallelism adds hosts to the sequence dimension so local attention and cache capacity scale with the number of ranks.

## How It Works

Each rank keeps the model's TP-sharded weights and a slice of the request. A ring circulates either KV blocks or query blocks, depending on which message is cheaper for the current new-token length and cache-hit rate. Partial softmax outputs are merged with log-sum-exp statistics, so the result is exact dense attention rather than a sparse approximation.

## Tradeoffs

Context parallelism is strongest for long prefill, where attention compute can hide ring communication. Decode has less compute to overlap, and query padding plus an output All-to-All can make time-to-incremental-token worse as the CP group grows. CP also replicates the model across hosts and therefore complements rather than replaces tensor parallelism.

## Common Confusions

- **Context parallelism vs. sequence parallelism:** Context parallelism here targets inference, persistent KV state, and multi-turn decode; sequence parallelism is primarily a training strategy with a related sequence-sharding idea.
- **Context parallelism vs. sparse attention:** CP preserves every exact attention interaction; sparse attention reduces the number of interactions.

## Where It Appears

- [Context Parallelism for Scalable Million-Token Inference](../algorithms/context-parallelism/index.md) - Exact pass-KV and pass-Q ring attention for Llama3 405B, scaling to 1M-token prefill on 128 H100 GPUs.
- [vLLM DCP and PCP: Decode and Prefill Context Parallelism](../frameworks/vllm/vllm-context-parallelism.md) - Separates prefill batch partitioning from decode KV sharding in the V1 serving runtime.

- [vLLM DCP Attention: From Local LSE to Exact Global Output](../frameworks/vllm/dcp-attention/index.md) - Shows how DCP merges local outputs and LSE statistics to recover exact global attention.

## Related Terms

- [Ring Attention](ring-attention.md) - The blockwise communication pattern used to compute attention across shards.
- [Sequence Parallelism](sequence-parallelism.md) - The training-oriented predecessor with a related sequence-dimension split.
- [Tensor Parallelism](tensor-parallelism.md) - The complementary strategy that shards model weights.
- [KV Cache](kv-cache.md) - The request-specific state distributed by context parallelism.
