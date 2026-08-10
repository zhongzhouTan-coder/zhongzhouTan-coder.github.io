---
title: "Ring Attention"
summary: "An exact attention schedule that circulates query or KV blocks around a device ring and merges partial softmax results."
tooltip: "Ring attention avoids an all-at-once exchange of the full context by sending blocks between neighboring ranks while computation proceeds. It can circulate KV for local queries or circulate queries over stationary KV, with a numerically stable merge restoring exact attention."
layout: default
confidence: high
category: algorithms
sources:
  - raw/algorithms/context-parallelism-scalable-million-token-inference--arxiv-2411.01783v3.pdf
aliases:
  - ring self-attention
  - RSA
appears_in:
  - docs/algorithms/context-parallelism/index.md
updated: 2026-08-10
---

# Ring Attention

**Ring Attention** is an exact attention schedule that circulates query or key/value blocks around a logical device ring and merges the resulting partial softmax computations.

## Why It Exists

Long sequences create attention and memory blocks that are too large for one device, while a full all-gather can put communication directly on the critical path. A ring exposes one manageable block at a time and overlaps its transfer with attention work.

## How It Works

In pass-KV, every rank keeps its query block and sends its KV block to a neighbor until all ranks have been visited. In pass-Q, KV stays resident and query blocks travel instead; partial outputs return to their source rank through an output redistribution. Each partial attention result carries a log-sum-exp statistic, allowing an exact stable merge across blocks.

## Tradeoffs

Ring attention needs repeated point-to-point traffic and becomes communication-bound when local attention work is too small. Pass-Q saves bandwidth for cache-heavy decode but adds an All-to-All-style output exchange; pass-KV is often better for full prefill with many new queries.

## Common Confusions

- **Ring attention vs. sparse attention:** Ring attention changes the schedule, not which tokens attend to which; it remains exact.
- **Pass-KV vs. pass-Q:** They are two traffic directions in the same exact ring family, selected for different query/KV size ratios.

## Where It Appears

- [Context Parallelism for Scalable Million-Token Inference](../algorithms/context-parallelism/index.md) - Adapts pass-KV and pass-Q ring attention to full prefill, persistent-KV prefill, and decode.
- [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889) - Earlier blockwise ring formulation for near-infinite-context training.

## Related Terms

- [Context Parallelism](context-parallelism.md) - Inference system strategy built around ring attention.
- [Sequence Parallelism](sequence-parallelism.md) - Training strategy that uses Ring Self-Attention.
- [All-to-All](all-to-all.md) - Output redistribution required by pass-Q.
