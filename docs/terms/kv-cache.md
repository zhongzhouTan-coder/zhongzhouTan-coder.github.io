---
title: "KV Cache"
summary: "The stored key and value tensors from earlier tokens that autoregressive attention reuses instead of recomputing the whole prefix."
tooltip: "A KV cache stores each layer's past attention keys and values so decoding reuses the prefix. It removes repeated projection work, but its memory and per-token attention read grow with context length."
layout: default
confidence: high
category: algorithms
sources:
  - raw/algorithms/attention-is-all-you-need--arxiv-1706.03762.pdf
  - raw/algorithms/transformers-are-rnns-linear-attention--arxiv-2006.16236v3.pdf
  - raw/training/kimi-linear-expressive-efficient-attention--paper.pdf
  - raw/training/k3-technical-report--paper.pdf
  - raw/frameworks/vllm-pagedattention-serving-framework--arxiv-2309.06180v1.pdf
aliases:
  - key-value cache
  - key/value cache
appears_in:
  - docs/algorithms/linear-attention/index.md
  - docs/training/kimi-linear/index.md
  - docs/training/kimi-k3/index.md
  - docs/frameworks/vllm-continuous-batching/index.md
updated: 2026-08-02
---

# KV Cache

**KV Cache** is the per-layer store of attention keys and values produced by earlier tokens, reused during autoregressive decoding so the model does not recompute the entire prefix.

## Why It Exists

Without caching, generating token $i$ repeats key and value projections for tokens $1$ through $i-1$. Caching makes those projections persistent and reduces the repeated work, which is essential for practical Transformer decoding.

## How It Works

Each new token appends one key vector and one value vector per relevant layer and KV head. Its query attends over the cached keys and combines the corresponding values. Projection work per step becomes constant, but cache memory and the attention read still grow linearly with context length.

## Tradeoffs

Long contexts can make the cache the dominant memory consumer and constrain batch size. MQA, GQA, MLA, quantization, paging, sparse attention, and linear attention reduce different parts of this cost; only recurrent-state approaches make stored state independent of sequence length.

## Common Confusions

- **KV cache vs. model weights:** Weights are fixed learned parameters; the KV cache is request-specific runtime state.
- **KV cache vs. linear-attention state:** A KV cache preserves token-level keys and values; a linear-attention state merges them into a fixed-size summary.

## Where It Appears

- [Transformers Are RNNs: Linear Attention](../algorithms/linear-attention/index.md) — Contrasts explicit growing key/value history with a fixed-size recurrent summary.
- [Kimi Linear](../training/kimi-linear/index.md) — Uses recurrent KDA states in most layers and full KV caches in periodic MLA layers.
- [Kimi K3](../training/kimi-k3/index.md) — Adds external KV-cache retention for million-token partial rollouts and aligns KDA state lifecycles with MLA cache blocks.
- [vLLM Continuous Batching](../frameworks/vllm-continuous-batching/index.md) — Shows how paged KV capacity gates per-iteration admission, completion, and preemption.

## Related Terms

- [Linear Attention](linear-attention.md) — Replaces explicit token history with accumulated key–value statistics.
- [Kimi Delta Attention](kimi-delta-attention.md) — A Kimi-family fixed-state attention mechanism used to reduce cache pressure.
- [Continuous Batching](continuous-batching.md) — Uses iteration-level admission to keep available KV capacity productive.
