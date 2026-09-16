---
title: "Low-Rank Approximation"
summary: "Represents a matrix with a smaller rank so its dominant patterns can be stored or computed with fewer dimensions."
tooltip: "Low-rank approximation keeps the most important directions of a matrix and drops weaker directions. It reduces storage or computation when the data has redundant structure, but accuracy depends on the chosen rank and whether the approximation is used for indexing or final computation."
layout: default
confidence: high
category: algorithms
sources:
  - raw/frameworks/kvswap-disk-aware-kv-cache-offloading-on-device-inference--arxiv-2511.11907v2.pdf
  - raw/algorithms/deepseek-v2-multi-head-latent-attention--arxiv-2405.04434.pdf
aliases:
  - low-rank compression
  - low-rank representation
mention_lint: canonical
appears_in:
  - docs/frameworks/kvswap/index.md
  - docs/algorithms/attention-variants/deepseek-v2-mla.md
updated: 2026-09-15
---

# Low-Rank Approximation

**Low-Rank Approximation** represents a matrix with fewer independent directions while preserving its dominant structure.

## Why It Exists

Many learned tensors contain correlated or redundant dimensions. Keeping only the strongest directions can reduce memory traffic and storage, which is especially useful when a full KV cache is too large for the fast memory tier.

## How It Works

Given a matrix $K$, singular value decomposition writes it as $K=U\operatorname{diag}(S)V^{\top}$. Keeping the first $r$ right-singular directions gives a smaller projection $A$ and a compact representation such as $K_{lr}=KA$, where $r$ is much smaller than the original feature width. KVSwap uses this compact K representation for importance ranking; DeepSeek-V2 uses a learned latent representation as part of the attention cache itself.

## Tradeoffs

A smaller rank saves memory but loses weaker directions. The acceptable rank depends on the data distribution and on the role of the approximation: an index used to choose entries can tolerate more error than a representation used directly to compute the model's output.

## Common Confusions

- **Low-rank approximation vs. quantization:** Low-rank approximation reduces dimensions or independent directions; quantization reduces bits per retained value. They can be combined.
- **Low-rank approximation vs. eviction:** Low-rank approximation keeps a compressed representation of the data; eviction removes selected entries.
- **Low-rank approximation vs. an exact latent architecture:** A post-training summary may approximate an existing K cache, while an architecture such as MLA is trained to consume a latent cache as part of its exact model computation.

## Where It Appears

- [KVSwap: Disk-Aware KV Cache Offloading for Long-Context On-Device Inference](../frameworks/kvswap/index.md) — Uses an offline SVD-derived K summary only to rank disk-resident KV groups, enabling more aggressive compression than a summary used for final attention.
- [DeepSeek-V2 Multi-Head Latent Attention](../algorithms/attention-variants/deepseek-v2-mla.md) — Uses low-rank latent K/V state and absorbed projections to reduce the persistent cache representation.

## Related Terms

- [KV Cache](kv-cache.md) — A common tensor store whose representation can be approximated.
- [KV Cache Offloading](kv-cache-offloading.md) — Changes the storage tier and can use low-rank metadata for selection.
