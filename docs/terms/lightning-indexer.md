---
title: "Lightning Indexer"
summary: "A learned top-k block selector in compressed sparse attention that scores compressed KV blocks against the query and returns the indices of the most relevant blocks for core attention."
tooltip: "The Lightning Indexer is DeepSeek-V4's sparse-selection mechanism: a small query/key scorer that picks the top-k compressed KV blocks a CSA layer should attend to. It only needs the ordering of block scores, which is why serving frameworks can quantize its key cache to 8 bits without losing much accuracy."
layout: default
confidence: high
category: algorithms
sources:
  - raw/training/deepseek-v4--paper.pdf
aliases:
  - indexer
  - index branch
appears_in:
  - docs/training/deepseek/deepseek-v4/index.md
  - docs/training/efficient-attention/minimax-sparse-attention/index.md
  - docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md
  - docs/frameworks/vllm-ascend/deepseek-v4-inference.md
updated: 2026-08-05
---

# Lightning Indexer

**Lightning Indexer** is a learned top-k block selector in compressed sparse attention that scores compressed KV blocks against the query and returns the indices of the most relevant blocks for core attention.

## Why It Exists

Compressed sparse attention shrinks the KV cache by merging tokens into blocks (DeepSeek-V4 compresses by 4× per CSA layer), but compression alone does not tell the model *which* compressed blocks matter for a given query. Without a selector, attention would have to either attend densely to all compressed blocks (expensive) or drop blocks blindly (inaccurate). The Lightning Indexer provides the sparse selection step: it decides which compressed blocks deserve full attention.

## How It Works

For each query token, a small indexer query is generated from the shared latent query representation. Separate indexer keys are produced at a reduced head dimension. The indexer scores every compressed block, aggregates per-head scores with learned head weights, and keeps the top-k scoring blocks. Core attention then runs only over those selected blocks (plus the sliding-window branch for recent tokens). Because the indexer only needs the *ordering* of block scores, serving frameworks can quantize its key cache and query to 8 bits — e.g. INT8 with per-token-head scales, or FP8 on Ascend A5 — while preserving selection quality.

## Tradeoffs

The indexer is a heuristic: 8-bit quantization or close score margins can reorder borderline blocks and change the top-k set. The head count and head dimension are fixed by the model config (64 heads × 128 dim in DeepSeek-V4), so indexer compute scales with the number of compressed blocks rather than full sequence length.

## Common Confusions

- **Lightning Indexer vs. top-k in plain DSA:** earlier DeepSeek sparse attention (V3.2) also selects top-k blocks, but the Lightning Indexer is the *learned, query-dependent* scorer introduced with DeepSeek-V4's CSA; it is not a fixed mask.
- **Indexer vs. MoE router:** a router selects *experts* per token to reduce FFN compute; the indexer selects *KV blocks* per query to reduce attention compute.
- **C4 vs. C8:** "C8" is the 8-bit quantization of the indexer cache and query implemented in vllm-ascend today; a 4-bit "C4" indexer quantization is not implemented yet (planned future work). Separately, "c4" is the DSV4 layer type (compress ratio 4) that owns an indexer.

## Where It Appears

- [DeepSeek-V4: Million-Token Context via Hybrid Compressed Attention](../training/deepseek/deepseek-v4/index.md) — Introduces the Lightning Indexer as the sparse-selection stage of CSA layers.
- [DeepSeek-V4 Lightning Indexer C8 Quantization](../frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md) — vllm-ascend's INT8/FP8 indexer cache and quantized top-k operators.
- [MiniMax Sparse Attention](../training/efficient-attention/minimax-sparse-attention/index.md) — A sibling approach with a learned index branch for block selection.

## Related Terms

- [KV Cache](kv-cache.md) — The attention state the indexer selects blocks from.
- [Mixture of Experts](mixture-of-experts.md) — A router-based sparse selection mechanism for FFN capacity, often confused with attention indexers.
