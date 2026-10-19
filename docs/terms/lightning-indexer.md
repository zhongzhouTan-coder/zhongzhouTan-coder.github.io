---
title: "Lightning Indexer"
summary: "A lightweight learned selector that scores visible token or block candidates and returns the top-k indices used by core sparse attention."
tooltip: "The Lightning Indexer is a small query/key scorer used by DeepSeek Sparse Attention to choose which past tokens or compressed blocks deserve core attention. Although far cheaper per pair than main attention, its full-prefix scan can become a long-context bottleneck when repeated at every layer."
layout: default
confidence: high
category: algorithms
sources:
  - raw/training/model-families/deepseek/deepseek-v4--paper.pdf
  - raw/training/model-families/deepseek/deepseek-v4.1-technical-report--paper.pdf
  - raw/training/efficient-attention/indexcache-accelerating-sparse-attention-cross-layer-index-reuse--arxiv-2603.12201v1.pdf
aliases:
  - indexer
  - index branch
appears_in:
  - docs/algorithms/model-architectures/deepseek-v3-2/index.md
  - docs/frameworks/serving/vllm/deepseek/v4-attention-code-reading.md
  - docs/frameworks/compilers-kernels/triton/triton-in-vllm.md
  - docs/frameworks/serving/vllm-ascend/architecture.md
  - docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md
  - docs/frameworks/serving/vllm-ascend/deepseek-v4-lightning-indexer-c8.md
  - docs/frameworks/serving/vllm-ascend/index.md
  - docs/frameworks/serving/vllm-ascend/qwen3.5-qwen3.6-inference.md
  - docs/training/model-families/deepseek/deepseek-v4/index.md
  - docs/training/model-families/deepseek/deepseek-v4.1/index.md
  - docs/training/efficient-attention/index.md
  - docs/training/efficient-attention/indexcache/index.md
  - docs/training/efficient-attention/minimax-sparse-attention/index.md
  - docs/training/efficient-attention/swat-sliding-window-attention/index.md
updated: 2026-09-16
---

# Lightning Indexer

**Lightning Indexer** is a lightweight learned selector that scores visible token or block candidates and returns the top-k indices used by core sparse attention.

## Why It Exists

Sparse attention saves core compute only after deciding *which* parts of the context matter. Without a content-dependent selector, attention must either scan every token in the expensive main branch or rely on a fixed mask that can miss distant evidence. The Lightning Indexer supplies that routing decision for token-level DSA and compressed-block variants.

## How It Works

For each query token, small indexer query and key projections score all visible candidates. DeepSeek-V3.2 DSA selects individual token positions; DeepSeek-V4's compressed sparse attention scores compressed candidates and combines multiple indexer heads before top-k. Core attention then runs only over the selected positions. Because selection depends mainly on score ordering, serving frameworks can quantize the indexer cache and query to 8 bits—such as INT8 with per-token-head scales or FP8 on Ascend A5—while aiming to preserve the ranking.

## Tradeoffs

The indexer is a heuristic: quantization or close score margins can reorder borderline candidates and change the top-k set. Its per-pair work is much smaller than main attention, but a full-prefix scan still has quadratic prefill complexity and linear-per-token decode cost. [IndexCache](../training/efficient-attention/indexcache/index.md) reports that repeating this scan at every layer can dominate long-context prefill.

## Common Confusions

- **Token-level vs. compressed-block selection:** DeepSeek-V3.2 DSA selects token positions, while later compressed-attention designs select compressed entries or blocks. Both use a learned, query-dependent indexer rather than a fixed mask.
- **Indexer vs. MoE router:** a router selects *experts* per token to reduce FFN compute; the indexer selects *KV blocks* per query to reduce attention compute.
- **C4 vs. C8:** "C8" is the 8-bit quantization of the indexer cache and query implemented in vllm-ascend today; a 4-bit "C4" indexer quantization is not implemented yet (planned future work). Separately, "c4" is the DSV4 layer type (compress ratio 4) that owns an indexer.

## Where It Appears

- [DeepSeek-V4 Inference on Ascend](../frameworks/serving/vllm-ascend/deepseek-v4-inference.md) — Connects the indexer to the end-to-end DSA serving stack and heterogeneous KV cache.
- [DeepSeek-V4: Million-Token Context via Hybrid Compressed Attention](../training/model-families/deepseek/deepseek-v4/index.md) — Introduces the Lightning Indexer as the sparse-selection stage of CSA layers.
- [DeepSeek-V4.1-Flash](../training/model-families/deepseek/deepseek-v4.1/index.md) — Reuses indexer state across CSA2 layers and restricts later decoder indexers to a hierarchical candidate pool.
- [DeepSeek-V4 Lightning Indexer C8 Quantization](../frameworks/serving/vllm-ascend/deepseek-v4-lightning-indexer-c8.md) — vllm-ascend's INT8/FP8 indexer cache and quantized top-k operators.
- [MiniMax Sparse Attention](../training/efficient-attention/minimax-sparse-attention/index.md) — A sibling approach with a learned index branch for block selection.
- [IndexCache](../training/efficient-attention/indexcache/index.md) — Shows that adjacent DSA layers often produce redundant top-k selections and reuses the index tensor across layers.
- [DeepSeek-V3.2: Sparse Attention, Scaled RL, and Thinking in Tool-Use](../algorithms/model-architectures/deepseek-v3-2/index.md) — DeepSeek-V3.2 introduces DeepSeek Sparse Attention (DSA) for sub-quadratic long-context efficiency, a scaled GRPO recipe with.
- [DeepSeek V4 Attention: Code Reading Map](../frameworks/serving/vllm/deepseek/v4-attention-code-reading.md) — A navigable map of the DeepSeek V4 hybrid compressed attention implementation across vLLM (NVIDIA/AMD/XPU) and vllm-ascend.
- [Triton in Practice: How vLLM and vllm-ascend Use Triton](../frameworks/compilers-kernels/triton/triton-in-vllm.md) — A codebase-driven tour of Triton kernel development in vLLM (NVIDIA GPU) and vllm-ascend (Ascend NPU), covering infrastructure.
- [vLLM-Ascend Architecture: How the Ascend NPU Port Integrates with vLLM](../frameworks/serving/vllm-ascend/architecture.md) — A code-reading tour of how vllm-ascend maps onto vLLM's six-layer stack and extends upstream vLLM for Ascend NPU execution.
- [vLLM Ascend](../frameworks/serving/vllm-ascend/index.md) — vLLM's Ascend NPU port: code-reading notes and MoE forward implementation insights.
- [Qwen3.5 / Qwen3.6 Inference Path on vLLM Ascend](../frameworks/serving/vllm-ascend/qwen3.5-qwen3.6-inference.md) — A code-reading tour of the shared qwen3_5-family inference path: Qwen3.5-27B / Qwen3.6-27B (dense hybrid Mamba-Transformer.
- [Efficient Attention Training](../training/efficient-attention/index.md) — Training approaches built around efficient attention mechanisms: sparse, sliding-window, and delta-rule recurrent attention.
- [SWAT: Sliding Window Attention Training](../training/efficient-attention/swat-sliding-window-attention/index.md) — Trains transformers from scratch with sigmoid-based sliding window attention, replacing softmax to eliminate attention sink and.

## Related Terms

- [KV Cache](kv-cache.md) — The attention state the indexer selects blocks from.
- [Cross-Layer Index Reuse](cross-layer-index-reuse.md) — Reuses one indexer's selected positions across several sparse-attention layers.
- [Mixture of Experts](mixture-of-experts.md) — A router-based sparse selection mechanism for FFN capacity, often confused with attention indexers.
