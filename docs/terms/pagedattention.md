---
title: "PagedAttention"
summary: "An attention algorithm that stores a sequence's KV cache in fixed-size non-contiguous blocks addressed through a per-request block table, enabling paged memory management."
tooltip: "PagedAttention divides the KV cache into fixed-size blocks that can live at arbitrary physical addresses. Attention reads each block through a block table, so memory can be allocated on demand and shared via reference counts, like OS paging."
layout: default
confidence: high
category: algorithms
sources:
  - raw/frameworks/vllm-pagedattention-serving-framework--arxiv-2309.06180v1.pdf
aliases:
  - paged attention
appears_in:
  - docs/algorithms/attention-variants/grouped-query-attention/index.md
  - docs/algorithms/attention-variants/multi-query-attention.md
  - docs/algorithms/flashattention/flashattention-2.md
  - docs/algorithms/flashattention/flashattention-4.md
  - docs/algorithms/flashattention/flashattention.md
  - docs/algorithms/foundations/transformer.md
  - docs/frameworks/dspark/index.md
  - docs/frameworks/index.md
  - docs/frameworks/sarathi/index.md
  - docs/frameworks/sglang/index.md
  - docs/frameworks/triton/triton-in-vllm.md
  - docs/frameworks/vllm/index.md
  - docs/frameworks/vllm/vllm-block-management/index.md
  - docs/frameworks/vllm/vllm-code-learning-path.md
  - docs/frameworks/vllm/vllm-continuous-batching/index.md
  - docs/frameworks/vllm/vllm-framework.md
  - docs/frameworks/vllm/vllm-overview.md
  - docs/training/deepseek/deepseek-v4/index.md
updated: 2026-08-03
---

# PagedAttention

**PagedAttention** is an attention algorithm that stores a sequence's keys and values in fixed-size non-contiguous blocks, addressed through a per-request block table, so KV-cache memory can be allocated, shared, and freed at block granularity.

## Why It Exists

Autoregressive serving is memory-bound: KV cache grows with every generated token, and pre-reserving contiguous chunks wastes 60-80% of KV memory through reserved, internal, and external fragmentation. PagedAttention makes KV memory behave like OS virtual memory so waste is bounded and freed blocks are reusable.

## How It Works

The sequence's KV cache is partitioned into blocks of $B$ tokens each (default 16). A logical block $j$ holds $K_j$ and $V_j$ and maps through the request's block table to an arbitrary physical block. Attention computes blockwise:

$$A_{ij} = \frac{\exp(q_i^\top K_j / \sqrt{d})}{\sum_{t=1}^{\lceil i/B \rceil} \exp(q_i^\top K_t \mathbf{1}/\sqrt{d})}, \quad o_i = \sum_{j=1}^{\lceil i/B \rceil} V_j A_{ij}^\top.$$

The kernel fetches each block separately, and a new physical block is allocated only when the previous one fills — bounding per-request waste to at most one block.

## Tradeoffs

Block size trades kernel parallelism against fragmentation: larger blocks read more positions in parallel but waste more on short sequences. The kernel pays extra overhead for block-table indirection and variable sequence lengths.

## Common Confusions

- **PagedAttention vs. RadixAttention:** PagedAttention pages memory at fixed block size and shares via reference counts; SGLang's RadixAttention additionally organizes cached prefixes in a radix tree for finer-grained reuse.
- **PagedAttention vs. FlashAttention:** FlashAttention optimizes the attention *compute* (tiling, IO); PagedAttention optimizes the KV *storage* (paging). They are orthogonal and both appear in vLLM.

## Where It Appears

- [vLLM: PagedAttention Serving Framework](../frameworks/vllm/vllm-framework.md) — The original paper insight: memory waste analysis, block tables, copy-on-write, scheduling.
- [vLLM Continuous Batching](../frameworks/vllm/vllm-continuous-batching/index.md) — How paged KV-slot allocation gates per-iteration admission in the V1 scheduler.
- [vLLM Block Table Management](../frameworks/vllm/vllm-block-management/index.md) — The V1 code stack that implements paging: block pool, per-group managers, prefix caching, and worker block-table tensors.
- [Grouped-Query Attention in Llama 2](../algorithms/attention-variants/grouped-query-attention/index.md) — Explains why Llama 2 uses grouped-query attention for its 34B and 70B models: it cuts KV-cache pressure like multi-query.
- [Multi-Query Attention: One Write-Head is All You Need](../algorithms/attention-variants/multi-query-attention.md) — Replaces per-head key/value projections with a single shared K/V pair across all attention heads, eliminating the heads.
- [FlashAttention-2: Better Parallelism and Work Partitioning](../algorithms/flashattention/flashattention-2.md) — FlashAttention-2 algorithm: reduced non-matmul overhead, sequence-parallel attention blocks, warp-level work partitioning, and.
- [FlashAttention-4: Blackwell Attention Kernel Co-Design](../algorithms/flashattention/flashattention-4.md) — FlashAttention-4 algorithm and kernel-pipeline techniques for faster exact attention on NVIDIA Blackwell GPUs.
- [FlashAttention: IO-Aware Exact Attention](../algorithms/flashattention/flashattention.md) — Original FlashAttention algorithm: tiled exact attention, online softmax, recomputation, IO complexity, block-sparse extension.
- [The Transformer: Attention Is All You Need](../algorithms/foundations/transformer.md) — The foundational paper that introduced the Transformer architecture, dispensing with recurrence and convolutions entirely in.
- [DSpark: Confidence-Scheduled Speculative Decoding](../frameworks/dspark/index.md) — DeepSeek's DSpark speculative decoding framework, combining semi-autoregressive draft generation with hardware-aware confidence.
- [Frameworks](../frameworks/index.md) — Framework pages covering LLM serving systems and structured language-model programming runtimes.
- [Sarathi: Chunked Prefills for Efficient LLM Inference](../frameworks/sarathi/index.md) — Sarathi improves LLM serving by splitting prefills into compute-sized chunks and piggybacking decode tokens on them to raise.
- [SGLang: Structured Language Model Programs](../frameworks/sglang/index.md) — SGLang framework architecture, programming model, runtime optimizations, and evaluation results for efficient structured LLM.
- [Triton in Practice: How vLLM and vllm-ascend Use Triton](../frameworks/triton/triton-in-vllm.md) — A codebase-driven tour of Triton kernel development in vLLM (NVIDIA GPU) and vllm-ascend (Ascend NPU), covering infrastructure.
- [vLLM](../frameworks/vllm/index.md) — vLLM serving framework pages: PagedAttention paper, request-flow code learning path, continuous batching, and Kimi K3 code.
- [vLLM Code Learning Path and Request Flow](../frameworks/vllm/vllm-code-learning-path.md) — A code-oriented map of the current vLLM serving stack, the request lifecycle, and an achievement-driven path to build a mini.
- [vLLM Architecture and Code Organization Overview](../frameworks/vllm/vllm-overview.md) — A top-down code-reading map of the vLLM repository at commit a0c092ee72c0: how the V1 serving engine, model executor, config.
- [DeepSeek-V4: Million-Token Context via Hybrid Compressed Attention](../training/deepseek/deepseek-v4/index.md) — DeepSeek-V4 introduces hybrid Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA), Manifold-Constrained.

## Related Terms

- [KV Cache](kv-cache.md) — The stored keys/values that PagedAttention pages.
- [Block Table](block-table.md) — The logical-to-physical mapping PagedAttention relies on.
- [Continuous Batching](continuous-batching.md) — The iteration-level scheduling that consumes paged capacity.
