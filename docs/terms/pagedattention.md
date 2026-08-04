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
  - docs/frameworks/vllm/vllm-framework.md
  - docs/frameworks/vllm/vllm-continuous-batching/index.md
  - docs/frameworks/vllm/vllm-block-management/index.md
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

## Related Terms

- [KV Cache](kv-cache.md) — The stored keys/values that PagedAttention pages.
- [Block Table](block-table.md) — The logical-to-physical mapping PagedAttention relies on.
- [Continuous Batching](continuous-batching.md) — The iteration-level scheduling that consumes paged capacity.
