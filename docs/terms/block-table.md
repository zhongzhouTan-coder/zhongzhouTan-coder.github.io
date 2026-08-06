---
title: "Block Table"
summary: "The per-request logical-to-physical mapping that tells paged attention kernels which physical KV block holds each logical block of a sequence."
tooltip: "A block table is a sequence's page table for its KV cache: logical block index i maps to one physical block ID. Kernels read it to fetch KV blocks in constant time, and the serving engine appends to it as tokens are generated."
layout: default
confidence: high
category: frameworks
sources:
  - raw/frameworks/vllm-pagedattention-serving-framework--arxiv-2309.06180v1.pdf
aliases:
  - block tables
appears_in:
  - docs/frameworks/index.md
  - docs/frameworks/triton/triton-in-vllm.md
  - docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md
  - docs/frameworks/vllm/index.md
  - docs/frameworks/vllm/vllm-block-management/index.md
  - docs/frameworks/vllm/vllm-code-learning-path.md
  - docs/frameworks/vllm/vllm-continuous-batching/index.md
  - docs/frameworks/vllm/vllm-framework.md
  - docs/frameworks/vllm/vllm-overview.md
updated: 2026-08-06
---

# Block Table

**Block Table** is the per-request mapping from logical KV block index to physical block ID that lets paged attention kernels fetch a sequence's non-contiguous KV blocks in constant time.

## Why It Exists

Paged KV caches store a sequence's blocks at arbitrary physical addresses. Without a mapping, kernels could not find a logical block cheaply; with one, memory can be allocated on demand, shared across requests, and freed block-by-block without moving data.

## How It Works

A request's KV cache is a list of logical blocks filled left to right. The block table records, for each logical block, the physical block ID and how many slots are filled. Each table entry is fixed-size, so the mapping is dense and cheap to ship to the worker, where it becomes an int32 tensor that the attention kernel indexes as `block_table[row][col]`. A new row/block is appended only when the previous block fills.

## Tradeoffs

The table itself costs a small constant indirection per block per step. If block size is small, tables are long and kernel indirection overhead grows; if block size is large, internal fragmentation grows. Append-only maintenance keeps worker block tables stable across steps but means identical cached blocks are not de-duplicated.

## Common Confusions

- **Block table vs. slot mapping:** The block table maps logical blocks to physical blocks; the slot mapping maps individual scheduled tokens to exact physical slots. Both are passed to the kernel, the latter derived from the former.
- **Block table vs. page table:** Same idea as OS page tables; block tables are per-request and per-KV-cache-group rather than per-process.

## Where It Appears

- [vLLM: PagedAttention Serving Framework](../frameworks/vllm/vllm-framework.md) — Introduces block tables as the vLLM analogue of an OS page table.
- [vLLM Block Table Management](../frameworks/vllm/vllm-block-management/index.md) — How the V1 codebase builds, shares, prefixes-caches, and materializes block tables on the worker.
- [vLLM-Ascend Prefill and Decode Scheduling: Qwen3.5 GQA](../frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md) — Shows block tables crossing the scheduler-to-FIA handoff for mixed decode and prompt chunks.
- [Frameworks](../frameworks/index.md) — Framework pages covering LLM serving systems and structured language-model programming runtimes.
- [Triton in Practice: How vLLM and vllm-ascend Use Triton](../frameworks/triton/triton-in-vllm.md) — A codebase-driven tour of Triton kernel development in vLLM (NVIDIA GPU) and vllm-ascend (Ascend NPU), covering infrastructure.
- [vLLM](../frameworks/vllm/index.md) — vLLM serving framework pages: PagedAttention paper, request-flow code learning path, continuous batching, and Kimi K3 code.
- [vLLM Code Learning Path and Request Flow](../frameworks/vllm/vllm-code-learning-path.md) — A code-oriented map of the current vLLM serving stack, the request lifecycle, and an achievement-driven path to build a mini.
- [vLLM Continuous Batching: Scheduler, KV Blocks, and Runtime Flow](../frameworks/vllm/vllm-continuous-batching/index.md) — A code-backed explanation of how vLLM rebuilds a token-level batch each engine iteration, mixes prefill and decode work.
- [vLLM Architecture and Code Organization Overview](../frameworks/vllm/vllm-overview.md) — A top-down code-reading map of the vLLM repository at commit a0c092ee72c0: how the V1 serving engine, model executor, config.

## Related Terms

- [PagedAttention](pagedattention.md) — The attention algorithm that consumes block tables.
- [KV Cache](kv-cache.md) — The data the block table addresses.
