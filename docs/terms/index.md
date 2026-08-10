---
title: "Terms Glossary"
summary: "Alphabetical glossary of technical terms used across the knowledge base, with concise definitions and cross-links to papers and insight pages."
layout: default
confidence: high
sources:
  - AGENTS.md
  - .github/instructions/docs-terms.instructions.md
updated: 2026-08-10
---

# Terms Glossary

Quick-lookup definitions for technical concepts that appear across multiple papers in this knowledge base. Each term page is a self-contained definition page with backlinks to the papers that use it.

## General

- [Monkey-Patching](monkey-patching.md) — Runtime replacement, deletion, or redirection of functions, attributes, modules, or environment lookups so code runs against controlled behavior without editing the original source.

## Training

- [Pipeline Parallelism](pipeline-parallelism.md) — Assigns layer ranges to stages and streams microbatches through them for model capacity and throughput.
- [Microbatch](microbatch.md) — A small chunk of a training batch used to enable pipeline parallelism; the unit of work injected into a pipeline schedule.
- [Mixture of Experts](mixture-of-experts.md) — Routes each token through a small subset of many expert feed-forward networks to increase total capacity without activating every parameter.
- [Hyper-Connections](hyper-connections.md) — Widens the residual stream into n parallel streams mixed by learnable mappings; mHC constrains that mixing to a doubly stochastic manifold to keep it trainable at scale.
- [Scatter/Gather](scatter-gather.md) — A pipeline-boundary optimization that sends different activation shards across slow links, then reconstructs the full tensor on the destination side with fast local collectives.
- [All-Gather](all-gather.md) — A many-to-many collective that gathers per-rank shards into one full tensor and delivers that complete result to every rank.
- [All-to-All](all-to-all.md) — A many-to-many collective that scatters each rank's data to all ranks while gathering from all ranks, effectively performing a distributed matrix transpose.
- [All-Reduce](all-reduce.md) — A many-to-many collective that reduces equally shaped tensors across ranks and delivers the identical reduced result to every rank.
- [Sequence Parallelism](sequence-parallelism.md) — A distributed training strategy that splits the input sequence along the length dimension across GPUs, forming a fourth parallelism dimension orthogonal to data, pipeline, and tensor parallelism.
- [Tensor Parallelism](tensor-parallelism.md) — Splits weight matrices and hidden or head dimensions across accelerator ranks.

## Algorithms

- [Context Parallelism](context-parallelism.md) — Distributes a long sequence and its KV state across ranks while preserving exact attention.
- [Grouped-Query Attention](grouped-query-attention.md) — Shares each key/value head across a group of query heads to reduce KV-cache and communication cost.
- [Delta Rule](delta-rule.md) — Corrects an associative memory using the error between its current key-addressed prediction and the target value.
- [General Matrix Multiply (GEMM)](gemm.md) — The dense multiply-accumulate kernel C = A×B whose execution rate is the standard performance reference for linear-algebra workloads on GPUs and NPUs.
- [Inner Product](inner-product.md) — The scalar dot product Σ xᵢyᵢ; GEMM is the matrix of inner products between rows of A and columns of B, and attention scores are dot products.
- [Kimi Delta Attention](kimi-delta-attention.md) — Extends delta-rule recurrent memory with channel-wise decay and hardware-efficient chunkwise computation.
- [Lightning Indexer](lightning-indexer.md) — A learned top-k block selector in compressed sparse attention that scores compressed KV blocks against the query and returns the indices of the most relevant blocks for core attention.
- [Kronecker Product](kronecker-product.md) — The block-structured matrix product A⊗B that builds a large matrix from two smaller ones by scaling copies of B by the entries of A.
- [KV Cache](kv-cache.md) — Stores earlier attention keys and values so autoregressive decoding reuses the prefix instead of recomputing it.
- [Linear Attention](linear-attention.md) — Factors query–key similarity through feature maps so key–value associations can be accumulated without an explicit quadratic attention matrix.
- [Matrix Tiling](matrix-tiling.md) — Blocking a GEMM (or any kernel) into tiles that fit on-chip SRAM and registers so operands are loaded from global memory few times and reused many times.
- [Outer Product](outer-product.md) — A rank-1 matrix u vᵀ formed from two vectors; GEMM can be computed by accumulating outer products of columns of A with rows of B.
- [Ring Attention](ring-attention.md) — Circulates query or KV blocks around a rank ring and merges partial softmax results into exact attention.

## Hardware

- [FP8](fp8.md) — An 8-bit floating-point family used to reduce model and activation memory traffic at a controlled numerical cost.
- [Global Memory](global-memory.md) — The off-chip device memory on an accelerator (GPU HBM or Ascend GM) that holds full tensors; kernels move tiles from it into on-chip storage to compute.
- [Memory Banking](memory-banking.md) — Partitioning on-chip SRAM into banks so parallel accesses to different addresses hit different banks in the same cycle, avoiding bank conflicts.
- [Microscaling](microscaling.md) — A block-floating-point representation that shares one scale across a small group of narrow elements.
- [Systolic Array](systolic-array.md) — A regular grid of processing elements where data flows rhythmically between neighbors so each weight is reused across many multiply-accumulates without re-fetching.

## Frameworks

- [Chunked Prefill](chunked-prefill.md) — Splits a long prompt into causal chunks so prompt work can share iterations with decode work.

- [Continuous Batching](continuous-batching.md) — Rebuilds the active LLM-serving batch at each model iteration so finished work can leave and newly ready work can enter.
- [Block Table](block-table.md) — The per-request logical-to-physical mapping that tells paged attention kernels which physical KV block holds each logical block of a sequence.
- [PagedAttention](pagedattention.md) — An attention algorithm that stores a sequence's KV cache in fixed-size non-contiguous blocks addressed through a per-request block table, enabling paged memory management.

## Benchmarks

_No terms yet._
