---
title: "Terms Glossary"
summary: "Alphabetical glossary of technical terms used across the knowledge base, with concise definitions and cross-links to papers and insight pages."
layout: default
confidence: high
sources:
  - AGENTS.md
  - .github/instructions/docs-terms.instructions.md
updated: 2026-07-29
---

# Terms Glossary

Quick-lookup definitions for technical concepts that appear across multiple papers in this knowledge base. Each term page is a self-contained definition page with backlinks to the papers that use it.

## Training

- [Microbatch](microbatch.md) — A small chunk of a training batch used to enable pipeline parallelism; the unit of work injected into a pipeline schedule.
- [Mixture of Experts](mixture-of-experts.md) — Routes each token through a small subset of many expert feed-forward networks to increase total capacity without activating every parameter.
- [Scatter/Gather](scatter-gather.md) — A cross-node communication optimization in pipeline parallelism that avoids sending redundant activation tensors over slow inter-node links.
- [All-Gather](all-gather.md) — An NCCL collective that gathers shards across ranks into a full replicated tensor; the split/all-gather pattern reduces cross-node traffic at pipeline boundaries.
- [All-Reduce](all-reduce.md) — An NCCL collective that sums tensors element-wise across ranks; powers gradient synchronization and tensor-parallel output aggregation.
- [Sequence Parallelism](sequence-parallelism.md) — A distributed training strategy that splits the input sequence along the length dimension across GPUs, forming a fourth parallelism dimension orthogonal to data, pipeline, and tensor parallelism.

## Algorithms

- [Delta Rule](delta-rule.md) — Corrects an associative memory using the error between its current key-addressed prediction and the target value.
- [Kimi Delta Attention](kimi-delta-attention.md) — Extends delta-rule recurrent memory with channel-wise decay and hardware-efficient chunkwise computation.
- [KV Cache](kv-cache.md) — Stores earlier attention keys and values so autoregressive decoding reuses the prefix instead of recomputing it.
- [Linear Attention](linear-attention.md) — Factors query–key similarity through feature maps so key–value associations can be accumulated without an explicit quadratic attention matrix.

## Hardware

- [Global Memory](global-memory.md) — The off-chip device memory on an accelerator (GPU HBM or Ascend GM) that holds full tensors; kernels move tiles from it into on-chip storage to compute.

## Frameworks

_No terms yet._

## Benchmarks

_No terms yet._
