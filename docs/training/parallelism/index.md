---
title: "Training Parallelism"
summary: "Data, tensor, pipeline, and sequence parallelism techniques for large-model training."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-03
---

# Training Parallelism

- [Megatron-LM: GPU-Cluster Training Parallelism](megatron-lm/index.md) — Covers both Megatron-LM papers: intra-layer tensor parallelism with f/g conjugate operators and column-parallel GEMM splitting (2019), and the PTD-P recipe for trillion-parameter GPT models (2021).
- [GPipe: Micro-Batch Pipeline Parallelism](gpipe/index.md) — Synchronous micro-batch pipeline parallelism with activation recomputation: splits mini-batches into micro-batches, pipelines them through partitioned layers, and applies synchronous gradient updates for near-linear speedup.
- [Sequence Parallelism: Splitting Sequences Across GPUs](sequence-parallelism/index.md) — Distributes input sequence chunks across GPUs with Ring Self-Attention (RSA), introducing the fourth parallelism dimension alongside data, pipeline, and tensor parallelism.
