---
title: "Training Parallelism"
summary: "Data, tensor, pipeline, and sequence parallelism techniques for large-model training."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-14
---

# Training Parallelism

- [Megatron-LM: GPU-Cluster Training Parallelism](megatron-lm/index.md) — Covers both Megatron-LM papers: intra-layer [tensor parallelism](../../terms/tensor-parallelism.md) with f/g conjugate operators and column-parallel [GEMM](../../terms/gemm.md) splitting (2019), and the PTD-P recipe for trillion-parameter GPT models (2021).
- [GPipe: Micro-Batch Pipeline Parallelism](gpipe/index.md) — Synchronous [micro-batch](../../terms/microbatch.md) [pipeline parallelism](../../terms/pipeline-parallelism.md) with activation recomputation: splits mini-batches into micro-batches, pipelines them through partitioned layers, and applies synchronous gradient updates for near-linear speedup.
- [Sequence Parallelism: Splitting Sequences Across GPUs](sequence-parallelism/index.md) — [Sequence parallelism](../../terms/sequence-parallelism.md): distributes input sequence chunks across GPUs with Ring Self-Attention (RSA), introducing the fourth parallelism dimension alongside data, pipeline, and tensor parallelism.
