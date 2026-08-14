---
title: "Pipeline Parallelism"
summary: "A model-parallel strategy that assigns contiguous layer stages to different devices and streams microbatches through them."
tooltip: "Pipeline parallelism splits a model by depth, placing different layer ranges on different devices. Microbatches keep stages busy, but warmup, drain, and stage dependencies create pipeline bubbles and it does not by itself reduce the latency of one long sequence."
layout: default
confidence: high
category: training
sources:
  - raw/algorithms/context-parallelism-scalable-million-token-inference--arxiv-2411.01783v3.pdf
aliases:
  - PP
  - pipeline model parallelism
appears_in:
  - docs/algorithms/context-parallelism/index.md
  - docs/frameworks/vllm/vllm-context-parallelism.md
  - docs/frameworks/sarathi/index.md
  - docs/training/index.md
  - docs/training/mhc/index.md
  - docs/training/parallelism/gpipe/index.md
  - docs/training/parallelism/index.md
  - docs/training/parallelism/megatron-lm/index.md
  - docs/training/parallelism/sequence-parallelism/index.md
updated: 2026-08-14
---

# Pipeline Parallelism

**Pipeline Parallelism** is a model-parallel strategy that assigns different layer ranges to stages and streams microbatches through those stages.

## Why It Exists

A model can be too large for one device even after tensor sharding, while independent microbatches provide enough work to keep several layer stages busy. Pipeline parallelism addresses model capacity and aggregate throughput.

## How It Works

Stage $i$ runs its assigned layers, sends activations to stage $i+1$, and receives the next microbatch from stage $i-1$. Schedules overlap microbatches, but warmup and drain periods leave pipeline bubbles.

## Tradeoffs

Pipeline parallelism does not split the tokens of one request across stages, so it is not a direct solution to long-context prefill latency. It composes with context parallelism: PP can divide layers while CP divides sequence state.

## Common Confusions

- **Pipeline vs. tensor parallelism:** Pipeline parallelism splits by layer depth; tensor parallelism splits each layer's matrix operations.
- **Pipeline vs. context parallelism:** Pipeline parallelism mainly improves throughput and model fit; context parallelism targets sequence-length latency and KV capacity.

## Where It Appears

- [Context Parallelism for Scalable Million-Token Inference](../algorithms/context-parallelism/index.md) - Describes why pipeline parallelism alone does not remove long-prompt latency.
- [vLLM DCP and PCP: Decode and Prefill Context Parallelism](../frameworks/vllm/vllm-context-parallelism.md) — PCP currently rejects PP greater than one in its MRV2 validation.

- [Sarathi: Chunked Prefills for Efficient LLM Inference](../frameworks/sarathi/index.md)
- [Training](../training/index.md)
- [mHC: Manifold-Constrained Hyper-Connections](../training/mhc/index.md)
- [GPipe: Micro-Batch Pipeline Parallelism](../training/parallelism/gpipe/index.md)
- [Training Parallelism](../training/parallelism/index.md)
- [Megatron-LM: GPU-Cluster Training Parallelism](../training/parallelism/megatron-lm/index.md)
- [Sequence Parallelism: Splitting Sequences Across GPUs](../training/parallelism/sequence-parallelism/index.md)

## Related Terms

- [Tensor Parallelism](tensor-parallelism.md) - Splits layer computation across ranks.
- [Context Parallelism](context-parallelism.md) - Splits sequence state across ranks.
- [Microbatch](microbatch.md) - The unit streamed through pipeline stages.
