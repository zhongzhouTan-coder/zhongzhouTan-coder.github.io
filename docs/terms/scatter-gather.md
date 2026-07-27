---
title: "Scatter/Gather"
summary: "A cross-node communication optimization in pipeline parallelism that reduces redundant activation transfers over slow inter-node links."
tooltip: "Scatter/gather is a communication pattern that sends only the needed shard of an activation tensor across a slow inter-node link, then reconstructs the full tensor inside the destination node. In pipeline-parallel training with tensor parallelism, this avoids sending the same activation redundantly from every tensor-parallel rank. It matters most when inter-node bandwidth is the bottleneck."
layout: default
confidence: high
category: training
sources:
  - raw/infer-architecture/megatron-lm.pdf
aliases:
  - scatter-gather
  - scatter gather
appears_in:
  - docs/training/megatron-lm/index.md
updated: 2026-07-27
---

# Scatter/Gather

**Scatter/Gather** is a communication optimization in Megatron-LM that reduces the amount of cross-node data sent at pipeline-stage boundaries when tensor parallelism is active. It avoids sending the same activation tensor redundantly from every tensor-parallel rank.

## Why It Exists

In a pipeline-parallel setup with tensor parallelism (e.g., $t = 8$ within a DGX node), the activation tensor at a pipeline boundary is often replicated across all $t$ ranks. Naively, every rank sends its copy to the next pipeline stage over inter-node InfiniBand (IB). That means the **same tensor crosses the slow IB link $t$ times** — wasting bandwidth that could otherwise carry useful work.

## How It Works

1. **Scatter (sender side):** The activation tensor is split into $t$ equal chunks. Each tensor-parallel rank sends only one chunk over IB to the corresponding rank in the next pipeline stage.

2. **Gather (receiver side):** The receiving ranks perform an intra-node all-gather over fast NVLink to reconstruct the full tensor.

This reduces the cross-node payload from $b \cdot s \cdot h$ (full tensor, $t$ redundant copies) to $\frac{b \cdot s \cdot h}{t}$ (one shard per rank).

## Impact

Megatron-LM reports up to **11% throughput improvement** for communication-intensive schedules (large batch sizes with interleaved 1F1B). The optimization is especially important for the interleaved schedule, which increases pipeline communication frequency — without scatter/gather, the redundant cross-node traffic would cancel out the interleaving benefit.

## Where It Appears

- [Megatron-LM: GPU-Cluster Training Parallelism](../training/megatron-lm/index.md) — Introduced as part of the PTD-P training system; Section 5.7 shows the 11% throughput gain on GPT-3 175B.
- The technique is accelerator-agnostic and applies to any distributed training system where intra-node bandwidth (NVLink) vastly exceeds inter-node bandwidth (InfiniBand).

## Related Terms

- [Microbatch](microbatch.md) — Smaller microbatches increase communication frequency, making scatter/gather more important.
