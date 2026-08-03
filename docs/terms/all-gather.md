---
title: "All-Gather"
summary: "An NCCL collective that gathers shards distributed across ranks into a full replicated tensor on every rank."
tooltip: "All-gather is a collective communication primitive where each rank contributes a shard of a tensor, and every rank receives the full concatenated tensor. In distributed training, the split/all-gather pattern is a key optimization at pipeline boundaries: split a tensor into shards before a slow cross-node link, transmit one shard per rank, then all-gather over fast intra-node NVLink to reconstruct the full tensor on the receiving side."
layout: default
confidence: high
category: training
sources:
  - raw/training/megatron-lm-gpu-cluster-training-parallelism--paper.pdf
  - raw/training/sequence-parallelism-long-sequence-training--arxiv-2105.13120.pdf
aliases:
  - allgather
  - all_gather
  - all-gather collective
  - split/all-gather
appears_in:
  - docs/training/parallelism/megatron-lm/index.md
  - docs/training/parallelism/sequence-parallelism/index.md
updated: 2026-07-27
---

# All-Gather

**All-gather** is an NCCL collective communication primitive that gathers equally-sized tensor shards distributed across a group of ranks, and delivers the full concatenated tensor to every rank in the group.

## Why It Exists

Distributed training constantly moves tensors across device boundaries. When you split a model or data across GPUs, you often need to reassemble a partial tensor so every rank has a complete copy — for example, to reconstruct an activation tensor after it crosses a pipeline boundary, or to gather gradients before an optimizer step. All-gather is the standard collective for this: every rank contributes its shard, and every rank gets the full result.

In the context of pipeline-parallel training with tensor parallelism, the **split/all-gather** pattern addresses a specific bottleneck: inter-node links (InfiniBand) are much slower than intra-node links (NVLink). Sending the same activation tensor from every tensor-parallel rank over IB wastes bandwidth.

## How It Works

An all-gather over $t$ ranks, each holding a shard of size $s$, proceeds as:

1. Each rank $i$ holds shard $i$ of the tensor (equally sized).
2. After the collective, every rank holds the concatenation of all $t$ shards — a tensor $t \times$ the size of each shard.

NCCL implements all-gather efficiently via a ring algorithm: each rank sends its shard to the next rank and forwards received shards around the ring until all ranks have all shards.

### The Split/All-Gather Pattern at Pipeline Boundaries

When tensor parallelism and pipeline parallelism are composed (e.g., Megatron-LM PTD-P), activations at pipeline boundaries are often replicated across all $t$ tensor-parallel ranks inside a node. The naive approach sends the full tensor $t$ times over IB. The optimized pattern instead:

1. **Split (sender side):** The activation tensor is divided into $t$ shards. Each tensor-parallel rank sends only its shard over IB to the corresponding rank at the next pipeline stage.
2. **All-gather (receiver side):** The $t$ receiving ranks perform an intra-node all-gather over NVLink to reconstruct the full tensor.

This reduces cross-node traffic from $b \cdot s \cdot h$ (redundant copies) to $\frac{b \cdot s \cdot h}{t}$ (one shard per rank per send).

## Why Sequence Parallelism Avoids This Cost

Sequence parallelism splits the input sequence into chunks across GPUs at the start. Activations are thus **already split** along the sequence dimension — there is no replication to undo. When a sequence-parallel model is combined with pipeline parallelism, activations pass between pipeline stages as pre-split chunks, with no need for an extra split/all-gather step. This is one reason sequence parallelism achieves better throughput when composed with pipeline parallelism than tensor parallelism does.

## Common Confusions

- **All-gather vs. [all-reduce](all-reduce.md):** All-reduce sums tensors element-wise and delivers the sum; all-gather concatenates shards and delivers the full tensor. All-reduce preserves element count; all-gather increases it by $t \times$.
- **All-gather vs. scatter/gather:** Scatter distributes a full tensor into shards across ranks; gather collects shards into one rank. All-gather does both — distributes and collects — delivering the full result to every rank.
- **Split/all-gather vs. scatter/gather (Megatron-LM optimization):** These are the same pattern described from different angles. "Scatter/gather" is Megatron-LM's name for the pipeline-boundary optimization; "split/all-gather" is the same two-step process: split the tensor, send shards, all-gather to reconstruct.

## Where It Appears

- [Megatron-LM: GPU-Cluster Training Parallelism](../training/parallelism/megatron-lm/index.md) — The scatter/gather optimization uses split + all-gather at pipeline boundaries, yielding up to 11% throughput improvement.
- [Sequence Parallelism: Splitting Sequences Across GPUs](../training/parallelism/sequence-parallelism/index.md) — Sequence parallelism avoids the split/all-gather cost entirely because activations are already chunked along the sequence dimension.

## Related Terms

- [Scatter/Gather](scatter-gather.md) — Megatron-LM's name for the split/all-gather pipeline-boundary optimization.
- [Microbatch](microbatch.md) — More microbatches mean more pipeline boundary crossings, making split/all-gather more impactful.
- [Sequence Parallelism](sequence-parallelism.md) — Orthogonal parallelism dimension that avoids the need for split/all-gather at pipeline boundaries.
