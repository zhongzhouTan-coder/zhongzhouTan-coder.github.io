---
title: "All-Reduce"
summary: "An NCCL collective that sums tensors element-wise across ranks and delivers the result to every rank."
tooltip: "All-reduce is the most common communication collective in distributed training. Every rank contributes a tensor, the tensors are summed element-wise across all ranks, and every rank receives the identical sum. It powers gradient synchronization in data parallelism and output aggregation in tensor parallelism. The key cost is that all-reduce bandwidth scales with tensor size, not with the number of ranks — so large tensors and frequent calls are the bottlenecks."
layout: default
confidence: high
category: training
sources:
  - raw/training/megatron-lm-gpu-cluster-training-parallelism--paper.pdf
  - raw/training/sequence-parallelism-long-sequence-training--arxiv-2105.13120.pdf
aliases:
  - allreduce
  - all_reduce
  - all-reduce collective
appears_in:
  - docs/training/parallelism/megatron-lm/index.md
  - docs/training/parallelism/sequence-parallelism/index.md
  - docs/training/foundation-models/llama.md
  - docs/frameworks/vllm/vllm-framework.md
updated: 2026-07-27
---

# All-Reduce

**All-reduce** is an NCCL collective communication primitive that sums tensors element-wise across a group of ranks and delivers the identical summed tensor to every rank in the group.

## Why It Exists

Distributed training constantly needs to combine partial results from multiple GPUs. When each data-parallel replica computes its own gradient, you need to average those gradients before the optimizer step. When tensor parallelism splits a matrix multiplication, the partial outputs must be summed across ranks. All-reduce is the one collective that does both jobs: reduce (sum) across ranks and broadcast the result to everyone.

Without all-reduce, distributed training would need a separate reduce followed by a broadcast — doubling the communication time. All-reduce folds these into a single optimized operation, typically implemented as a ring or tree algorithm that keeps all links busy simultaneously.

## How It Works

Given $t$ ranks, each holding a tensor of the same shape, all-reduce computes:

$$\text{result} = \sum_{i=0}^{t-1} \text{tensor}_i$$

and delivers `result` to every rank. The dominant implementation is the **ring all-reduce**: ranks are arranged in a logical ring, each rank sends its data to the next rank while receiving from the previous, and partial sums accumulate as data circulates. After $2(t-1)$ steps, every rank has the full sum.

The bandwidth cost is $2 \cdot \frac{t-1}{t} \cdot \text{size}$ — nearly independent of $t$ for large $t$, meaning the cost is dominated by the tensor size, not the number of ranks.

## Where All-Reduce Appears in Training

| Parallelism | What gets all-reduced | Frequency |
|---|---|---|
| Data parallelism | Gradients | Once per batch |
| Tensor parallelism | MLP output (after column-parallel linear), attention output (after row-parallel linear) | Every forward + backward pass per layer |
| Pipeline parallelism | None directly (uses P2P send/recv) | — |

Tensor parallelism's all-reduces are the most expensive because they happen **every layer, every forward and backward pass**. This is why tensor parallelism is kept within a single node (NVLink): moving those all-reduces across slower inter-node links (InfiniBand) would cripple throughput.

## Why Sequence Parallelism Avoids Them

Sequence parallelism has **zero all-reduces in MLP blocks**. Each device computes its chunk's MLP independently because the linear layers operate on each token in isolation — no cross-token aggregation is needed. The only communication is ring P2P in the attention block (circulating $K$ and $V$). This is the structural reason sequence parallelism achieves better throughput when composed with pipeline parallelism: it removes the frequent all-reduces that tensor parallelism pays.

## Common Confusions

- **All-reduce vs. all-gather:** All-reduce sums tensors element-wise; all-gather concatenates shards. All-reduce preserves element count; all-gather increases it by $t \times$.
- **All-reduce vs. reduce:** Reduce sums across ranks but delivers the result to only one rank. All-reduce delivers it to everyone.
- **All-reduce vs. reduce-scatter:** Reduce-scatter sums and then scatters the result so each rank gets a different shard. All-reduce = reduce-scatter followed by all-gather.

## Where It Appears

- [Megatron-LM: GPU-Cluster Training Parallelism](../training/parallelism/megatron-lm/index.md) — Tensor parallelism creates frequent all-reduces inside each Transformer layer; keeping them within a DGX node over NVLink is a core design constraint.
- [Sequence Parallelism: Splitting Sequences Across GPUs](../training/parallelism/sequence-parallelism/index.md) — Sequence parallelism avoids all-reduces in MLP blocks entirely because each chunk's linear layers operate independently.
- [vLLM: PagedAttention Serving Framework](../frameworks/vllm/vllm-framework.md) — Workers synchronize intermediate results through all-reduce in tensor-parallel inference.
- [LLaMA: Open and Efficient Foundation Language Models](../training/foundation-models/llama.md) — LLaMA overlaps all-reduce communication with computation to hide gradient-synchronization latency.

## Related Terms

- [All-Gather](all-gather.md) — The gather counterpart; all-reduce sums, all-gather concatenates.
- [Scatter/Gather](scatter-gather.md) — Pipeline optimization that avoids redundant cross-node sends; relies on intra-node all-gather.
- [Sequence Parallelism](sequence-parallelism.md) — Avoids all-reduces in MLP blocks by splitting along the sequence dimension instead of hidden/head dimensions.
