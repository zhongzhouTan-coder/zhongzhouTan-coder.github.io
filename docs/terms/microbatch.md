---
title: "Microbatch"
summary: "A small chunk of a training batch used to enable pipeline parallelism in distributed training."
tooltip: "A microbatch is one slice of a larger training batch that moves through a pipeline-parallel model. Instead of waiting for an entire batch to finish stage by stage, the system streams many microbatches through different pipeline stages so GPUs stay busy. The tradeoff is that very small microbatches can reduce GPU efficiency."
layout: default
confidence: high
category: training
sources:
  - raw/training/megatron-lm-gpu-cluster-training-parallelism--paper.pdf
  - raw/training/gpipe-micro-batch-pipeline-parallelism--arxiv-1811.06965v5.pdf
aliases:
  - micro-batch
  - micro batch
appears_in:
  - docs/frameworks/sarathi/index.md
  - docs/training/index.md
  - docs/training/parallelism/gpipe/index.md
  - docs/training/parallelism/index.md
  - docs/training/parallelism/megatron-lm/index.md
updated: 2026-07-27
---

# Microbatch

**Microbatch** is a small subdivision of a global training batch, used as the unit of work that flows through a pipeline-parallel model. Pipeline parallelism splits model layers across devices; microbatches keep all those devices busy by streaming work through them like an assembly line.

## Why It Exists

Without microbatches, pipeline parallelism would be useless. If you send one giant batch through a pipeline of $p$ devices, device 0 does all its work, then sits idle while devices $1 \dots p-1$ process. That idle time — the **pipeline bubble** — can waste up to 50% of GPU capacity. Splitting the batch into $m$ microbatches lets all devices work concurrently on different microbatches, shrinking the bubble to a warmup and drain phase at the edges.

## How It Works

Given a global batch size $B$ and data-parallel size $d$, the number of microbatches per pipeline is:

$$m = \frac{B}{b \cdot d}$$

where $b$ is the microbatch size (samples per microbatch). The pipeline bubble fraction is approximately:

$$\text{bubble} \approx \frac{p - 1}{m}$$

So a larger $m$ (more, smaller microbatches) shrinks the bubble — but smaller $b$ also means smaller matrix multiplications and lower GPU arithmetic intensity. The optimal $b$ balances GPU utilization against pipeline occupancy. Megatron-LM reports that the optimal microbatch size is problem-dependent and can vary throughput by up to 15%.

## Scheduling Strategies

Two main ways to order microbatches:

- **GPipe (all-forward-all-backward):** Run all $m$ forward passes, then all $m$ backward passes. Simple but requires stashing activations for all $m$ microbatches — high memory.
- **1F1B (one-forward-one-backward):** After a warmup phase, alternate one forward and one backward per device. Limits in-flight microbatches to $p$, dramatically reducing activation memory. Megatron-LM's **interleaved 1F1B** assigns multiple model chunks ($v$) per device to further shrink the bubble.

## Where It Appears

- [GPipe: Micro-Batch Pipeline Parallelism](../training/parallelism/gpipe/index.md) — Introduces microbatches as the core mechanism for synchronous pipeline-parallel training; defines the bubble overhead formula $O(\frac{K-1}{M+K-1})$ and the $M \ge 4K$ rule.
- [Megatron-LM: GPU-Cluster Training Parallelism](../training/parallelism/megatron-lm/index.md) — Defines the microbatch-bubble tradeoff for PTD-P training and reports throughput sensitivity to microbatch size.
- [GPipe (Huang et al., 2019)](https://arxiv.org/abs/1811.06965) — Introduced microbatches as the mechanism for pipeline-parallel training of giant neural networks.
- [Sarathi: Chunked Prefills for Efficient LLM Inference](../frameworks/sarathi/index.md) — Applies uniform pipeline microbatches to LLM inference to reduce bubbles.
- [Training](../training/index.md) — Training and fine-tuning pages covering optimization behavior, transfer learning, and generalization in large models.
- [Training Parallelism](../training/parallelism/index.md) — Data, tensor, pipeline, and sequence parallelism techniques for large-model training.

## Related Terms

- [Scatter/Gather](scatter-gather.md) — Optimization at pipeline boundaries that becomes more important when microbatches are small and interleaving increases communication frequency.
