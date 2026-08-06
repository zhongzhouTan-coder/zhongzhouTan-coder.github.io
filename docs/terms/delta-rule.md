---
title: "Delta Rule"
summary: "An online error-correction update that changes an associative memory in proportion to the difference between its current prediction and a target value."
tooltip: "The delta rule reads what a memory currently predicts at a key, measures the error against the desired value, and writes only that correction. In linear recurrent models it enables targeted key–value replacement instead of indiscriminate accumulation or decay."
layout: default
confidence: high
category: algorithms
sources:
  - raw/training/gated-delta-networks-improving-mamba2-with-delta-rule--arxiv-2412.06464.pdf
  - raw/training/kimi-linear-expressive-efficient-attention--paper.pdf
  - raw/training/k3-technical-report--paper.pdf
aliases:
  - delta update rule
  - Widrow-Hoff rule
appears_in:
  - docs/frameworks/triton/triton-in-vllm.md
  - docs/training/efficient-attention/gated-delta-networks/index.md
  - docs/training/efficient-attention/index.md
  - docs/training/index.md
  - docs/training/kimi/kimi-k3/index.md
  - docs/training/kimi/kimi-linear/index.md
updated: 2026-07-29
---

# Delta Rule

**Delta Rule** is an online error-correction update that changes a model's prediction at an input in proportion to the difference between its current prediction and the desired target.

## Why It Exists

Simply adding key–value [outer products](outer-product.md) makes a fixed-size associative memory accumulate interference. Global decay frees capacity but weakens every association together. The delta rule instead lets a new observation correct the memory specifically in the direction addressed by its key.

## How It Works

For matrix memory $\mathbf S$, key $\mathbf k$, target value $\mathbf v$, and step size $\beta$, the update is

$$
\mathbf S'=\mathbf S+\beta(\mathbf v-\mathbf S\mathbf k)\mathbf k^T.
$$

First read the current prediction $\mathbf S\mathbf k$, subtract it from the target, and write that residual back along $\mathbf k$. DeltaNet interprets this as an online gradient step; Gated DeltaNet adds adaptive global decay, and Kimi Delta Attention makes that decay channel-wise.

## Tradeoffs

The update is only as selective as the learned keys: overlapping key directions can interfere. Correcting one association at a time also does not rapidly clear a whole obsolete context, motivating a separate forgetting gate.

## Common Confusions

- **Delta rule vs. global decay:** Delta updates one key-aligned direction; decay scales the whole state.
- **Delta rule vs. Hebbian write:** A Hebbian-style update adds $\mathbf v\mathbf k^T$ directly; the delta rule first subtracts the memory's current prediction.

## Where It Appears

- [Gated Delta Networks](../training/efficient-attention/gated-delta-networks/index.md) — Combines the delta rule with data-dependent scalar decay and a chunkwise parallel training algorithm.
- [Kimi Linear](../training/kimi/kimi-linear/index.md) — Extends Gated DeltaNet to channel-wise decay under a hardware-efficient DPLR constraint.
- [Kimi K3](../training/kimi/kimi-k3/index.md) — Uses KDA's delta-rule update inside the 3:1 KDA/MLA hybrid attention stack.
- [Triton in Practice: How vLLM and vllm-ascend Use Triton](../frameworks/triton/triton-in-vllm.md) — A codebase-driven tour of Triton kernel development in vLLM (NVIDIA GPU) and vllm-ascend (Ascend NPU), covering infrastructure.
- [Efficient Attention Training](../training/efficient-attention/index.md) — Training approaches built around efficient attention mechanisms: sparse, sliding-window, and delta-rule recurrent attention.
- [Training](../training/index.md) — Training and fine-tuning pages covering optimization behavior, transfer learning, and generalization in large models.

## Related Terms

- [Linear Attention](linear-attention.md) — The broader fixed-state attention family in which DeltaNet operates.
- [Kimi Delta Attention](kimi-delta-attention.md) — A channel-wise delta-rule attention mechanism.
