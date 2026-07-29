---
title: "Mixture of Experts"
summary: "A sparse model architecture that routes each token through a small subset of many expert feed-forward networks to increase total capacity without activating every parameter."
tooltip: "Mixture of Experts increases model width by adding many specialist FFN experts. A router chooses a few experts per token, so total parameters can be huge while active compute stays much smaller."
layout: default
confidence: high
category: training
sources:
  - raw/algorithms/deepseek-v2-multi-head-latent-attention--arxiv-2405.04434.pdf
  - raw/training/deepseek-v4--paper.pdf
  - raw/training/kimi-linear-expressive-efficient-attention--paper.pdf
  - raw/training/k3-technical-report--paper.pdf
aliases:
  - MoE
  - sparse expert model
appears_in:
  - docs/algorithms/deepseek-v2-mla.md
  - docs/training/deepseek-v4/index.md
  - docs/training/kimi-linear/index.md
  - docs/training/kimi-k3/index.md
updated: 2026-07-29
---

# Mixture of Experts

**Mixture of Experts** is a sparse neural-network architecture where a router sends each token to a small subset of many expert feed-forward networks, increasing total parameter capacity while keeping per-token active compute limited.

## Why It Exists

Dense scaling activates every parameter for every token, so larger models cost proportionally more at training and inference. MoE separates total capacity from active compute: many experts store knowledge and specialization, but only a few are evaluated per token.

## How It Works

A router scores experts for each token and selects a Top-k subset. The selected experts process the token representation, their outputs are weighted by router scores, and the aggregate returns to the main model stream. Large MoE systems usually combine routed experts with shared experts, expert parallelism, load-balancing rules, and communication overlap.

## Tradeoffs

MoE introduces routing instability, expert load imbalance, dispatch/combination communication, and train–inference consistency issues. Extreme expert counts need explicit load-balancing mechanisms such as routing biases, quantile balancing, redundant experts, or expert-parallel planning.

## Common Confusions

- **Total parameters vs. active parameters:** Total parameters count all experts; active parameters count only experts selected for one token.
- **MoE vs. attention:** MoE usually expands feed-forward/channel mixing capacity, while attention controls token mixing.

## Where It Appears

- [DeepSeek-V2 Multi-Head Latent Attention](../algorithms/deepseek-v2-mla.md) — Uses DeepSeekMoE alongside MLA to reduce active FFN compute.
- [DeepSeek-V4](../training/deepseek-v4/index.md) — Uses a 1.6T/284B-active MoE model with hybrid compressed attention.
- [Kimi Linear](../training/kimi-linear/index.md) — Evaluates KDA/MLA hybrid attention on a 48B MoE with 3B active parameters.
- [Kimi K3](../training/kimi-k3/index.md) — Scales to 2.8T total parameters, 104B active, and 896 routed experts per layer.

## Related Terms

- [All-Gather](all-gather.md) — Common distributed primitive in parallel training systems.
- [All-Reduce](all-reduce.md) — Common gradient-synchronization primitive.
