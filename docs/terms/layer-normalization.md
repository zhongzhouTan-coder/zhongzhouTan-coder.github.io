---
title: "Layer Normalization"
summary: "A normalization method that standardizes each example or token across its feature dimensions, independently of other batch members."
tooltip: "Layer normalization computes mean and variance from one token's hidden features, then applies learned featurewise scale and shift. Unlike BatchNorm, its output does not depend on the other examples in the batch."
layout: default
confidence: medium
category: algorithms
sources:
  - raw/algorithms/transformer-layernorm--web-2026-08-13-a67f03dab584.html
  - raw/algorithms/transformer-layernorm--web-2026-08-13-a67f03dab584.metadata.json
  - derived/web-markdown/algorithms/transformer-layernorm--web-2026-08-13-a67f03dab584.md
aliases:
  - LayerNorm
  - layer norm
  - LN
appears_in:
  - docs/algorithms/foundations/layer-normalization/index.md
  - docs/algorithms/foundations/transformer.md
  - docs/algorithms/foundations/index.md
  - docs/algorithms/index.md
  - docs/frameworks/triton/triton-in-vllm.md
  - docs/frameworks/vllm-ascend/architecture.md
  - docs/frameworks/vllm/vllm-overview.md
  - docs/hardware/quantization/flatquant/index.md
  - docs/training/foundation-models/gpt-2.md
  - docs/training/index.md
  - docs/training/parallelism/megatron-lm/index.md
updated: 2026-08-13
---

# Layer Normalization

**Layer Normalization** is a normalization method that standardizes one example or token across its feature dimensions, independently of other batch members.

## Why It Exists

Sequence models process variable lengths and may train or infer with very different batch compositions. Batch-dependent statistics can therefore be noisy or mismatched, while LayerNorm gives each token a stable, local normalization rule.

## How It Works

For a hidden vector $x$ of width $D$, LayerNorm computes its population mean $\mu$ and variance $\sigma^2$, then returns

$$y=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,$$

where $\gamma$ and $\beta$ are learned featurewise parameters. In a Transformer tensor shaped `[batch, sequence, hidden]`, the reduction is normally over `hidden` only.

## Tradeoffs

LayerNorm is independent of batch size and needs no running statistics, but it discards the token's overall feature mean and scale before the learned affine transform. It can also be less suitable than BatchNorm when population-level channel statistics are useful.

## Common Confusions

- **LayerNorm vs. BatchNorm:** LayerNorm reduces features within one token or example; BatchNorm reduces examples for each feature or channel.
- **Statistics vs. parameters:** Mean and variance are recomputed from the current input; only $\gamma$ and $\beta$ are learned.
- **Normalization vs. normal distribution:** Standardization gives zero mean and unit variance before the affine transform; it does not make the values Gaussian.

## Where It Appears

- [Layer Normalization in Transformers](../algorithms/foundations/layer-normalization/index.md) — Derives the operation, contrasts its axis with BatchNorm, and traces one token through the computation.
- [The Transformer: Attention Is All You Need](../algorithms/foundations/transformer.md) — Uses LayerNorm around every attention and feed-forward sublayer in the original post-norm architecture.

- [Attention Foundations](../algorithms/foundations/index.md)
- [Algorithms](../algorithms/index.md)
- [Triton in Practice: How vLLM and vllm-ascend Use Triton](../frameworks/triton/triton-in-vllm.md)
- [vLLM-Ascend Architecture: How the Ascend NPU Port Integrates with vLLM](../frameworks/vllm-ascend/architecture.md)
- [vLLM Architecture and Code Organization Overview](../frameworks/vllm/vllm-overview.md)
- [FlatQuant: Fast Learnable Affine Quantization](../hardware/quantization/flatquant/index.md)
- [GPT-2: Language Models are Unsupervised Multitask Learners](../training/foundation-models/gpt-2.md)
- [Training](../training/index.md)
- [Megatron-LM: GPU-Cluster Training Parallelism](../training/parallelism/megatron-lm/index.md)
