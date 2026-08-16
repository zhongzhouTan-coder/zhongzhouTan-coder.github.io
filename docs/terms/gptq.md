---
title: "GPTQ"
summary: "A second-order post-training weight-quantization algorithm that compensates rounding errors using calibration activations and scales the procedure through shared column ordering and block updates."
tooltip: "GPTQ quantizes a model's weights after training and adjusts the remaining weights to compensate each rounding error. Shared ordering, blockwise updates, and Cholesky-based curvature information make that second-order correction practical for very large language models."
layout: default
confidence: high
category: hardware
sources:
  - raw/hardware/gptq-accurate-post-training-quantization--arxiv-2210.17323v2.pdf
aliases:
  - Accurate Post-Training Quantization for Generative Pre-trained Transformers
mention_lint: canonical
appears_in:
  - docs/hardware/quantization/gptq/index.md
  - docs/hardware/quantization/flatquant/index.md
updated: 2026-08-16
---

# GPTQ

**GPTQ** is a second-order post-training weight-quantization algorithm that rounds weights to a low-bit grid while adjusting the remaining weights to preserve each layer's outputs on calibration data.

## Why It Exists

Round-to-nearest scales to huge models but can destroy accuracy below 8 bits, while earlier second-order methods repeat expensive greedy bookkeeping for each weight row. GPTQ keeps error compensation while reorganizing it into shared, GPU-friendly work.

## How It Works

For each linear layer, GPTQ records representative input activations and uses `H = 2XXᵀ` as the Hessian of the layer-output reconstruction loss. After quantizing a column, inverse-Hessian information determines how to adjust the columns that remain in full precision.

All output rows follow one column order, 128-column blocks accumulate corrections before a global matrix update, and a damped Cholesky formulation supplies stable correction rows. The result is weight-only quantization that the paper applies to 175B-parameter models in a few hours.

## Tradeoffs

GPTQ needs calibration data and an offline quantization pass. Its original form does not quantize activations or reduce the multiplication count; deployment speed depends on packed-weight kernels and on inference being limited by weight memory traffic.

## Common Confusions

- **GPTQ vs. round-to-nearest:** Both use a fixed low-bit grid, but GPTQ compensates rounding error in the still-unquantized weights.
- **GPTQ vs. quantization-aware training:** GPTQ changes a pretrained model without retraining; QAT simulates quantization during training or fine-tuning.
- **GPTQ vs. a numeric format:** GPTQ is a calibration and rounding algorithm, not a bit encoding such as INT4 or FP4.

## Where It Appears

- [GPTQ: Second-Order Weight Quantization at LLM Scale](../hardware/quantization/gptq/index.md) — Source-defining explanation of the algorithm, deployment kernel, and OPT/BLOOM results.
- [FlatQuant](../hardware/quantization/flatquant/index.md) — Uses GPTQ as a weight-rounding solver and comparison baseline, while showing that learned flattening can make RTN competitive.

## Related Terms

- [Post-Training Quantization](post-training-quantization.md) — The broader one-shot model-compression family containing GPTQ.
- [GEMM](gemm.md) — The matrix multiplication whose memory and arithmetic behavior determines whether compressed weights improve runtime.
