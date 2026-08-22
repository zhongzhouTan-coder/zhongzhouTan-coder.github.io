---
title: "Post-Training Quantization"
summary: "Model quantization applied to an already-trained network using a small calibration set or weight statistics instead of full retraining."
tooltip: "Post-training quantization converts a trained model to lower-precision weights or activations with little or no parameter training. It is cheaper than quantization-aware training, but accuracy depends on calibration, the quantization grid, and how the method handles outliers and rounding error."
layout: default
confidence: high
category: hardware
sources:
  - raw/hardware/gptq-accurate-post-training-quantization--arxiv-2210.17323v2.pdf
  - raw/hardware/flatquant-fast-learnable-affine-quantization--arxiv-2410.09426v4.pdf
aliases:
  - PTQ
  - one-shot quantization
mention_lint: canonical
appears_in:
  - docs/hardware/quantization/gptq/index.md
  - docs/hardware/quantization/flatquant/index.md
  - docs/hardware/quantization/quarot/index.md
updated: 2026-08-21
---

# Post-Training Quantization

**Post-Training Quantization** is model quantization performed after training, using pretrained weights and usually a small calibration set instead of retraining the model end to end.

## Why It Exists

Training or fine-tuning billion-parameter models with simulated low-precision arithmetic can be prohibitively expensive. PTQ offers a shorter deployment path: start with an existing checkpoint and transform it into a smaller or faster representation.

## How It Works

A PTQ method chooses a numeric grid, scale granularity, and which tensors to quantize. Simple methods derive ranges from weights or calibration activations and round to the nearest grid point. More accurate methods correct rounding errors, learn clipping or scaling parameters, rotate or transform distributions, or keep sensitive values at higher precision.

PTQ may be weight-only, activation-only, or weight-and-activation quantization. Calibration is normally much smaller than training, but it should expose representative tensor distributions.

## Tradeoffs

PTQ is inexpensive relative to retraining, yet it has less freedom to recover from quantization error. Very low bit widths, activation outliers, calibration-domain mismatch, and missing optimized kernels can erase its accuracy or latency benefits.

## Common Confusions

- **PTQ vs. quantization-aware training:** PTQ starts from a finished model and avoids full retraining; QAT exposes model parameters to simulated quantization during optimization.
- **Weight-only vs. W4A4:** Weight-only PTQ leaves activations in higher precision; W4A4 quantizes both weights and activations to 4 bits.
- **Compression vs. speed:** Fewer bits always reduce storage, but latency improves only when hardware and kernels exploit the representation for the target workload.

## Where It Appears

- [GPTQ](../hardware/quantization/gptq/index.md) — Uses activation-derived second-order correction to make 3–4-bit weight-only PTQ practical at 175B scale.
- [FlatQuant](../hardware/quantization/flatquant/index.md) — Learns affine transformations, scaling, and clipping for accurate weight-and-activation PTQ.
- [QuaRot](../hardware/quantization/quarot/index.md) — Rotates activations to remove outliers so plain 4-bit weight-and-activation PTQ works end to end.

## Related Terms

- [GPTQ](gptq.md) — A named second-order PTQ algorithm for weight-only quantization.
- [FP8](fp8.md) — A low-precision floating-point family that can be used by training or post-training workflows.
