---
title: "Microscaling"
summary: "A block-floating-point representation that shares one scale across a small group of narrow elements."
tooltip: "Microscaling stores one shared scale with a block of narrow element values, amortizing exponent metadata and alignment work. OCP MX is the best-known standard family, but the word does not mean that every microscaled format has the same block size or encoding."
layout: default
confidence: high
category: hardware
sources:
  - raw/hardware/microscaling-mx-formats--ocp-v1.0.pdf
appears_in:
  - docs/hardware/quantization/microscaling-mx-formats/index.md
  - docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md
  - docs/hardware/quantization/index.md
  - docs/hardware/quantization/nvfp4.md
updated: 2026-08-09
---

# Microscaling

**Microscaling** is a block-floating-point representation that applies one shared scale to a small group of same-type narrow elements.

## Why It Exists

Scalar floating-point values repeat exponent metadata and force hardware to align products individually. Sharing a scale across a block reduces storage and moves much of that alignment work outside the inner reduction, which is useful for AI matrix multiplication and convolution.

## How It Works

An MX block contains one scale `X`, `k` private elements `P_i`, and a chosen block size. OCP MX fixes `k=32` and uses an 8-bit E8M0 scale, while choosing FP4, FP6, FP8, or INT8 for the private elements. A block therefore costs `w + k*d` bits, such as 200 bits for MXFP6.

## Tradeoffs

The shared scale must cover the whole block. An outlier can therefore reduce the effective precision of smaller values, and the implementation still has to choose the physical layout, accumulator precision, conversion policy, and operations around the dot product.

## Common Confusions

- **Microscaling vs. scalar FP4:** Microscaling describes the shared-scale block, not merely the bit width of each private element.
- **OCP MX vs. shared-microexponent MX:** The OCP MX Alliance formats use one E8M0 scale plus private elements; the ISCA 2023 BDR work uses a related but distinct hierarchy of shared microexponents.

## Where It Appears

- [Microscaling (MX) Formats: Block Floating Point for AI Hardware](../hardware/quantization/microscaling-mx-formats/index.md) - OCP MX block structure, concrete encodings, conversion rules, dot products, and hardware history.
- [MiniMax GQA W4A4 Quantization Path](../frameworks/vllm/minimax-gqa-w4a4-quantization-path.md) - Uses MXFP4 as one of the hardware-backed W4A4 paths.
- [Quantization](../hardware/quantization/index.md) - Groups MX with other low-precision numeric formats.
- [NVFP4: Blackwell 4-Bit Floating Point](../hardware/quantization/nvfp4.md) - Contrasts NVFP4's fractional, 16-element scaling with OCP MX's power-of-two, 32-element blocks.

## Related Terms

- [GEMM](gemm.md)
- [Inner Product](inner-product.md) - Scalar reduction used by MX dot products.
