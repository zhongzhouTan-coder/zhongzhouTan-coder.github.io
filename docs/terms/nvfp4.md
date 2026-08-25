---
title: "NVFP4"
summary: "NVFP4 is NVIDIA's four-bit floating-point format with E2M1 payloads, FP8 E4M3 micro-block scales, and a tensor-level scale in the public Blackwell recipe."
tooltip: "NVFP4 stores four-bit E2M1 values and uses fractional FP8 E4M3 scales for groups of 16 values; NVIDIA's public training recipe adds a separate FP32 tensor scale. It improves local range fitting over power-of-two MXFP4, but its scale range, layout, and Blackwell hardware requirements matter in practice."
layout: default
confidence: high
category: hardware
sources:
  - raw/hardware/nvfp4-te--web-2026-07-28-6625830e9e9e.html
  - raw/hardware/nvfp4-te--web-2026-07-28-6625830e9e9e.metadata.json
  - raw/hardware/nvfp4-blog--web-2026-07-28-a2f3eb0ba3bb.html
  - raw/hardware/nvfp4-blog--web-2026-07-28-a2f3eb0ba3bb.metadata.json
  - raw/hardware/hif4-format-for-language-model-inference--arxiv-2602.11287v1.pdf
aliases:
  - NVIDIA FP4
mention_lint: canonical
appears_in:
  - docs/hardware/quantization/nvfp4.md
  - docs/hardware/quantization/hif4/index.md
  - docs/hardware/quantization/microscaling-mx-formats/index.md
updated: 2026-08-25
---

# NVFP4

**NVFP4** is NVIDIA's four-bit floating-point format for Blackwell-era tensor computation, combining E2M1 payload values with fractional FP8 E4M3 micro-block scales and a tensor-level scale in the public recipe.

## Why It Exists

Plain FP4 has too few representable values for model weights and activations. NVFP4 uses a scale for each 16-value micro-block so the E2M1 payload can fit the local magnitude more closely than a single power-of-two scale.

## How It Works

The public NVIDIA recipe reconstructs values from a four-bit E2M1 payload, an E4M3 scale for each 16-element block, and an FP32 scale for the tensor. Training adds recipe-level mechanisms such as stochastic rounding, 2D weight scaling, and Random Hadamard Transform support; inference and training also depend on the supported Blackwell GEMM layout.

## Tradeoffs

Fractional scales and small blocks improve local fitting but require more scale handling and can need global scaling to keep E4M3 in range. The format is not a drop-in software type: hardware support, layout, conversion policy, and workload distribution determine the actual benefit.

## Common Confusions

- **NVFP4 vs. HiF4:** Both cost about 4.5 bits/value in the compared designs, but NVFP4 uses 16-value E2M1 blocks with fractional scales, while HiF4 uses 64 S1P2 values plus shared E1 micro-exponents.
- **NVFP4 vs. generic FP4:** NVFP4 is a hierarchical scaled format and deployment recipe, not just an unscaled four-bit E2M1 scalar.

## Where It Appears

- [NVFP4: Blackwell 4-Bit Floating Point](../hardware/quantization/nvfp4.md) — Public Transformer Engine and NVIDIA blog recipe, training mechanisms, layout limits, and deployment context.
- [HiFloat4 (HiF4): 4-Bit Block Floating Point for LLM Inference](../hardware/quantization/hif4/index.md) — Compares NVFP4 against a 64-value shared-metadata alternative for error, hardware, and LLM accuracy.
- [Microscaling (MX) Formats: Block Floating Point for AI Hardware](../hardware/quantization/microscaling-mx-formats/index.md) — Places NVFP4 alongside OCP MX and related block-floating designs.

## Related Terms

- [FP8](fp8.md)
- [Microscaling](microscaling.md)
- [Block Floating Point](block-floating-point.md)
- [HiFloat4 (HiF4)](hif4.md)
