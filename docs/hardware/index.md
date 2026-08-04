---
title: "Hardware and Numerics"
summary: "Hardware and numerics pages covering accelerator features, precision formats, and related implementation details."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-07-15
---

# Hardware and Numerics

## Kernels and Dataflow

- [Spatial GEMM: Blocked Outer-Product Matrix Multiply](spatial-gemm.md) — Blocked outer-product GEMM in the Spatial DSL: output tiling, MemFold/MemReduce pipelining, triple buffering, and multi-dimensional SRAM banking.

## Quantization

- [Quantization](quantization/index.md) — Category hub for post-training quantization methods and low-precision numeric formats.
- [FlatQuant: Fast Learnable Affine Quantization](quantization/flatquant.md)
- [NVFP4: Blackwell 4-Bit Floating Point](quantization/nvfp4.md)
