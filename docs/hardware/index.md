---
title: "Hardware and Numerics"
summary: "Hardware and numerics pages covering accelerator features, precision formats, and related implementation details."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-22
---

# Hardware and Numerics

## NVIDIA GPU Evolution

- [NVIDIA GPU Evolution](nvidia/index.md) — Topic hub for learning NVIDIA architecture generations and the expanding technology surface around them.
- [NVIDIA GPU Evolution: From Graphics to Accelerated Computing](nvidia/gpu-evolution-path.md) — A medium-confidence detailed comparison of Volta, Turing, Ampere, Hopper, Ada Lovelace, and Blackwell, plus the expanding technology surface around them.

## Kernels and Dataflow

- [Spatial GEMM: Blocked Outer-Product Matrix Multiply](spatial-gemm.md) — Blocked outer-product [GEMM](../terms/gemm.md) in the Spatial DSL: output tiling, MemFold/MemReduce pipelining, triple buffering, and multi-dimensional [SRAM banking](../terms/memory-banking.md).

## Quantization

- [Quantization](quantization/index.md) — Category hub for post-training quantization methods and low-precision numeric formats.
- [GPTQ: Second-Order Weight Quantization at LLM Scale](quantization/gptq/index.md)
- [FlatQuant: Fast Learnable Affine Quantization](quantization/flatquant/index.md)
- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](quantization/quarot/index.md)
- [NVFP4: Blackwell 4-Bit Floating Point](quantization/nvfp4.md)
