---
title: "Hardware and Numerics"
summary: "Hardware and numerics pages covering accelerator features, precision formats, and related implementation details."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-26
---

# Hardware and Numerics

## NVIDIA GPU Evolution

- [NVIDIA GPU Evolution](nvidia/index.md) — Topic hub for learning NVIDIA architecture generations and the expanding technology surface around them.
- [NVIDIA GPU Evolution: From Graphics to Accelerated Computing](nvidia/gpu-evolution-path.md) — A medium-confidence detailed comparison of Volta, Turing, Ampere, Hopper, Ada Lovelace, and Blackwell, plus the expanding technology surface around them.
- [NVIDIA Ada Lovelace Professional GPU Architecture](nvidia/ada-lovelace-professional-gpu-architecture/index.md) — AD102 hierarchy, cache, ray-tracing geometry engines, shader scheduling, neural graphics, FP8 Tensor Cores, and AV1 video.

## Ascend NPU

- [Ascend NPU](ascend/index.md) — Topic hub for Ascend NPU architecture, AI Core execution, memory hierarchy, and scale-up interconnects.
- [Ascend 950 NPU Architecture: Compute, Memory, and Unified Bus](ascend/ascend-950/index.md) — Third-generation DaVinci Cube/Vector compute, HiF8 and MX formats, chiplet UMA memory, NDDMA, STARS2.0, Unified Bus/URMA/CCU, DVPP, and 8,192-card supernodes.

## Kernels and Dataflow

- [DeepGEMM MegaMoE: Fused Communication and Expert Compute](deepgemm/index.md) — Beginner-oriented code reading of the FP8xFP4 fused MoE path: symmetric-memory dispatch, bounded ring scheduling, SM100 tensor-core execution, SwiGLU, and top-k combine.
- [Spatial GEMM: Blocked Outer-Product Matrix Multiply](spatial-gemm.md) — Blocked outer-product [GEMM](../terms/gemm.md) in the Spatial DSL: output tiling, MemFold/MemReduce pipelining, triple buffering, and multi-dimensional [SRAM banking](../terms/memory-banking.md).

## Quantization

- [Quantization](quantization/index.md) — Category hub for post-training quantization methods and low-precision numeric formats.
- [GPTQ: Second-Order Weight Quantization at LLM Scale](quantization/gptq/index.md)
- [FlatQuant: Fast Learnable Affine Quantization](quantization/flatquant/index.md)
- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](quantization/quarot/index.md)
- [NVFP4: Blackwell 4-Bit Floating Point](quantization/nvfp4.md)
