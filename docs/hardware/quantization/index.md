---
title: "Quantization"
summary: "Post-training quantization methods and low-precision numeric formats for LLM inference."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-16
---

# Quantization

- [GPTQ: Second-Order Weight Quantization at LLM Scale](gptq/index.md) — Activation-derived error compensation, shared column ordering, lazy block updates, Cholesky stability, 175B-scale calibration, and packed-weight decode results.
- [MX Formats: Block Floating Point for AI Hardware](microscaling-mx-formats/index.md) — [Microscaling](../../terms/microscaling.md) history and hardware analysis, OCP's 32-element MXFP4/6/8 and MXINT8 family, specification boundaries, and mixed-format compute flow.
- [FlatQuant: Fast Learnable Affine Quantization](flatquant/index.md) — Post-training LLM quantization method: learnable affine transformations, Kronecker factorization, per-channel scaling, learnable clipping, fused kernels, W4A4 accuracy, and inference latency results.
- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](quarot/index.md) — Fused Hadamard rotations that remove activation and KV-cache outliers, enabling end-to-end 4-bit weights, activations, and KV cache without higher-precision outlier features.
- [NVFP4: Blackwell 4-Bit Floating Point](nvfp4.md) — NVIDIA Blackwell NVFP4 format with two-level hierarchical [FP8](../../terms/fp8.md)/FP32 scaling, fractional E4M3 vs. power-of-two E8M0 comparison, 16-element micro-block quantization, Random Hadamard Transform, stochastic rounding, 2D weight scaling, [GEMM](../../terms/gemm.md) layout constraints, distributed training behavior, and deployment ecosystem.
