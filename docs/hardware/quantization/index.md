---
title: "Quantization"
summary: "Post-training quantization methods and low-precision numeric formats for LLM inference."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-25
---

# Quantization

- [GPTQ: Second-Order Weight Quantization at LLM Scale](gptq/index.md) — Activation-derived error compensation, shared column ordering, lazy block updates, Cholesky stability, 175B-scale calibration, and packed-weight decode results.
- [MX Formats: Block Floating Point for AI Hardware](microscaling-mx-formats/index.md) — [Microscaling](../../terms/microscaling.md) history and hardware analysis, OCP's 32-element MXFP4/6/8 and MXINT8 family, specification boundaries, and mixed-format compute flow.
- [HiFloat4 (HiF4): 4-Bit Block Floating Point for LLM Inference](hif4/index.md) — 64 S1P2 payloads plus E6M2/E1 micro-exponent metadata, BF16 conversion, mostly-integer 64-wide dot products, MSE and area/power analysis, and LLM inference comparisons against NVFP4.
- [FlatQuant: Fast Learnable Affine Quantization](flatquant/index.md) — Post-training LLM quantization method: learnable affine transformations, Kronecker factorization, per-channel scaling, learnable clipping, fused kernels, W4A4 accuracy, and inference latency results.
- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](quarot/index.md) — Fused Hadamard rotations that remove activation and KV-cache outliers, enabling end-to-end 4-bit weights, activations, and KV cache without higher-precision outlier features.
- [NVFP4: Blackwell 4-Bit Floating Point](nvfp4.md) — NVIDIA Blackwell NVFP4 data-format contract: E2M1 payloads, fractional [FP8](../../terms/fp8.md) E4M3 micro-block scales, FP32 tensor scaling, 16-element/16×16 layouts, plus training and [GEMM](../../terms/gemm.md) constraints.
