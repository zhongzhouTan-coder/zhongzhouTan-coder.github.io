---
title: "Quantization"
summary: "Post-training quantization methods and low-precision numeric formats for LLM inference."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-03
---

# Quantization

- [FlatQuant: Fast Learnable Affine Quantization](flatquant/index.md) — Post-training LLM quantization method: learnable affine transformations, Kronecker factorization, per-channel scaling, learnable clipping, fused kernels, W4A4 accuracy, and inference latency results.
- [NVFP4: Blackwell 4-Bit Floating Point](nvfp4.md) — NVIDIA Blackwell NVFP4 format with two-level hierarchical FP8/FP32 scaling, fractional E4M3 vs. power-of-two E8M0 comparison, 16-element micro-block quantization, Random Hadamard Transform, stochastic rounding, 2D weight scaling, GEMM layout constraints, distributed training behavior, and deployment ecosystem.
