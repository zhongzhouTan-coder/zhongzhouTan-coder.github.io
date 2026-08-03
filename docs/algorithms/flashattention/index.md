---
title: "FlashAttention"
summary: "The FlashAttention algorithm and kernel family across GPU generations: IO-aware exact attention, parallelism, Hopper asynchrony, and Blackwell co-design."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-03
---

# FlashAttention

- [FlashAttention: IO-Aware Exact Attention](flashattention.md) — Original IO-aware exact attention algorithm: tiling, online softmax, recomputation, IO complexity, block-sparse extension, landscape evolutionary tree, and training/runtime results.
- [FlashAttention-2: Better Parallelism and Work Partitioning](flashattention-2.md) — Exact attention kernel optimization: reduced non-matmul overhead, sequence-parallel thread blocks, warp-level work partitioning, causal block skipping, and A100/H100 performance results.
- [FlashAttention-3: Hopper Asynchrony and FP8 Attention](flashattention-3.md) — Hopper attention kernel design: warp specialization, TMA/WGMMA asynchrony, GEMM-softmax overlap, FP8 block quantization, incoherent processing, and speed/accuracy results.
- [FlashAttention-4: Blackwell Attention Kernel Co-Design](flashattention-4.md) — Exact attention algorithm and Blackwell kernel design: asymmetric scaling response, exponential emulation, conditional softmax rescaling, TMEM-based pipelining, 2-CTA backward pass, LPT scheduling, and performance results.
