---
title: "Algorithms"
summary: "Algorithm pages covering inference algorithms, attention kernels, and scheduling methods."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-07-15
---

# Algorithms

- [Matrix Exponentiation for Linear Transitions](matrix-exponentiation.md)
- [FlashAttention: IO-Aware Exact Attention](flashattention.md) — Original IO-aware exact attention algorithm: tiling, online softmax, recomputation, IO complexity, block-sparse extension, landscape evolutionary tree, and training/runtime results.
- [FlashAttention-2: Better Parallelism and Work Partitioning](flashattention-2.md) — Exact attention kernel optimization: reduced non-matmul overhead, sequence-parallel thread blocks, warp-level work partitioning, causal block skipping, Big Picture comparison diagram, landscape showing FA1→FA2 utilization gap closure, and A100/H100 performance results.
- [FlashAttention-3: Hopper Asynchrony and FP8 Attention](flashattention-3.md) — Hopper attention kernel design: warp specialization, TMA/WGMMA asynchrony, GEMM-softmax overlap, FP8 block quantization, incoherent processing, Big Picture pipeline diagram, landscape of Hopper hardware exploitation, and speed/accuracy results.
- [FlashAttention-4: Blackwell Attention Kernel Co-Design](flashattention-4.md) — Exact attention algorithm and Blackwell kernel design: asymmetric scaling response, exponential emulation, conditional softmax rescaling, TMEM-based pipelining, 2-CTA backward pass, LPT scheduling, landscape of shifting hardware bottlenecks, and performance results.
- [The Softmax Function: Properties, Motivation, and Interpretation](softmax.md) — Comprehensive tutorial on the softmax function: score-difference semantics, α parameter interpretation, three conceptual justifications (Gumbel noise, maximum entropy, exploration-exploitation), IO vs. IM model taxonomy, and complete mathematical properties with proofs.
