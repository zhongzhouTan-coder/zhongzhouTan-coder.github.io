---
title: "Algorithms"
summary: "Algorithm pages covering inference algorithms, attention kernels, and scheduling methods."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-13
---

# Algorithms

## Foundations

- [Foundations](foundations/index.md) — Category hub for foundational sequence-modeling concepts.
- [The Transformer: Attention Is All You Need](foundations/transformer.md) — The foundational architecture that replaces recurrence with multi-head self-attention: scaled dot-product attention, multi-head parallelism, sinusoidal positional encoding, encoder-decoder stacks, and the training recipe that birthed modern LLMs.
- [Layer Normalization in Transformers](foundations/layer-normalization/index.md) ([term](../terms/layer-normalization.md)) — Token-local normalization across hidden features, contrasted with BatchNorm through equations, source diagrams, runtime behavior, and implementation pitfalls.
- [The Softmax Function: Properties, Motivation, and Interpretation](foundations/softmax.md) — Comprehensive tutorial on the softmax function: score-difference semantics, α parameter interpretation, three conceptual justifications (Gumbel noise, maximum entropy, exploration-exploitation), IO vs. IM model taxonomy, and complete mathematical properties with proofs.
- [Recurrent Neural Networks: From RNN to LSTM](foundations/recurrent-neural-networks/index.md) — Beginner-oriented path from sequence order and shared recurrent weights to hidden-state memory, long-term dependency failures, LSTM gates, and [linear attention](../terms/linear-attention.md)'s RNN execution mode.

## Linear Algebra Foundations

- [Kronecker Product](kronecker-product.md) — The block-structured matrix product A⊗B ([matrix direct product](../terms/kronecker-product.md)), foundational to tensor factorization and the Kronecker factorization trick behind FlatQuant's learnable affine transforms.

## FlashAttention

- [FlashAttention](flashattention/index.md) — Category hub for the FlashAttention algorithm and kernel family.
- [FlashAttention: IO-Aware Exact Attention](flashattention/flashattention.md) — Original IO-aware exact attention algorithm: tiling, online softmax, recomputation, IO complexity, block-sparse extension, landscape evolutionary tree, and training/runtime results.
- [FlashAttention-2: Better Parallelism and Work Partitioning](flashattention/flashattention-2.md) — Exact attention kernel optimization: reduced non-matmul overhead, sequence-parallel thread blocks, warp-level work partitioning, causal block skipping, Big Picture comparison diagram, landscape showing FA1→FA2 utilization gap closure, and A100/H100 performance results.
- [FlashAttention-3: Hopper Asynchrony and FP8 Attention](flashattention/flashattention-3.md) — Hopper attention kernel design: warp specialization, TMA/WGMMA asynchrony, [GEMM](../terms/gemm.md)-softmax overlap, FP8 block quantization, incoherent processing, Big Picture pipeline diagram, landscape of Hopper hardware exploitation, and speed/accuracy results.
- [FlashAttention-4: Blackwell Attention Kernel Co-Design](flashattention/flashattention-4.md) — Exact attention algorithm and Blackwell kernel design: asymmetric scaling response, exponential emulation, conditional softmax rescaling, TMEM-based pipelining, 2-CTA backward pass, LPT scheduling, landscape of shifting hardware bottlenecks, and performance results.

## Attention Variants

- [Attention Variants](attention-variants/index.md) — Category hub for attention designs that reduce query/key redundancy or KV-cache pressure.
- [Multi-Query Attention: One Write-Head is All You Need](attention-variants/multi-query-attention.md) — Eliminates per-head key/value projections, sharing one K/V across all heads; reduces incremental decoder inference cost 12× (46→3.8 µs/token) with negligible quality loss; orthogonal to local attention and ancestor of Grouped-Query Attention (GQA).
- [Grouped-Query Attention in Llama 2](attention-variants/grouped-query-attention/index.md) — Llama 2's 34B/70B attention choice: use 8 KV groups to cut KV-cache pressure while preserving more quality and cleaner tensor-parallel serving than single-KV-head MQA.
- [Collaborative Multi-Head Attention: Collaborate Instead of Concatenate](attention-variants/collaborative-attention.md) — Redesigns MHA with shared key/query projections and per-head mixing vectors, enabling 4× compression of Q/K dimensions; CP tensor decomposition for post-hoc conversion of pretrained models.
- [DeepSeek-V2 Multi-Head Latent Attention](attention-variants/deepseek-v2-mla.md) — DeepSeek-V2's MLA design: low-rank joint K/V latent cache, decoupled RoPE side channel, query compression, [MoE](../terms/mixture-of-experts.md) serving context, and reported 93.3% KV-cache reduction with 5.76× maximum generation throughput versus DeepSeek 67B.

## Linear Attention

- [Transformers Are RNNs: Linear Attention](linear-attention/index.md) — Kernel feature maps and associative reordering eliminate the quadratic attention matrix; causal prefix states expose a fixed-size recurrent interpretation and its accuracy/throughput tradeoffs.
- [Linear Attention Without Softmax: Su Jianlin's Survey](linear-attention/linear-attention-without-softmax.md) — 苏剑林's blog survey identifying softmax as the root cause of attention's O(n²) complexity; catalogs three linear attention families (kernel maps, double-softmax, cosine-similarity Taylor approximation) and their autoregressive generation support.
- [Matrix Exponentiation for Linear Transitions](linear-attention/matrix-exponentiation.md) — Binary matrix exponentiation, transition-matrix construction, linear recurrences, augmented state vectors, and fixed linear dynamic programming.

## Distributed Inference

- [Context Parallelism for Scalable Million-Token Inference](context-parallelism/index.md) — Exact pass-KV/pass-Q ring attention, load-balanced sequence sharding, adaptive cache-aware traffic selection, and 128-GPU million-token prefill scaling.

## Model Papers

- [DeepSeek-V3.2: Sparse Attention, Scaled RL, and Thinking in Tool-Use](deepseek-v3.2/index.md) — DeepSeek-V3.2's three innovations: DSA sparse attention with a learnable [lightning indexer](../terms/lightning-indexer.md) and top-k token selection reducing core complexity to $O(L \cdot k)$, a scaled GRPO recipe with four stabilization tricks (unbiased KL, off-policy masking, Keep Routing, Keep Sampling Mask), and a cold-start + synthetic agentic task pipeline (1,827 environments, 85K prompts) that unifies chain-of-thought reasoning with tool-use at scale.
