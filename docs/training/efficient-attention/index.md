---
title: "Efficient Attention Training"
summary: "Training approaches built around efficient attention mechanisms: sparse, sliding-window, and delta-rule recurrent attention."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-09-10
---

# Efficient Attention Training

- [MiniMax Sparse Attention (MSA)](minimax-sparse-attention/index.md) — Blockwise sparse attention co-designed with GQA: lightweight [Index Branch](../../terms/lightning-indexer.md) selects top-k KV blocks per GQA group, Main Branch computes exact softmax attention over only the selected blocks, trained with KL alignment loss.
- [SWAT: Sliding Window Attention Training](swat-sliding-window-attention/index.md) — Sigmoid-based sliding window attention training: replaces softmax with sigmoid to eliminate attention sink, combines balanced bidirectional ALiBi with RoPE for training stability.
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](gated-delta-networks/index.md) — Fixed-state recurrent memory combining global adaptive decay with key-targeted correction ([Delta Rule](../../terms/delta-rule.md)); decay-aware chunkwise WY training, and SWA/Mamba2 hybrids.
- [YOCO: You Only Cache Once](yoco/index.md) — Decoder-decoder architecture with a bounded-memory self-decoder, one shared global KV cache, causal cross-decoder layers, and chunk-parallel long-context training.
- [Universal YOCO for Efficient Depth Scaling](yoco/universal-yoco.md) — Loops YOCO's shallow efficient-attention self-decoder with shared parameters so recursion buys effective depth while the global KV cache is still written once; reports 41.78 → 47.08 average downstream scores, near-YOCO KV memory, and flat prefill throughput from 8K to 256K.
