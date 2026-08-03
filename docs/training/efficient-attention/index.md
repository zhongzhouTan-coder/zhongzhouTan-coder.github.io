---
title: "Efficient Attention Training"
summary: "Training approaches built around efficient attention mechanisms: sparse, sliding-window, and delta-rule recurrent attention."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-03
---

# Efficient Attention Training

- [MiniMax Sparse Attention (MSA)](minimax-sparse-attention/index.md) — Blockwise sparse attention co-designed with GQA: lightweight Index Branch selects top-k KV blocks per GQA group, Main Branch computes exact softmax attention over only the selected blocks, trained with KL alignment loss.
- [SWAT: Sliding Window Attention Training](swat-sliding-window-attention/index.md) — Sigmoid-based sliding window attention training: replaces softmax with sigmoid to eliminate attention sink, combines balanced bidirectional ALiBi with RoPE for training stability.
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](gated-delta-networks/index.md) — Fixed-state recurrent memory combining global adaptive decay with key-targeted correction, decay-aware chunkwise WY training, and SWA/Mamba2 hybrids.
