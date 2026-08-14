---
title: "Attention Foundations"
summary: "Foundational sequence-modeling concepts behind attention-based LLMs: the Transformer, softmax, and RNN-to-LSTM background."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-13
---

# Attention Foundations

- [The Transformer: Attention Is All You Need](transformer.md) — The foundational architecture: scaled dot-product attention, multi-head self-attention, sinusoidal positional encoding, encoder-decoder stacks, and the training recipe that launched modern LLMs.
- [Layer Normalization in Transformers](layer-normalization/index.md) ([term](../../terms/layer-normalization.md)) — Why each token is normalized across hidden features rather than across a batch, with equations, axis diagrams, implementation cautions, and a worked trace.
- [The Softmax Function: Properties, Motivation, and Interpretation](softmax.md) — Tutorial covering score-difference semantics, α parameter interpretation, three conceptual justifications, IO vs. IM model taxonomy, and complete mathematical properties.
- [Recurrent Neural Networks: From RNN to LSTM](recurrent-neural-networks/index.md) — Sequence processing through shared recurrent weights and hidden state, the long-term dependency problem, LSTM gating, and the conceptual bridge to [linear attention](../../terms/linear-attention.md)'s RNN mode.
