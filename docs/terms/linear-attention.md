---
title: "Linear Attention"
summary: "An attention family that factors query–key similarity through feature maps so key–value associations can be accumulated without an explicit quadratic attention matrix."
tooltip: "Linear attention replaces the full token-by-token attention matrix with a fixed-size key–value summary. It makes sequence processing linear in length and causal decoding recurrent, but usually changes or approximates softmax and can lose exact token-level retrieval."
layout: default
confidence: high
category: algorithms
sources:
  - raw/algorithms/transformers-are-rnns-linear-attention--arxiv-2006.16236v3.pdf
  - raw/training/kimi-linear-expressive-efficient-attention--paper.pdf
aliases:
  - linear transformer
  - kernelized attention
appears_in:
  - docs/algorithms/linear-attention/index.md
  - docs/training/kimi-linear/index.md
updated: 2026-07-29
---

# Linear Attention

**Linear Attention** is an attention family that factors query–key similarity through finite-dimensional feature maps, allowing values to be read from an accumulated key–value summary instead of an explicit token-by-token attention matrix.

## Why It Exists

Softmax attention constructs an $N \times N$ score matrix, so its direct time and memory grow quadratically with sequence length. Autoregressive decoding also retains a history whose size and read cost grow with context. Linear attention seeks linear full-sequence cost and a fixed-size recurrent state.

## How It Works

For a non-negative feature map $\phi$, associativity changes the evaluation order:

$$
(\phi(Q)\phi(K)^T)V=\phi(Q)(\phi(K)^TV).
$$

The right side summarizes keys and values before applying queries. Under causal masking, prefix states $S_i=\sum_{j\le i}\phi(K_j)V_j^T$ and $Z_i=\sum_{j\le i}\phi(K_j)$ are updated once per token and queried as $\phi(Q_i)^TS_i/\phi(Q_i)^TZ_i$.

Modern variants add learned forgetting, delta-rule updates, convolution, or occasional full-attention layers to make the fixed-capacity state more selective.

## Tradeoffs

Linear attention usually changes or approximates the softmax kernel, and its fixed-size state can blur distinct past tokens. It excels when context length dominates feature dimensions and exact random access is less important; hybrids retain full-attention layers for difficult retrieval cases.

## Common Confusions

- **Linear attention vs. FlashAttention:** Linear attention changes or approximates the attention rule to avoid quadratic pairwise scores; FlashAttention computes exact softmax attention with IO-aware tiling.
- **Linear attention vs. sparse attention:** Linear attention compresses all history into a summary; sparse attention retains explicit access to a selected subset of tokens.
- **Linear attention vs. an ordinary RNN:** Causal linear attention has a recurrent matrix state, but keeps Transformer-style query, key, value projections and can use parallel prefix-style training.

## Where It Appears

- [Transformers Are RNNs: Linear Attention](../algorithms/linear-attention/index.md) — Establishes kernel factorization, associative reordering, and the causal recurrent formulation.
- [Kimi Linear](../training/kimi-linear/index.md) — Extends the family with channel-wise forgetting, delta-rule updates, and periodic full-attention layers.

## Related Terms

- [KV Cache](kv-cache.md) — Explicit key/value history used by conventional autoregressive attention.
