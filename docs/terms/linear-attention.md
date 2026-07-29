---
title: "Linear Attention"
summary: "An attention family that factors query–key similarity through feature maps so key–value associations can be accumulated without an explicit quadratic attention matrix."
tooltip: "Linear attention replaces the full token-by-token attention matrix with a fixed-size key–value summary. It makes sequence processing linear in length and causal decoding recurrent, but usually changes or approximates softmax and can lose exact token-level retrieval."
layout: default
confidence: high
category: algorithms
sources:
  - raw/algorithms/transformers-are-rnns-linear-attention--arxiv-2006.16236v3.pdf
  - raw/training/gated-delta-networks-improving-mamba2-with-delta-rule--arxiv-2412.06464.pdf
  - raw/training/kimi-linear-expressive-efficient-attention--paper.pdf
  - raw/training/k3-technical-report--paper.pdf
aliases:
  - linear transformer
  - kernelized attention
appears_in:
  - docs/algorithms/linear-attention/index.md
  - docs/training/gated-delta-networks/index.md
  - docs/training/kimi-linear/index.md
  - docs/training/kimi-k3/index.md
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

The right side summarizes keys and values before applying queries. Under causal masking, prefix states $S_i=\sum_{j\le i}\phi(K_j)V_j^T$ (a $C \times M$ matrix) and $Z_i=\sum_{j\le i}\phi(K_j)$ (a $C$-vector) are updated once per token and queried as $\phi(Q_i)^TS_i/\phi(Q_i)^TZ_i$, where the numerator is an $M$-vector weighted value sum and the denominator is the scalar sum of the same similarity weights.

Modern variants add learned forgetting, delta-rule updates, convolution, or occasional full-attention layers to make the fixed-capacity state more selective.

## How to Think About Queries, Keys, and Values

In both traditional and linear attention, $Q$, $K$, and $V$ come from the same token representation ($Q = XW_Q$, $K = XW_K$, $V = XW_V$) but serve different roles. The model learns $W_K$ and $W_V$ independently so addressing and content can **specialize**.

| Role | What it answers | Analogy |
|---|---|---|
| **$K$ (key)** | "How do I advertise this token so relevant queries find it?" | The **address label** — which folders to file in, and how strongly |
| **$V$ (value)** | "What useful information should this token contribute?" | The **payload** — the document being filed |
| **$Q$ (query)** | "Given my current context, which past tokens do I need?" | The **search** — which folders to read from, and how strongly |

The routing weight is always a **$Q$–$K$ match**, not $K$ alone: in traditional attention it's $\text{softmax}(q_i^T k_j)$, and in linear attention it's $\phi(Q_i)^T \phi(K_j)$. $K$ provides the address; $Q$ provides the search criteria.

### The state matrix as a filing cabinet

$S_i = \sum_{j \le i} \phi(K_j) V_j^T$ is a $C \times M$ association table — rows are feature types, columns are value dimensions. Each new token adds a rank-1 contribution: $\phi(K_j)$ says *how strongly* this token belongs to each feature type, and $V_j$ is *what content* gets stored under those types. A query $\phi(Q_i)$ reads the cabinet by taking a weighted blend of rows.

This is the same filing-cabinet intuition as traditional attention — the difference is that linear attention **compresses the cabinet first** ($S_i$) and then queries it, while traditional attention flips through every individual folder ($k_j$) for every query.

## Tradeoffs

Linear attention usually changes or approximates the softmax kernel, and its fixed-size state can blur distinct past tokens. It excels when context length dominates feature dimensions and exact random access is less important; hybrids retain full-attention layers for difficult retrieval cases.

## Common Confusions

- **Linear attention vs. FlashAttention:** Linear attention changes or approximates the attention rule to avoid quadratic pairwise scores; FlashAttention computes exact softmax attention with IO-aware tiling.
- **Linear attention vs. sparse attention:** Linear attention compresses all history into a summary; sparse attention retains explicit access to a selected subset of tokens.
- **Linear attention vs. an ordinary RNN:** Causal linear attention has a recurrent matrix state, but keeps Transformer-style query, key, value projections and can use parallel prefix-style training.

## Where It Appears

- [Transformers Are RNNs: Linear Attention](../algorithms/linear-attention/index.md) — Establishes kernel factorization, associative reordering, and the causal recurrent formulation.
- [Gated Delta Networks](../training/gated-delta-networks/index.md) — Adds adaptive global decay and key-targeted delta updates to a matrix-valued recurrent state.
- [Kimi Linear](../training/kimi-linear/index.md) — Extends the family with channel-wise forgetting, delta-rule updates, and periodic full-attention layers.
- [Kimi K3](../training/kimi-k3/index.md) — Scales Kimi-family linear attention to a 2.8T MoE with lower-bounded decay for BF16 Tensor Core kernels.

## Related Terms

- [KV Cache](kv-cache.md) — Explicit key/value history used by conventional autoregressive attention.
- [Delta Rule](delta-rule.md) — Key-targeted online error correction for associative memory.
- [Kimi Delta Attention](kimi-delta-attention.md) — A channel-wise gated delta-rule member of the linear-attention family.
