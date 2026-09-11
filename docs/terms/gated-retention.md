---
title: "Gated Retention"
summary: "A causal recurrent attention mechanism whose learned, head-wise decay controls how much prior key-value state survives each token."
tooltip: "Gated retention combines a data-dependent decay with recurrent key-value state. Its parallel, recurrent, and chunkwise forms are mathematically equivalent, so training can use matrix operations while inference keeps a fixed-size state."
layout: default
confidence: high
category: algorithms
sources:
  - raw/training/yoco-you-only-cache-once--arxiv-2405.05254v2.pdf
aliases:
  - gRet
  - gRetNet
  - RetNet-3
appears_in:
  - docs/training/efficient-attention/yoco/index.md
updated: 2026-09-10
---

# Gated Retention

**Gated Retention** is a causal recurrent attention mechanism whose learned, head-wise decay controls how much prior key-value state survives each token.

## Why It Exists

Full self-attention stores a growing history, while a fixed recurrent state can forget useful information or cannot adapt its memory horizon. Gated retention makes the horizon data-dependent: a token can preserve a head's state or decay it before writing new content.

## How It Works

For token $n$, the recurrent form maintains a matrix state and reads it with the query:

$$
S_n = \gamma_n S_{n-1} + K_n^\top V_n, \qquad o_n = Q_n S_n.
$$

The head-wise gate $\gamma_n$ is predicted from the current input. The paper gives equivalent parallel, recurrent, and chunkwise-recurrent forms: the parallel form is convenient for training, the recurrent form has constant inference memory, and the chunkwise form uses dense work inside a chunk while carrying one boundary state between chunks.

## Tradeoffs

- A fixed-size state removes the token-level history, so retrieval quality depends on learned decay and state capacity.
- Head-wise rather than element-wise decay is less expressive but maps cleanly to GPU tensor cores.
- Chunkwise execution improves hardware utilization, but requires a careful equivalent formulation and a tuned chunk size.

## Common Confusions

- **Gated retention vs. sliding-window attention:** retention summarizes history in a recurrent state; a sliding window preserves explicit keys and values for only the latest $C$ tokens.
- **Gated retention vs. YOCO:** gRet is one possible self-decoder mechanism inside YOCO; YOCO's decoder-decoder cache topology is the larger architectural contribution.

## Where It Appears

- [YOCO: You Only Cache Once](../training/efficient-attention/yoco/index.md) — Uses gated retention as the default self-decoder mechanism, switching between parallel/chunkwise prefill and recurrent generation.

## Related Terms

- [Linear Attention](linear-attention.md) — Accumulates key-value statistics into a fixed-size recurrent state.
- [Sliding-Window Attention](sliding-window-attention.md) — Bounds explicit attention history to a fixed local window.
- [KV Cache](kv-cache.md) — Stores explicit token-level keys and values in conventional decoder layers.
