---
title: "Sliding-Window Attention"
summary: "Causal attention that lets each token read only a fixed number of preceding tokens, keeping its KV state bounded by the window size."
tooltip: "Sliding-window attention applies a local causal mask so the latest token attends only to the previous C positions. It reduces memory and compute growth, but distant information must be carried by other layers or a separate global memory."
layout: default
confidence: high
category: algorithms
sources:
  - raw/training/yoco-you-only-cache-once--arxiv-2405.05254v2.pdf
  - raw/training/swat-sliding-window-attention-training--paper.pdf
aliases:
  - SWA
  - local attention
  - local/sliding-window attention
appears_in:
  - docs/training/efficient-attention/yoco/index.md
updated: 2026-09-10
---

# Sliding-Window Attention

**Sliding-Window Attention** is causal attention that restricts each token to a fixed window of preceding tokens instead of the entire prefix.

## Why It Exists

The KV state of full causal attention grows with sequence length. A fixed window keeps the active history bounded, making long-context inference and training cheaper when a separate mechanism can preserve or recover information outside the window.

## How It Works

For window size $C$, a query at position $i$ can attend to keys $j$ satisfying $i-C < j \leq i$; all other scores are masked to $-\infty$. The explicit KV state is therefore $\mathcal O(C)$ per layer rather than $\mathcal O(N)$.

YOCO uses SWA as an alternative self-decoder module: the self-decoder produces the one global cache consumed by its cross-decoder, so the model can keep a local computation path while retaining a global readout path.

## Tradeoffs

- A small window lowers memory but can lose dependencies that are never carried into the global representation.
- Local attention is simpler than a recurrent state, but it still stores explicit keys and values for $C$ positions.
- SWA does not itself remove a conventional cross-decoder cache; it is complementary to cache sharing or compression.

## Common Confusions

- **Sliding-window attention vs. sparse global attention:** SWA has a fixed contiguous neighborhood; sparse global patterns may add selected distant positions.
- **Sliding-window attention vs. gated retention:** SWA keeps explicit recent tokens, while retention compresses history into a recurrent state.

## Where It Appears

- [YOCO: You Only Cache Once](../training/efficient-attention/yoco/index.md) — Uses SWA as the simpler constant-memory self-decoder alternative to gated retention.

## Related Terms

- [Gated Retention](gated-retention.md) — A recurrent fixed-state alternative for YOCO's self-decoder.
- [KV Cache](kv-cache.md) — The explicit key/value history whose size SWA bounds by the window.
- [Linear Attention](linear-attention.md) — A broader family of fixed-state attention mechanisms.
