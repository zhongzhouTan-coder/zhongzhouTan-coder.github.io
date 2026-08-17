---
title: "Tree Attention"
summary: "An ancestry-aware attention pattern that evaluates many speculative-decoding branches in one transformer pass without cross-branch information leakage."
tooltip: "Tree attention flattens a candidate tree for batched transformer execution while allowing each node to attend only to its ancestors. It lets speculative decoders verify multiple possible continuations together without making sibling branches condition on one another."
layout: default
confidence: high
category: frameworks
sources:
  - raw/frameworks/eagle-speculative-sampling-feature-uncertainty--arxiv-2401.15077v3.pdf
  - raw/frameworks/eagle-2-dynamic-draft-trees--arxiv-2406.16858v2.pdf
aliases:
  - tree-structured attention
mention_lint: canonical
appears_in:
  - docs/frameworks/eagle/index.md
  - docs/frameworks/eagle-2/index.md
updated: 2026-08-17
---

# Tree Attention

**Tree Attention** is an ancestry-aware attention pattern that evaluates multiple autoregressive branches together while preventing tokens on different branches from seeing one another.

## Why It Exists

Speculative decoders often draft several alternative continuations. Running each branch separately wastes parallel capacity, but flattening every node under an ordinary causal mask leaks sibling tokens into one another and no longer represents valid autoregressive sequences.

## How It Works

The runtime flattens tree nodes into one token sequence and builds a custom mask. A node may attend to the verified prefix, itself, and its ancestors; it cannot attend to a sibling or any node descended from another branch. The transformer can therefore score the tree in one batched pass while each node's hidden state matches its own root-to-node history.

During verification, the speculative-decoding acceptance rule follows one valid path through the scored tree. The flat tensor is an execution layout; the mask preserves the logical tree.

## Tradeoffs

Tree attention increases candidate coverage without adding a target-model pass, but it processes more token positions and requires tree construction, flattening, position bookkeeping, and a nonstandard mask. It helps only when extra accepted tokens repay that overhead.

## Common Confusions

- **Tree attention vs. a static draft tree:** Tree attention is the execution and masking mechanism; the tree shape may be fixed, learned, or selected dynamically.
- **Tree attention vs. ordinary causal attention:** Causal attention exposes all earlier flattened positions. Tree attention exposes only earlier positions on the same ancestor path.

## Where It Appears

- [EAGLE: Feature-Level Speculative Sampling](../frameworks/eagle/index.md) - Builds a multi-branch feature-level draft and verifies it in one target-model pass.
- [EAGLE-2: Context-Aware Dynamic Draft Trees](../frameworks/eagle-2/index.md) - Changes the tree per context, then constructs the corresponding ancestry mask after reranking.

## Related Terms

- [Speculative Decoding](speculative-decoding.md) - The lossless draft-and-verify framework that consumes tree candidates.
- [KV Cache](kv-cache.md) - Cached target-model state reused while scoring speculative branches.
