---
title: "Cross-Layer Index Reuse"
summary: "A sparse-attention optimization in which selected token or block indices computed at an anchor layer are reused by subsequent layers."
tooltip: "Cross-layer index reuse avoids repeating a sparse selector at every Transformer layer. An anchor layer computes top-k token or block indices, and nearby layers reuse them until another anchor refreshes the selection."
layout: default
confidence: high
category: algorithms
sources:
  - raw/training/efficient-attention/indexcache-accelerating-sparse-attention-cross-layer-index-reuse--arxiv-2603.12201v1.pdf
  - raw/training/model-families/deepseek/deepseek-v4.1-technical-report--paper.pdf
aliases:
  - index reuse
  - cross-layer attention index reuse
mention_aliases:
  - cross-layer attention index reuse
mention_lint: canonical
appears_in:
  - docs/training/efficient-attention/indexcache/index.md
  - docs/training/model-families/deepseek/deepseek-v4.1/index.md
updated: 2026-09-16
---

# Cross-Layer Index Reuse

**Cross-Layer Index Reuse** is a sparse-attention optimization in which selected token or block indices computed at an anchor layer are reused by subsequent layers.

## Why It Exists

Dynamic sparse attention saves core-attention work by selecting a small part of a long context, but the selector may still scan the full prefix at every layer. Neighboring layers often choose highly overlapping positions, so repeating selection wastes compute that grows with both sequence length and model depth.

## How It Works

An anchor or Full layer scores visible positions, selects a top-k index set, and stores that small routing result. Following Shared or Reuse layers skip selection and run their own sparse attention over the stored positions. A later anchor refreshes the set before layer needs diverge too far.

The central design problem is anchor placement. IndexCache shows that aggregate overlap alone is insufficient for a frozen model: a few differing but important tokens can affect every downstream layer. It instead uses end-to-end calibration loss, or trains each retained selector against the averaged attention distributions of its consumer layers.

## Tradeoffs

Reuse removes selector compute but constrains several layers to the same candidate positions. Long reuse spans, poorly placed anchors, or out-of-distribution traffic can omit layer-specific evidence. Training-aware reuse can reduce that mismatch but requires model updates; training-free reuse requires a representative calibration set and nontrivial pattern search.

## Common Confusions

- **Index reuse vs. KV-cache sharing:** index reuse shares *which positions to read*; KV sharing reuses the stored key/value representations themselves. A design such as CSA2 can do both, while IndexCache focuses on indices.
- **Index reuse vs. fixed sparse attention:** reused indices remain query-dependent because an anchor computes them dynamically. A fixed window or block mask does not inspect content.
- **Indexer cache vs. KV cache:** the cached index tensor is small routing metadata and can be overwritten at each anchor; it is not the token-by-token key/value state used by attention.

## Where It Appears

- [IndexCache](../training/efficient-attention/indexcache/index.md) — Systematically selects retained indexer layers with loss-guided search or trains them with multi-layer distillation.
- [DeepSeek-V4.1-Flash](../training/model-families/deepseek/deepseek-v4.1/index.md) — CSA2 Reuse layers share top-k selections while Full/Reindex layers refresh them, alongside global KV sharing.
- TidalDecode, Kascade, and HySparse — Related methods described by IndexCache that reuse selections from full-attention anchor layers.

## Related Terms

- [Lightning Indexer](lightning-indexer.md) — A learned content-dependent selector whose output can be reused.
- [KV Cache](kv-cache.md) — Persistent attention state that may be shared independently of indices.
- [Sliding-Window Attention](sliding-window-attention.md) — A content-independent local pattern that does not require top-k index refreshes.
