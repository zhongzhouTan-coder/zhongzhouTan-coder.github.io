---
title: "Chunked KV-Cache Compression"
summary: "Stores fewer key/value entries by summarizing consecutive token windows at a fixed stride, trading long-context cache cost for phase-dependent retention."
tooltip: "Chunked KV-cache compression replaces several neighboring tokens' key/value states with fewer summaries. It reduces long-context memory and attention work, but makes a token's representation depend on its position relative to the compression stride."
layout: default
confidence: high
category: algorithms
sources:
  - raw/algorithms/attention/phase-sensitivity-chunked-kv-cache-compression--arxiv-2609.36322v1.pdf
  - raw/training/model-families/deepseek/deepseek-v4--paper.pdf
  - raw/training/model-families/deepseek/deepseek-v4.1-technical-report--paper.pdf
aliases:
  - Chunked KV compression
  - Windowed KV-cache compression
mention_lint: canonical
appears_in:
  - docs/algorithms/attention/phase-sensitivity-chunked-kv-cache/index.md
  - docs/training/model-families/deepseek/deepseek-v4/index.md
  - docs/training/model-families/deepseek/deepseek-v4.1/index.md
updated: 2026-10-10
---

# Chunked KV-Cache Compression

**Chunked KV-Cache Compression** maps consecutive token windows into fewer cached key/value entries, reducing long-context state while making each token's representation depend on its position in the compression cycle.

## Why It Exists

An ordinary [KV cache](kv-cache.md) retains keys and values for every token, so its memory grows with context length. A compressor can combine multiple neighboring positions into a smaller set of entries before a future query is known, reducing cache size and the number of positions later attention must read.

## How It Works

A model groups source tokens into windows of width $W$ and emits compressed entries at stride $S$. Learned payload maps and gates, or a simpler aggregation such as uniform averaging, form the compressed keys and values. The source phase $\phi = t \bmod S$ identifies where a token at position $t$ falls relative to the stride; shifting the prefix can change its summary even when the token and its relationship to nearby content stay the same.

With non-overlapping windows, $W=S$. Overlapping designs can use $S<W$, so a token participates in multiple windows while its phase still follows the stride. Compression is distinct from sparse attention: compression changes how many key/value entries are stored, while sparsity selects which stored entries a query reads.

## Tradeoffs

Fewer entries can reduce cache and attention cost, but summaries are formed before later query intent is known and may not preserve every source detail equally. [Periodic Weak Spots](../algorithms/attention/phase-sensitivity-chunked-kv-cache/index.md) measures phase-dependent retrieval gaps in chunk-compressed models; average accuracy alone may conceal a weak residue.

## Common Confusions

- **Chunked KV-cache compression vs. cache quantization:** Compression combines source positions into fewer entries; quantization stores each value with fewer bits.
- **Chunked KV-cache compression vs. sparse attention:** Compression changes the stored representation; sparse attention selects a subset of stored positions for a query.
- **Window size vs. stride:** $W$ is how many tokens a summary covers, while $S$ is how often the compressor advances. They are equal only in non-overlapping designs.

## Where It Appears

- [Periodic Weak Spots: Phase Sensitivity from Chunked KV-Cache Compression](../algorithms/attention/phase-sensitivity-chunked-kv-cache/index.md) — Measures stride-periodic retrieval gaps and analyzes phase-specialized attention contributions.
- [DeepSeek-V4: Million-Token Context via Hybrid Compressed Attention](../training/model-families/deepseek/deepseek-v4/index.md) — Combines Compressed Sparse Attention and Heavily Compressed Attention with different compression rates.
- [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](../training/model-families/deepseek/deepseek-v4.1/index.md) — Builds on compressed global KV with cross-layer reuse and additional cache optimizations.

## Related Terms

- [KV Cache](kv-cache.md) — The key/value state that compression reduces.
- [Sliding-Window Attention](sliding-window-attention.md) — Restricts attention to recent uncompressed positions.
