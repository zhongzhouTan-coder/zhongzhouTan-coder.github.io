---
title: "Attention Variants"
summary: "Attention designs that reduce query/key redundancy or KV-cache pressure: multi-query, grouped-query, collaborative, and latent attention."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-03
---

# Attention Variants

- [Multi-Query Attention: One Write-Head is All You Need](multi-query-attention.md) — Shares one K/V across all attention heads, shrinking incremental decoder memory bandwidth 8× for a 12× inference speedup with negligible quality loss.
- [Grouped-Query Attention in Llama 2](grouped-query-attention/index.md) — Llama 2's 34B/70B GQA decision: 8 KV groups, 30B MHA/MQA/GQA ablation, higher large-batch throughput, and simpler 8-GPU tensor-parallel serving than MQA.
- [Collaborative Multi-Head Attention](collaborative-attention.md) — Redesigns MHA with shared key/query projections and per-head mixing vectors, enabling 4× compression of Q/K dimensions; CP tensor decomposition for post-hoc conversion of pretrained models.
- [DeepSeek-V2 Multi-Head Latent Attention](deepseek-v2-mla.md) — DeepSeek-V2's MLA design: low-rank joint K/V latent cache, decoupled RoPE, query compression, [MoE](../../terms/mixture-of-experts.md) serving context, 93.3% KV-cache reduction, and 5.76× maximum generation throughput versus DeepSeek 67B.
