---
title: "Kimi"
summary: "Kimi model family: the Kimi Linear attention architecture and the Kimi K3 frontier model."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-03
---

# Kimi

- [Kimi Linear: Expressive Efficient Attention Architecture](kimi-linear/index.md) — Hybrid linear attention: KDA with channel-wise gating extends Gated DeltaNet, 3:1 KDA-to-MLA layer ratio with NoPE, 48B MoE (3B active); for the first time outperforms full attention across short/long/RL regimes, 6.3× decoding speedup at 1M context.
- [Kimi K3: Open 3T-Class Frontier Model](kimi-k3/index.md) — 2.8T/104B-active native multimodal MoE with hybrid KDA/MLA attention, Stable LatentMoE, 1M context, multi-effort agentic RL, MoonEP balanced expert-parallel training, and long-rollout cache/sandbox infrastructure.
