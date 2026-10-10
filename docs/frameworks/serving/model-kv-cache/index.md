---
title: "Model Architecture and KV Cache Paths: vLLM GPU and Ascend"
summary: "Five checkpoint-specific architecture and serving code readings across vLLM GPU and Ascend."
layout: default
confidence: medium
sources:
  - logs/index.md
updated: 2026-10-10
---

# Model Architecture and KV Cache Paths: vLLM GPU and Ascend

These pages expand the [October 2026 presentation brief](../../../../presentation-notes/open-model-architecture-kv-cache-gpu-ascend-2026-10.md) against released checkpoint configs and pinned source revisions. Each page explains the model's layer recipe, attention and expert paths, multimodal or draft boundaries, cache ownership, and GPU/Ascend implementation. Code reading is static; no device measurements were made.

- [GLM-5.3](glm-5.3.md) — 78-layer sparse MLA/MoE architecture and shared index decisions; distinguished from GLM-5.3-Flash.
- [DeepSeek-V4.1-Flash](deepseek-v4.1-flash.md) — causal encoder/decoder, mHC and MoE, shared compressed global KV, index history, compressor state, and local SWA.
- [MiniMax-M3](minimax-m3.md) — dense-to-sparse layer transition, routed experts, full GQA history, and block-sparse index cache.
- [Qwen3.8-Flash-Next](qwen3.8-flash-next.md) — GDN/QSA hybrid layers, hyper-connections, PLE, and the pinned Ascend implementation gap.
- [Kimi-K3](kimi-k3.md) — KDA/MLA hybrid layers, AttnRes, routed experts, and coherent recurrent/token-page state.

The payload estimates are useful floors, not measured capacity. For deployment, inventory tensor shape, stride, dtype, scale metadata, allocator padding, tensor parallel ownership, and prefix-hit state for each cache class.
