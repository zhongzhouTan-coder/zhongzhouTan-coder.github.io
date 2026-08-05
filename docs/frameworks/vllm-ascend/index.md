---
title: "vLLM Ascend"
summary: "vLLM's Ascend NPU port: code-reading notes and MoE forward implementation insights."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-03
---

# vLLM Ascend

- [vLLM-Ascend Architecture: How the Ascend NPU Port Integrates with vLLM](architecture.md) — Mapped onto vLLM's six-layer stack (reuse layers 1-4, replace layers 5-6), then the five integration mechanisms (plugin registration, NPUPlatform, ModelRegistry, [monkey-patches](../../terms/monkey-patching.md), custom backends), worker variants (NPUWorker/NPUWorker310/XliteWorker), execution flow from startup to per-step inference, ACL graph capture, HCCL communication, and what upstream vLLM code is reused as-is.
- [vLLM-Ascend Kimi K3 MoE Forward Insight](kimi-k3-moe-forward.md) — Latest-code insight for the Kimi K3-style routed-MoE forward substrate in vllm-ascend: patched FusedMoE construction, typed MoE stage contracts, Ascend routing, token dispatch, grouped MLP compute, routed-expert capture, Fused MC2, and dynamic EPLB.
- [DeepSeek-V4 Lightning Indexer C8 Quantization](deepseek-v4-lightning-indexer-c8.md) — How the DeepSeek-V4 Lightning Indexer runs on an 8-bit key cache and query in vllm-ascend: INT8 + FP16 scales on 910B/A2/A3, FP8 e4m3 + FP32 scales on A5, the quantized top-k custom operators, and the C4-vs-C8 naming.

Related: [vLLM](../vllm/index.md) — the upstream NVIDIA/AMD/XPU serving framework.
