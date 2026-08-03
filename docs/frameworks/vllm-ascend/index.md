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

- [vLLM-Ascend Architecture: How the Ascend NPU Port Integrates with vLLM](architecture.md) — Code-reading tour of the five integration mechanisms (plugin registration, NPUPlatform, ModelRegistry, [monkey-patches](../../terms/monkey-patching.md), custom backends), execution flow from startup to per-step inference, ACL graph capture, HCCL communication, and what upstream vLLM code is reused as-is.
- [vLLM-Ascend Kimi K3 MoE Forward Insight](kimi-k3-moe-forward.md) — Latest-code insight for the Kimi K3-style routed-MoE forward substrate in vllm-ascend: patched FusedMoE construction, typed MoE stage contracts, Ascend routing, token dispatch, grouped MLP compute, routed-expert capture, Fused MC2, and dynamic EPLB.

Related: [vLLM](../vllm/index.md) — the upstream NVIDIA/AMD/XPU serving framework.
