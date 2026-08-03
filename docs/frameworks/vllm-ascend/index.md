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

- [vLLM-Ascend Kimi K3 MoE Forward Insight](kimi-k3-moe-forward.md) — Latest-code insight for the Kimi K3-style routed-MoE forward substrate in vllm-ascend: patched FusedMoE construction, typed MoE stage contracts, Ascend routing, token dispatch, grouped MLP compute, routed-expert capture, Fused MC2, and dynamic EPLB.

Related: [vLLM](../vllm/index.md) — the upstream NVIDIA/AMD/XPU serving framework.
