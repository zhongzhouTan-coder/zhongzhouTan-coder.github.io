---
title: "DeepSeek"
summary: "DeepSeek model implementation readings that span vLLM and vllm-ascend codebases."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-03
---

# DeepSeek

- [DeepSeek V4 Attention: Code Reading Map](v4-attention-code-reading.md) — Navigable implementation map of DeepSeek V4's hybrid compressed attention across vLLM (NVIDIA/AMD/XPU) and vllm-ascend (Ascend NPU), covering CSA/HCA compressors, sparse MLA backends, heterogeneous [KV cache](../../terms/kv-cache.md), multi-stream overlap, and platform-specific kernel dispatch.
- [DeepSeek-V4 Inference on Ascend: The DSA Serving Stack](../vllm-ascend/deepseek-v4-inference.md) — End-to-end DeepSeek-V4 serving on Ascend NPUs: the model override with mHC [hyper-connections](../../terms/hyper-connections.md), hybrid c4/c128 compressor layers, the AscendDSA prefill/decode flow, the five-type heterogeneous KV cache, the sparse-attention custom operator, and the MTP draft model.

Related: [DSpark](../dspark/index.md) — DeepSeek's speculative decoding framework.
