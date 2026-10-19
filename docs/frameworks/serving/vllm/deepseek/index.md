---
title: "DeepSeek"
summary: "DeepSeek model implementation readings that span vLLM and vllm-ascend codebases."
layout: default
redirect_from:
  - /frameworks/deepseek/
  - /frameworks/serving/deepseek/
confidence: high
sources:
  - logs/index.md
updated: 2026-10-19
---

# DeepSeek

- [DeepSeek-V4 on NVIDIA vLLM: GPU Inference Path and Attention Architecture](v4-attention-code-reading.md) — Pinned worker round trip through the V4 model runner, mHC residual streams, hybrid compressed attention (CSA/HCA/SWA), heterogeneous [KV cache](../../../../terms/kv-cache.md), MegaMoE, and token sampling.
- [DeepSeek-V4.1-Flash on NVIDIA vLLM: GPU Inference Path](v4.1-gpu-inference-path/index.md) — Pinned worker round trip through the V4.1 wrapper, configuration-driven cache/indexer ownership, NVIDIA sparse-MLA backends, bounded SWA replay, and sampled-token return.
- [DeepSeek-V4 Inference on Ascend: The DSA Serving Stack](../../vllm-ascend/deepseek-v4-inference.md) — End-to-end DeepSeek-V4 serving on Ascend NPUs: the model override with mHC [hyper-connections](../../../../terms/hyper-connections.md), hybrid c4/c128 compressor layers, the AscendDSA prefill/decode flow, the five-type heterogeneous KV cache, the sparse-attention custom operator, and the MTP draft model.

Related: [DSpark](../../../speculative-decoding/dspark/index.md) — DeepSeek's speculative decoding framework.
