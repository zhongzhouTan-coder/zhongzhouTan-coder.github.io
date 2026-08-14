---
title: "vLLM Ascend"
summary: "vLLM's Ascend NPU port: code-reading notes and MoE forward implementation insights."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-14
---

# vLLM Ascend

- [Qwen3.5 MTP: Drafting and Target-Model Verification](qwen3.5-mtp.md) — Concrete Qwen3.5 MTP path: target hidden state plus token embedding create draft proposals, target logits are aligned to each proposed position, and vLLM's rejection sampler accepts a prefix, recovers the first rejection, or appends a bonus token.
- [vLLM-Ascend Architecture: How the Ascend NPU Port Integrates with vLLM](architecture.md) — Mapped onto vLLM's six-layer stack (reuse layers 1-4, replace layers 5-6), then the five integration mechanisms (plugin registration, NPUPlatform, ModelRegistry, [monkey-patches](../../terms/monkey-patching.md), custom backends), worker variants (NPUWorker/NPUWorker310/XliteWorker), execution flow from startup to per-step inference, ACL graph capture, HCCL communication, and what upstream vLLM code is reused as-is.
- [vLLM-Ascend Kimi K3 MoE Forward Insight](kimi-k3-moe-forward.md) — Latest-code insight for the Kimi K3-style routed-[MoE](../../terms/mixture-of-experts.md) forward substrate in vllm-ascend: patched FusedMoE construction, typed MoE stage contracts, Ascend routing, token dispatch, grouped MLP compute, routed-expert capture, Fused MC2, and dynamic EPLB.
- [DeepSeek-V4 Lightning Indexer C8 Quantization](deepseek-v4-lightning-indexer-c8.md) — How the DeepSeek-V4 [Lightning Indexer](../../terms/lightning-indexer.md) runs on an 8-bit key cache and query in vllm-ascend: INT8 + FP16 scales on 910B/A2/A3, [FP8](../../terms/fp8.md) e4m3 + FP32 scales on A5, the quantized top-k custom operators, and the C4-vs-C8 naming.
- [DeepSeek-V4 Inference on Ascend: The DSA Serving Stack](deepseek-v4-inference.md) — End-to-end DeepSeek-V4 serving on Ascend NPUs: the model override with mHC [hyper-connections](../../terms/hyper-connections.md), hybrid c4/c128 compressor layers, the AscendDSA prefill/decode flow, the five-type heterogeneous [KV cache](../../terms/kv-cache.md), the sparse-attention custom operator, and the MTP draft model.
- [Qwen3.5 / Qwen3.6 Inference Path on vLLM Ascend](qwen3.5-qwen3.6-inference.md) — How the shared `qwen3_5`-family models (Qwen3.5-27B / Qwen3.6-27B dense hybrid Mamba-Transformer, multimodal; Qwen3.5/Qwen3.6-35B-A3B and Qwen3.5-397B-A17B sparse MoE) run on Ascend by reusing the `qwen3_5` / `qwen3_5_moe` model types: patched hybrid decoder forward, GDN [linear attention](../../terms/linear-attention.md) plus FIA full attention, ModelSlim W8A8 quantization, `qwen3_5_mtp` speculative decoding, and ACL-graph capture at pinned revision `9a52ca5fc36c`.
- [vLLM-Ascend Prefill and Decode Scheduling: Qwen3.5 GQA](prefill-decode-scheduling-qwen3.5.md) — Token-budget scheduling, chunked prefills, mixed FIA GQA layouts, and the parallel GDN recurrent path.

Related: [vLLM](../vllm/index.md) — the upstream NVIDIA/AMD/XPU serving framework.
