---
title: "Frameworks"
summary: "Framework pages covering LLM serving systems and structured language-model programming runtimes."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-06
---

# Frameworks

## vLLM

- [vLLM](vllm/index.md) — Category hub for vLLM serving framework pages.
- [vLLM Architecture and Code Organization Overview](vllm/vllm-overview.md) — Start here: the six-layer mental model, the `vllm/` and `vllm/v1/` directory maps, component-by-component responsibilities, the request lifecycle across processes, and the main extension points.
- [vLLM: PagedAttention Serving Framework](vllm/vllm-framework.md) — LLM serving framework design, [PagedAttention](../terms/pagedattention.md) KV-cache paging, [block tables](../terms/block-table.md), copy-on-write sharing, scheduling/preemption, distributed execution, and throughput results.
- [vLLM Code Learning Path and Request Flow](vllm/vllm-code-learning-path.md) — Current vLLM codebase map, request lifecycle from OpenAI API entrypoint to worker execution, and an achievement-driven path to build a mini vLLM.
- [vLLM Continuous Batching: Scheduler, KV Blocks, and Runtime Flow](vllm/vllm-continuous-batching/index.md) — Current V1 iteration loop ([continuous batching](../terms/continuous-batching.md)), token and sequence budgets, running/waiting admission, [chunked prefill](../terms/chunked-prefill.md), paged KV-slot allocation, persistent worker batches, completion, and preemption.
- [vLLM Kimi K3 Code Reading Map](vllm/vllm-kimi-k3-code-reading.md) — Upstream vLLM Kimi K3 implementation map covering XTML request handling, multimodal wrapper, KimiLinear text model, hybrid [KDA](../terms/kimi-delta-attention.md)/MLA attention, latent [MoE](../terms/mixture-of-experts.md), DeepGEMM MegaMoE, MTP, and K3-specific kernels.

## vLLM Ascend

- [vLLM Ascend](vllm-ascend/index.md) — Category hub for vLLM's Ascend NPU port.
- [vLLM-Ascend Kimi K3 MoE Forward Insight](vllm-ascend/kimi-k3-moe-forward.md) — Latest-code insight for the Kimi K3-style routed-MoE forward substrate in vllm-ascend: patched FusedMoE construction, typed MoE stage contracts, Ascend routing, token dispatch, grouped MLP compute, routed-expert capture, Fused MC2, and dynamic EPLB.
- [DeepSeek-V4 Lightning Indexer C8 Quantization](vllm-ascend/deepseek-v4-lightning-indexer-c8.md) — How the DeepSeek-V4 [Lightning Indexer](../terms/lightning-indexer.md) runs on an 8-bit key cache and query in vllm-ascend: INT8 + FP16 scales on 910B/A2/A3, FP8 e4m3 + FP32 scales on A5, the quantized top-k custom operators, and the C4-vs-C8 naming.
- [DeepSeek-V4 Inference on Ascend: The DSA Serving Stack](vllm-ascend/deepseek-v4-inference.md) — End-to-end DeepSeek-V4 serving on Ascend NPUs: the model override with mHC [hyper-connections](../terms/hyper-connections.md), hybrid c4/c128 compressor layers, the AscendDSA prefill/decode flow, the five-type heterogeneous [KV cache](../terms/kv-cache.md), the sparse-attention custom operator, and the MTP draft model.
- [Qwen3.5 / Qwen3.6 Inference Path on vLLM Ascend](vllm-ascend/qwen3.5-qwen3.6-inference.md) — How the shared `qwen3_5`-family models (Qwen3.5-27B / Qwen3.6-27B dense hybrid Mamba-Transformer, multimodal; Qwen3.5/Qwen3.6-35B-A3B and Qwen3.5-397B-A17B sparse MoE) run on Ascend by reusing the `qwen3_5` / `qwen3_5_moe` model types: GDN [linear attention](../terms/linear-attention.md) plus FIA full attention, ModelSlim W8A8 quantization, `qwen3_5_mtp` speculative decoding, and ACL-graph capture at pinned revision `9a52ca5fc36c`.
- [vLLM-Ascend Prefill and Decode Scheduling: Qwen3.5 GQA](vllm-ascend/prefill-decode-scheduling-qwen3.5.md) — Token-budget scheduling, chunked prefills, mixed FIA GQA layouts, and the parallel GDN recurrent path.

## Triton

- [Triton: Tiled GPU Kernel Language and Compiler](triton/index.md) — Original Triton language (MAPL 2019): Triton-C tile-programming frontend, Triton-IR tile-level LLVM extensions, Triton-JIT compiler with hierarchical tiling, memory coalescing, shared memory allocation/synchronization, and auto-tuning.
- [Triton in Practice: How vLLM and vllm-ascend Use Triton](triton/triton-in-vllm.md) — Codebase-driven tour: vLLM's triton_utils infrastructure, custom op registration, ~163 kernel files across 12 categories, universal coding patterns, and vllm-ascend's CANN-backend adaptation.

## Triton Ascend

- [Triton Ascend: Ascend NPU Backend for Triton](triton-ascend/index.md) — Beginner-friendly architecture tour: five-layer design, TTIR→HIVM→LLVM→Linalg→Bisheng compilation flow, SIMD/SIMT/Unstructured-in-SIMT three-mode compilation, and relationship to vllm-ascend.
- [Triton Ascend 算子机制学习路径](triton-ascend/operator-mechanisms.md) — 图解 AIC/AIV、UB/L1/L0、MTE 和异步指令队列，并用代码追踪、容量估算、profiling 症状和分阶段练习讲解 Vector、Cube 与 CV Fusion。

## SGLang

- [SGLang: Structured Language Model Programs](sglang/index.md) — Framework architecture, Python-embedded programming model, RadixAttention KV cache reuse, compressed FSM decoding, API speculative execution, and performance results.

## DeepSeek

- [DeepSeek](deepseek/index.md) — Category hub for DeepSeek model implementation readings.
- [DeepSeek V4 Attention: Code Reading Map](deepseek/v4-attention-code-reading.md) — Navigable implementation map of DeepSeek V4's hybrid compressed attention across vLLM (NVIDIA/AMD/XPU) and vllm-ascend (Ascend NPU).

## Other Frameworks

- [Sarathi: Chunked Prefills for Efficient LLM Inference](sarathi/index.md) — Chunked-prefill and decode-maximal batching that reuses prefill weight loads for decode tokens and reduces pipeline bubbles.
- [DSpark: Confidence-Scheduled Speculative Decoding](dspark/index.md) — DeepSeek speculative decoding framework that combines semi-autoregressive draft generation, calibrated confidence estimates, hardware-aware verification scheduling, and production DeepSeek-V4 deployment results.
- [Harbor: Agent Evaluation Framework (Code Reading)](harbor/index.md) — Repository-backed tour of Harbor's task packaging model, `Job -> JobPlan -> TrialQueue -> Trial -> Agent/Environment/Verifier` runtime, local/git/package/registry distribution, multi-step trials, and compile/exec workflows at pinned revision `97e65926410b`.
