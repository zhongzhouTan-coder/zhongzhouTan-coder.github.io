---
title: "Frameworks"
summary: "Framework pages covering LLM serving systems and structured language-model programming runtimes."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-04
---

# Frameworks

## vLLM

- [vLLM](vllm/index.md) — Category hub for vLLM serving framework pages.
- [vLLM: PagedAttention Serving Framework](vllm/vllm-framework.md) — LLM serving framework design, PagedAttention KV-cache paging, block tables, copy-on-write sharing, scheduling/preemption, distributed execution, and throughput results.
- [vLLM Code Learning Path and Request Flow](vllm/vllm-code-learning-path.md) — Current vLLM codebase map, request lifecycle from OpenAI API entrypoint to worker execution, and an achievement-driven path to build a mini vLLM.
- [vLLM Continuous Batching: Scheduler, KV Blocks, and Runtime Flow](vllm/vllm-continuous-batching/index.md) — Current V1 iteration loop, token and sequence budgets, running/waiting admission, chunked prefill, paged KV-slot allocation, persistent worker batches, completion, and preemption.
- [vLLM Kimi K3 Code Reading Map](vllm/vllm-kimi-k3-code-reading.md) — Upstream vLLM Kimi K3 implementation map covering XTML request handling, multimodal wrapper, KimiLinear text model, hybrid KDA/MLA attention, latent MoE, DeepGEMM MegaMoE, MTP, and K3-specific kernels.

## vLLM Ascend

- [vLLM Ascend](vllm-ascend/index.md) — Category hub for vLLM's Ascend NPU port.
- [vLLM-Ascend Kimi K3 MoE Forward Insight](vllm-ascend/kimi-k3-moe-forward.md) — Latest-code insight for the Kimi K3-style routed-MoE forward substrate in vllm-ascend: patched FusedMoE construction, typed MoE stage contracts, Ascend routing, token dispatch, grouped MLP compute, routed-expert capture, Fused MC2, and dynamic EPLB.

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

- [DSpark: Confidence-Scheduled Speculative Decoding](dspark/index.md) — DeepSeek speculative decoding framework that combines semi-autoregressive draft generation, calibrated confidence estimates, hardware-aware verification scheduling, and production DeepSeek-V4 deployment results.
- [Harbor: Agent Evaluation Framework (Code Reading)](harbor/index.md) — Repository-backed tour of Harbor's task packaging model, `Job -> JobPlan -> TrialQueue -> Trial -> Agent/Environment/Verifier` runtime, local/git/package/registry distribution, multi-step trials, and compile/exec workflows at pinned revision `97e65926410b`.
