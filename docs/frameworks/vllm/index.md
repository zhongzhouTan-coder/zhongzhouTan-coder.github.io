---
title: "vLLM"
summary: "vLLM serving framework pages: architecture, PagedAttention, scheduling, prefill/decode disaggregation, context parallelism, and model code readings."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-14
---

# vLLM

- [vLLM Architecture and Code Organization Overview](vllm-overview.md) — Start here: the six-layer mental model, the `vllm/` and `vllm/v1/` directory maps, component-by-component responsibilities, the request lifecycle across processes, and the main extension points.
- [vLLM: PagedAttention Serving Framework](vllm-framework.md) — LLM serving framework design, [PagedAttention](../../terms/pagedattention.md) KV-cache paging, [block tables](../../terms/block-table.md), copy-on-write sharing, scheduling/preemption, distributed execution, and throughput results.
- [vLLM Code Learning Path and Request Flow](vllm-code-learning-path.md) — Current vLLM codebase map, request lifecycle from OpenAI API entrypoint to worker execution, and an achievement-driven path to build a mini vLLM.
- [vLLM Continuous Batching: Scheduler, KV Blocks, and Runtime Flow](vllm-continuous-batching/index.md) — Current V1 iteration loop ([continuous batching](../../terms/continuous-batching.md)), token and sequence budgets, running/waiting admission, [chunked prefill](../../terms/chunked-prefill.md), paged KV-slot allocation, persistent worker batches, completion, and preemption.
- [vLLM Prefill/Decode Disaggregated Deployment Path](prefill-decode-disaggregated-deployment/index.md) — Deployment-oriented request trace across the router, prefill pool, NIXL KV-transfer plane, and decode pool, with pull/push modes, independent scaling, compatibility gates, and failure policy.
- [vLLM Block Table Management: From PagedAttention to the V1 KV Cache Stack](vllm-block-management/index.md) — Deep dive into the V1 block pool, per-group [KV cache](../../terms/kv-cache.md) managers, hash-based prefix caching, refcount/copy-on-write sharing, block recycling, and the worker-side block table tensors consumed by PagedAttention kernels.
- [vLLM DCP and PCP: Decode and Prefill Context Parallelism](vllm-context-parallelism.md) — Code-reading map of DCP KV ownership, PCP batch partitioning, exact LSE attention merging, cache block scaling, and support boundaries. <!-- termlint-ignore: context-parallelism -- Navigation label already links the dedicated context-parallelism insight. -->
- [vLLM DCP Attention: From Local LSE to Exact Global Output](dcp-attention/index.md) — Focused derivation of the DCP attention path, stable LSE correction kernel, exactness proof, and AG+RS versus AG+AR output distribution.
- [vLLM Kimi K3 Code Reading Map](vllm-kimi-k3-code-reading.md) — Upstream vLLM Kimi K3 implementation map covering XTML request handling, multimodal wrapper, KimiLinear text model, hybrid [KDA](../../terms/kimi-delta-attention.md)/MLA attention, latent [MoE](../../terms/mixture-of-experts.md), DeepGEMM MegaMoE, MTP, and K3-specific kernels.
- [MiniMax GQA W4A4 Quantization Path: GPU (vLLM) and NPU (vllm-ascend)](minimax-gqa-w4a4-quantization-path.md) — Beginner-oriented W4A4 mental model, layer placement, offline/load/runtime lifecycle, GPU and NPU implementation paths, hardware fallbacks, and pinned code evidence.

Related: [vLLM Ascend](../vllm-ascend/index.md) — the Ascend NPU port of vLLM.
