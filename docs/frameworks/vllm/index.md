---
title: "vLLM"
summary: "vLLM serving framework pages: PagedAttention paper, request-flow code learning path, continuous batching, and Kimi K3 code readings."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-08-03
---

# vLLM

- [vLLM: PagedAttention Serving Framework](vllm-framework.md) — LLM serving framework design, PagedAttention KV-cache paging, block tables, copy-on-write sharing, scheduling/preemption, distributed execution, and throughput results.
- [vLLM Code Learning Path and Request Flow](vllm-code-learning-path.md) — Current vLLM codebase map, request lifecycle from OpenAI API entrypoint to worker execution, and an achievement-driven path to build a mini vLLM.
- [vLLM Continuous Batching: Scheduler, KV Blocks, and Runtime Flow](vllm-continuous-batching/index.md) — Current V1 iteration loop, token and sequence budgets, running/waiting admission, chunked prefill, paged KV-slot allocation, persistent worker batches, completion, and preemption.
- [vLLM Kimi K3 Code Reading Map](vllm-kimi-k3-code-reading.md) — Upstream vLLM Kimi K3 implementation map covering XTML request handling, multimodal wrapper, KimiLinear text model, hybrid KDA/MLA attention, latent MoE, DeepGEMM MegaMoE, MTP, and K3-specific kernels.

Related: [vLLM Ascend](../vllm-ascend/index.md) — the Ascend NPU port of vLLM.
