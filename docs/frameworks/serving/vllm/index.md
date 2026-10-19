---
title: "vLLM"
summary: "vLLM serving framework pages: architecture, PagedAttention, scheduling, prefill/decode disaggregation, context parallelism, and model code readings."
layout: default
redirect_from:
  - /frameworks/vllm/
confidence: high
sources:
  - logs/index.md
  - raw/frameworks/serving/vllm/vllm-codebase--github-02aa10539f15.md
  - derived/repo-analysis/frameworks/serving/vllm/vllm/02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb/important-files.md
  - derived/repo-analysis/frameworks/serving/vllm/vllm/02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb/deepseek-v4.1-gpu-inference-path.md
updated: 2026-10-19
---

# vLLM

The pinned vLLM source record and code reading notes at commit `02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb` retain the October 2026 checkpoint analysis evidence.

- [MiMo-V2.6-Pro-MOPD: vLLM Inference and Deployment](mimo-v2-6-pro-mopd-deployment.md) — Version and memory requirements, H200 TP8 and MI355X TP4 launch profiles, tool parsers, and optional DFlash decoding.
- [MiMo-V2.6-Flash-MOPD: vLLM Inference and Deployment](mimo-v2-6-flash-mopd-deployment.md) — Flash-specific TP4 launch, MXFP4 memory floor, H200 and AMD backend profiles, and optional DFlash configuration.
- [vLLM Architecture and Code Organization Overview](vllm-overview.md) — Start here: the six-layer mental model, the `vllm/` and `vllm/v1/` directory maps, component-by-component responsibilities, the request lifecycle across processes, and the main extension points.
- [DeepSeek Integrations](deepseek/index.md) — DeepSeek V4 and V4.1 implementation readings across vLLM and vllm-ascend.
- [GLM-5.2 on vLLM: Request-to-GPU Backend Inference Path](glm-5.2-inference-path.md) — Complete upstream request round trip through OpenAI admission, V1 scheduling, the CUDA GPU runner, configuration-driven shared sparse indexers, Hopper/Blackwell sparse-MLA kernels, routed/shared MoE, sampling, detokenization, response emission, and cleanup.
- [DeepSeek-V4.1-Flash on NVIDIA vLLM: GPU Inference Path](deepseek/v4.1-gpu-inference-path/index.md) — Worker-boundary trace through model resolution, V4.1 cache/indexer ownership, sparse-MLA backend selection, mHC/MoE, and sampled-token output.
- [GLM-5.1 on vLLM: Expert-Parallel MoE Path](glm-5.1-expert-parallel/index.md) — Code-backed trace from the shared GLM-5.1 model shell through EP group sizing, global-to-local expert mapping, router dispatch, local expert MLPs, backend combine, and residual return.
- [vLLM MHA Code Path: From QKV to Paged KV Cache](vllm-mha-code-path.md) — Decoder self-attention from tensor-parallel QKV projection and RoPE through per-step metadata, slot-mapped KV writes, paged backend execution, and the output projection.
- [vLLM: PagedAttention Serving Framework](vllm-framework.md) — LLM serving framework design, [PagedAttention](../../../terms/pagedattention.md) KV-cache paging, [block tables](../../../terms/block-table.md), copy-on-write sharing, scheduling/preemption, distributed execution, and throughput results.
- [vLLM Continuous Batching: Scheduler, KV Blocks, and Runtime Flow](vllm-continuous-batching/index.md) — Current V1 iteration loop ([continuous batching](../../../terms/continuous-batching.md)), token and sequence budgets, running/waiting admission, [chunked prefill](../../../terms/chunked-prefill.md), paged KV-slot allocation, persistent worker batches, completion, and preemption.
- [vLLM Data-Parallel Deployment: Internal and External Load Balancing](data-parallel-deployment/index.md) — Deployment strategy for internal, hybrid, per-rank external, and multi-port external load balancing, including MoE synchronization and launch topologies.
- [vLLM Prefill/Decode Disaggregated Deployment Path](prefill-decode-disaggregated-deployment/index.md) — Deployment-oriented request trace across the router, prefill pool, NIXL KV-transfer plane, and decode pool, with pull/push modes, independent scaling, compatibility gates, and failure policy.
- [vLLM Block Table Management: From PagedAttention to the V1 KV Cache Stack](vllm-block-management/index.md) — Deep dive into the V1 block pool, per-group [KV cache](../../../terms/kv-cache.md) managers, hash-based prefix caching, refcount/copy-on-write sharing, block recycling, and the worker-side block table tensors consumed by PagedAttention kernels.
- [vLLM Mamba2 and Linear-Attention Prefix-Cache Path](mamba2-linear-attention-prefix-cache/index.md) — How repeated token prefixes become recurrent-state checkpoints, how hybrid cache groups agree on one resume boundary, and how Mamba2/GDN/linear-attention layers reuse the state through shared `MambaSpec` plumbing.
- [vLLM DCP and PCP: Decode and Prefill Context Parallelism](vllm-context-parallelism.md) — How PCP partitions current-step query rows and restores token order while DCP partitions persistent KV rows and restores exact softmax normalization.
- [vLLM DCP Attention: From Local LSE to Exact Global Output](dcp-attention/index.md) — Focused derivation of the DCP attention path, stable LSE correction kernel, exactness proof, and AG+RS versus AG+AR output distribution.
- [vLLM Kimi K3 Code Reading Map](vllm-kimi-k3-code-reading.md) — Upstream vLLM Kimi K3 implementation map covering XTML request handling, multimodal wrapper, KimiLinear text model, hybrid [KDA](../../../terms/kimi-delta-attention.md)/MLA attention, latent [MoE](../../../terms/mixture-of-experts.md), DeepGEMM MegaMoE, MTP, and K3-specific kernels.
- [MiniMax GQA W4A4 Quantization Path: GPU (vLLM) and NPU (vllm-ascend)](minimax-gqa-w4a4-quantization-path.md) — Beginner-oriented W4A4 mental model, layer placement, offline/load/runtime lifecycle, GPU and NPU implementation paths, hardware fallbacks, and pinned code evidence.

Related: [vLLM Ascend](../vllm-ascend/index.md) — the Ascend NPU port of vLLM.
