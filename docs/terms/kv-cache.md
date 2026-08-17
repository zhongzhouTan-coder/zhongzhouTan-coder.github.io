---
title: "KV Cache"
summary: "The stored key and value tensors from earlier tokens that autoregressive attention reuses instead of recomputing the whole prefix."
tooltip: "A KV cache stores each layer's past attention keys and values so decoding reuses the prefix. It removes repeated projection work, but its memory and per-token attention read grow with context length."
layout: default
confidence: high
category: algorithms
sources:
  - raw/algorithms/attention-is-all-you-need--arxiv-1706.03762.pdf
  - raw/algorithms/transformers-are-rnns-linear-attention--arxiv-2006.16236v3.pdf
  - raw/training/kimi-linear-expressive-efficient-attention--paper.pdf
  - raw/training/k3-technical-report--paper.pdf
  - raw/frameworks/vllm-pagedattention-serving-framework--arxiv-2309.06180v1.pdf
  - raw/algorithms/context-parallelism-scalable-million-token-inference--arxiv-2411.01783v3.pdf
aliases:
  - key-value cache
  - key/value cache
mention_lint: off
appears_in:
  - docs/algorithms/attention-variants/deepseek-v2-mla.md
  - docs/algorithms/attention-variants/grouped-query-attention/index.md
  - docs/algorithms/attention-variants/multi-query-attention.md
  - docs/algorithms/foundations/recurrent-neural-networks/index.md
  - docs/algorithms/linear-attention/index.md
  - docs/algorithms/linear-attention/linear-attention-without-softmax.md
  - docs/benchmarks/serving-perf/aisbench-vs-evalscope-perf.md
  - docs/frameworks/deepseek/index.md
  - docs/frameworks/deepseek/v4-attention-code-reading.md
  - docs/frameworks/index.md
  - docs/frameworks/sarathi/index.md
  - docs/frameworks/sglang/index.md
  - docs/frameworks/triton-ascend/operator-mechanisms.md
  - docs/frameworks/triton/triton-in-vllm.md
  - docs/frameworks/vllm-ascend/architecture.md
  - docs/frameworks/vllm-ascend/deepseek-v4-inference.md
  - docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md
  - docs/frameworks/vllm-ascend/index.md
  - docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md
  - docs/frameworks/vllm/index.md
  - docs/frameworks/vllm/vllm-block-management/index.md
  - docs/frameworks/vllm/vllm-continuous-batching/index.md
  - docs/frameworks/vllm/vllm-framework.md
  - docs/frameworks/vllm/vllm-kimi-k3-code-reading.md
  - docs/frameworks/vllm/vllm-overview.md
  - docs/hardware/quantization/flatquant/index.md
  - docs/training/deepseek/deepseek-v4/index.md
  - docs/training/deepseek/index.md
  - docs/training/index.md
  - docs/training/kimi/kimi-k3/index.md
  - docs/training/kimi/kimi-linear/index.md
  - docs/training/parallelism/megatron-lm/index.md
  - docs/algorithms/context-parallelism/index.md
  - docs/frameworks/vllm/vllm-context-parallelism.md
  - docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md
  - docs/frameworks/vllm/dcp-attention/index.md
updated: 2026-08-17
---

# KV Cache

**KV Cache** is the per-layer store of attention keys and values produced by earlier tokens, reused during autoregressive decoding so the model does not recompute the entire prefix.

## Why It Exists

Without caching, generating token $i$ repeats key and value projections for tokens $1$ through $i-1$. Caching makes those projections persistent and reduces the repeated work, which is essential for practical Transformer decoding.

## How It Works

Each new token appends one key vector and one value vector per relevant layer and KV head. Its query attends over the cached keys and combines the corresponding values. Projection work per step becomes constant, but cache memory and the attention read still grow linearly with context length.

## Tradeoffs

Long contexts can make the cache the dominant memory consumer and constrain batch size. MQA, GQA, MLA, quantization, paging, sparse attention, and linear attention reduce different parts of this cost; only recurrent-state approaches make stored state independent of sequence length.

## Common Confusions

- **KV cache vs. model weights:** Weights are fixed learned parameters; the KV cache is request-specific runtime state.
- **KV cache vs. linear-attention state:** A KV cache preserves token-level keys and values; a linear-attention state merges them into a fixed-size summary.

## Where It Appears

- [Transformers Are RNNs: Linear Attention](../algorithms/linear-attention/index.md) — Contrasts explicit growing key/value history with a fixed-size recurrent summary.
- [Kimi Linear](../training/kimi/kimi-linear/index.md) — Uses recurrent KDA states in most layers and full KV caches in periodic MLA layers.
- [Kimi K3](../training/kimi/kimi-k3/index.md) — Adds external KV-cache retention for million-token partial rollouts and aligns KDA state lifecycles with MLA cache blocks.
- [vLLM Continuous Batching](../frameworks/vllm/vllm-continuous-batching/index.md) — Shows how paged KV capacity gates per-iteration admission, completion, and preemption.
- [vLLM Block Table Management](../frameworks/vllm/vllm-block-management/index.md) — Details how paged KV blocks, reference counts, and prefix caching implement the cache's paged storage.
- [FlatQuant](../hardware/quantization/flatquant/index.md) — Applies per-head learnable affine transforms to keys and values before low-bit KV-cache quantization (down to 2 bits).
- [vLLM-Ascend Architecture](../frameworks/vllm-ascend/architecture.md) — Shows how Ascend attention backends allocate and consume vLLM-compatible KV caches.
- [DeepSeek-V2 Multi-Head Latent Attention](../algorithms/attention-variants/deepseek-v2-mla.md) — Explains DeepSeek-V2's Multi-head Latent Attention: low-rank joint key/value compression, decoupled RoPE, MoE-scale inference.
- [Grouped-Query Attention in Llama 2](../algorithms/attention-variants/grouped-query-attention/index.md) — Explains why Llama 2 uses grouped-query attention for its 34B and 70B models: it cuts KV-cache pressure like multi-query.
- [Multi-Query Attention: One Write-Head is All You Need](../algorithms/attention-variants/multi-query-attention.md) — Replaces per-head key/value projections with a single shared K/V pair across all attention heads, eliminating the heads.
- [Recurrent Neural Networks: From RNN to LSTM](../algorithms/foundations/recurrent-neural-networks/index.md) — A beginner-oriented explanation of recurrent hidden state, shared weights across sequence steps, long-term dependency failures.
- [线性Attention的探索：Attention必须有个Softmax吗？](../algorithms/linear-attention/linear-attention-without-softmax.md) — Su Jianlin's influential blog survey on why softmax is the bottleneck of standard attention, how removing it enables O(n) linear.
- [AISBench Benchmark vs. EvalScope Perf](../benchmarks/serving-perf/aisbench-vs-evalscope-perf.md) — Competitive analysis of AISBench Benchmark and EvalScope Perf across load generation, workloads, metrics, extensibility.
- [DeepSeek](../frameworks/deepseek/index.md) — DeepSeek model implementation readings that span vLLM and vllm-ascend codebases.
- [DeepSeek V4 Attention: Code Reading Map](../frameworks/deepseek/v4-attention-code-reading.md) — A navigable map of the DeepSeek V4 hybrid compressed attention implementation across vLLM (NVIDIA/AMD/XPU) and vllm-ascend.
- [Frameworks](../frameworks/index.md) — Framework pages covering LLM serving systems and structured language-model programming runtimes.
- [Sarathi: Chunked Prefills for Efficient LLM Inference](../frameworks/sarathi/index.md) — Sarathi improves LLM serving by splitting prefills into compute-sized chunks and piggybacking decode tokens on them to raise.
- [SGLang: Structured Language Model Programs](../frameworks/sglang/index.md) — SGLang framework architecture, programming model, runtime optimizations, and evaluation results for efficient structured LLM.
- [Triton in Practice: How vLLM and vllm-ascend Use Triton](../frameworks/triton/triton-in-vllm.md) — A codebase-driven tour of Triton kernel development in vLLM (NVIDIA GPU) and vllm-ascend (Ascend NPU), covering infrastructure.
- [Triton Ascend Operator Mechanisms: Vector, Cube, and CV Fusion](../frameworks/triton-ascend/operator-mechanisms.md) — A practical learning path for Triton Ascend Vector, Cube, and CV fusion operators, grounded in AI Core compute units, on-chip.
- [vLLM](../frameworks/vllm/index.md) — vLLM serving framework pages: PagedAttention paper, continuous batching, and Kimi K3 code.
- [vLLM: PagedAttention Serving Framework](../frameworks/vllm/vllm-framework.md) — vLLM framework design, PagedAttention memory management, scheduling, decoding support, and serving performance results.
- [vLLM Kimi K3 Code Reading Map](../frameworks/vllm/vllm-kimi-k3-code-reading.md) — Code-reading map for upstream vLLM's real Kimi K3 implementation: request parsing, multimodal wrapper, KimiLinear text model.
- [vLLM Architecture and Code Organization Overview](../frameworks/vllm/vllm-overview.md) — A top-down code-reading map of the vLLM repository at commit a0c092ee72c0: how the V1 serving engine, model executor, config.
- [DeepSeek-V4 Inference on Ascend: The DSA Serving Stack in vllm-ascend](../frameworks/vllm-ascend/deepseek-v4-inference.md) — How vllm-ascend runs DeepSeek-V4 end to end on Ascend NPUs: model override with mHC hyper-connections, hybrid c4/c128 compressor.
- [DeepSeek-V4 Lightning Indexer C8 Quantization: INT8/FP8 Indexer Cache in vllm-ascend](../frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md) — How vllm-ascend quantizes the DeepSeek-V4 Lightning Indexer to 8 bits (C8): INT8 keys with FP16 scales on 910B/A2/A3, FP8 e4m3fn.
- [vLLM Ascend](../frameworks/vllm-ascend/index.md) — vLLM's Ascend NPU port: code-reading notes and MoE forward implementation insights.
- [Qwen3.5 / Qwen3.6 Inference Path on vLLM Ascend](../frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md) — A code-reading tour of the shared qwen3_5-family inference path: Qwen3.5-27B / Qwen3.6-27B (dense hybrid Mamba-Transformer.
- [DeepSeek-V4: Million-Token Context via Hybrid Compressed Attention](../training/deepseek/deepseek-v4/index.md) — DeepSeek-V4 introduces hybrid Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA), Manifold-Constrained.
- [DeepSeek](../training/deepseek/index.md) — DeepSeek model training papers: V4 hybrid compressed attention and V3.2 sparse attention with scaled RL.
- [Training](../training/index.md) — Training and fine-tuning pages covering optimization behavior, transfer learning, and generalization in large models.
- [Megatron-LM: GPU-Cluster Training Parallelism](../training/parallelism/megatron-lm/index.md) — Explains both Megatron-LM papers: intra-layer tensor model parallelism with f/g conjugate operators (8.3B, V100), and the PTD-P.
- [Context Parallelism for Scalable Million-Token Inference](../algorithms/context-parallelism/index.md) — Shards persistent KV state across context-parallel ranks and chooses whether to circulate KV or queries.
- [vLLM DCP and PCP: Decode and Prefill Context Parallelism](../frameworks/vllm/vllm-context-parallelism.md) — DCP shards persistent decode KV state; PCP gathers prefill cache inputs without duplicating decode writes.
- [vLLM Prefill/Decode Disaggregated Deployment Path](../frameworks/vllm/prefill-decode-disaggregated-deployment/index.md) — Transfers request-specific paged KV blocks from an independently scaled prefill pool into decode-owned storage.

- [vLLM DCP Attention: From Local LSE to Exact Global Output](../frameworks/vllm/dcp-attention/index.md)

## Related Terms

- [Linear Attention](linear-attention.md) — Replaces explicit token history with accumulated key–value statistics.
- [Kimi Delta Attention](kimi-delta-attention.md) — A Kimi-family fixed-state attention mechanism used to reduce cache pressure.
- [Continuous Batching](continuous-batching.md) — Uses iteration-level admission to keep available KV capacity productive.
- [PagedAttention](pagedattention.md) — The algorithm that stores KV cache in fixed-size non-contiguous blocks.
- [Block Table](block-table.md) — The logical-to-physical mapping used to address paged KV cache.
