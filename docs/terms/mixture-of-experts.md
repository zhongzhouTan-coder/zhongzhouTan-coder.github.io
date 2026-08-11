---
title: "Mixture of Experts"
summary: "A sparse model architecture that routes each token through a small subset of many expert feed-forward networks to increase total capacity without activating every parameter."
tooltip: "Mixture of Experts increases model width by adding many specialist FFN experts. A router chooses a few experts per token, so total parameters can be huge while active compute stays much smaller."
layout: default
confidence: high
category: training
sources:
  - raw/algorithms/deepseek-v2-multi-head-latent-attention--arxiv-2405.04434.pdf
  - raw/training/deepseek-v4--paper.pdf
  - raw/training/kimi-linear-expressive-efficient-attention--paper.pdf
  - raw/training/k3-technical-report--paper.pdf
aliases:
  - MoE
  - sparse expert model
appears_in:
  - docs/algorithms/attention-variants/deepseek-v2-mla.md
  - docs/algorithms/attention-variants/index.md
  - docs/algorithms/deepseek-v3.2/index.md
  - docs/algorithms/index.md
  - docs/frameworks/deepseek/v4-attention-code-reading.md
  - docs/frameworks/dspark/index.md
  - docs/frameworks/index.md
  - docs/frameworks/triton/triton-in-vllm.md
  - docs/frameworks/vllm-ascend/architecture.md
  - docs/frameworks/vllm-ascend/deepseek-v4-inference.md
  - docs/frameworks/vllm-ascend/index.md
  - docs/frameworks/vllm-ascend/kimi-k3-moe-forward.md
  - docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md
  - docs/frameworks/vllm/index.md
  - docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md
  - docs/frameworks/vllm/vllm-kimi-k3-code-reading.md
  - docs/hardware/quantization/flatquant/index.md
  - docs/hardware/quantization/nvfp4.md
  - docs/training/deepseek/deepseek-v4/index.md
  - docs/training/deepseek/index.md
  - docs/training/efficient-attention/minimax-sparse-attention/index.md
  - docs/training/index.md
  - docs/training/kimi/index.md
  - docs/training/kimi/kimi-k3/index.md
  - docs/training/kimi/kimi-linear/index.md
  - docs/training/mhc/index.md
  - docs/frameworks/vllm/vllm-context-parallelism.md
updated: 2026-08-10
---

# Mixture of Experts

**Mixture of Experts** is a sparse neural-network architecture where a router sends each token to a small subset of many expert feed-forward networks, increasing total parameter capacity while keeping per-token active compute limited.

## Why It Exists

Dense scaling activates every parameter for every token, so larger models cost proportionally more at training and inference. MoE separates total capacity from active compute: many experts store knowledge and specialization, but only a few are evaluated per token.

## How It Works

A router scores experts for each token and selects a Top-k subset. The selected experts process the token representation, their outputs are weighted by router scores, and the aggregate returns to the main model stream. Large MoE systems usually combine routed experts with shared experts, expert parallelism, load-balancing rules, and communication overlap.

## Tradeoffs

MoE introduces routing instability, expert load imbalance, dispatch/combination communication, and train–inference consistency issues. Extreme expert counts need explicit load-balancing mechanisms such as routing biases, quantile balancing, redundant experts, or expert-parallel planning.

## Common Confusions

- **Total parameters vs. active parameters:** Total parameters count all experts; active parameters count only experts selected for one token.
- **MoE vs. attention:** MoE usually expands feed-forward/channel mixing capacity, while attention controls token mixing.

## Where It Appears

- [DeepSeek-V2 Multi-Head Latent Attention](../algorithms/attention-variants/deepseek-v2-mla.md) — Uses DeepSeekMoE alongside MLA to reduce active FFN compute.
- [DeepSeek-V4](../training/deepseek/deepseek-v4/index.md) — Uses a 1.6T/284B-active MoE model with hybrid compressed attention.
- [Kimi Linear](../training/kimi/kimi-linear/index.md) — Evaluates KDA/MLA hybrid attention on a 48B MoE with 3B active parameters.
- [Kimi K3](../training/kimi/kimi-k3/index.md) — Scales to 2.8T total parameters, 104B active, and 896 routed experts per layer.
- [MiniMax GQA W4A4 Quantization Path](../frameworks/vllm/minimax-gqa-w4a4-quantization-path.md) — Traces quantized sparse-expert execution on GPU and Ascend NPU.
- [vLLM Kimi K3 Code Reading](../frameworks/vllm/vllm-kimi-k3-code-reading.md) — Maps Kimi K3's routed experts into the vLLM runtime.
- [vLLM-Ascend Architecture](../frameworks/vllm-ascend/architecture.md) — Shows where Ascend-specific MoE kernels replace upstream execution.
- [vLLM-Ascend Kimi K3 MoE Forward](../frameworks/vllm-ascend/kimi-k3-moe-forward.md) — Follows the full routed-expert forward path on Ascend.
- [Attention Variants](../algorithms/attention-variants/index.md) — Attention designs that reduce query/key redundancy or KV-cache pressure: multi-query, grouped-query, collaborative, and latent.
- [DeepSeek-V3.2: Sparse Attention, Scaled RL, and Thinking in Tool-Use](../algorithms/deepseek-v3.2/index.md) — DeepSeek-V3.2 introduces DeepSeek Sparse Attention (DSA) for sub-quadratic long-context efficiency, a scaled GRPO recipe with.
- [Algorithms](../algorithms/index.md) — Algorithm pages covering inference algorithms, attention kernels, and scheduling methods.
- [DeepSeek V4 Attention: Code Reading Map](../frameworks/deepseek/v4-attention-code-reading.md) — A navigable map of the DeepSeek V4 hybrid compressed attention implementation across vLLM (NVIDIA/AMD/XPU) and vllm-ascend.
- [DSpark: Confidence-Scheduled Speculative Decoding](../frameworks/dspark/index.md) — DeepSeek's DSpark speculative decoding framework, combining semi-autoregressive draft generation with hardware-aware confidence.
- [Frameworks](../frameworks/index.md) — Framework pages covering LLM serving systems and structured language-model programming runtimes.
- [Triton in Practice: How vLLM and vllm-ascend Use Triton](../frameworks/triton/triton-in-vllm.md) — A codebase-driven tour of Triton kernel development in vLLM (NVIDIA GPU) and vllm-ascend (Ascend NPU), covering infrastructure.
- [vLLM](../frameworks/vllm/index.md) — vLLM serving framework pages: PagedAttention paper, request-flow code learning path, continuous batching, and Kimi K3 code.
- [DeepSeek-V4 Inference on Ascend: The DSA Serving Stack in vllm-ascend](../frameworks/vllm-ascend/deepseek-v4-inference.md) — How vllm-ascend runs DeepSeek-V4 end to end on Ascend NPUs: model override with mHC hyper-connections, hybrid c4/c128 compressor.
- [vLLM Ascend](../frameworks/vllm-ascend/index.md) — vLLM's Ascend NPU port: code-reading notes and MoE forward implementation insights.
- [Qwen3.5 / Qwen3.6 Inference Path on vLLM Ascend](../frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md) — A code-reading tour of the shared qwen3_5-family inference path: Qwen3.5-27B / Qwen3.6-27B (dense hybrid Mamba-Transformer.
- [FlatQuant: Fast Learnable Affine Quantization](../hardware/quantization/flatquant/index.md) — FlatQuant flattens outlier-heavy weights and activations with per-layer learnable affine transformations (Kronecker-factorized.
- [NVFP4: Blackwell 4-Bit Floating Point](../hardware/quantization/nvfp4.md) — NVIDIA NVFP4 format with hierarchical FP8/FP32 scaling, micro-block quantization, Random Hadamard Transform, and Transformer.
- [DeepSeek](../training/deepseek/index.md) — DeepSeek model training papers: V4 hybrid compressed attention and V3.2 sparse attention with scaled RL.
- [MiniMax Sparse Attention (MSA)](../training/efficient-attention/minimax-sparse-attention/index.md) — Blockwise sparse attention co-designed with GQA: a lightweight Index Branch selects top-k KV blocks per group, Main Branch.
- [Training](../training/index.md) — Training and fine-tuning pages covering optimization behavior, transfer learning, and generalization in large models.
- [Kimi](../training/kimi/index.md) — Kimi model family: the Kimi Linear attention architecture and the Kimi K3 frontier model.
- [mHC: Manifold-Constrained Hyper-Connections](../training/mhc/index.md) — DeepSeek's mHC projects Hyper-Connections' residual mixing matrix onto the doubly stochastic manifold with Sinkhorn-Knopp.
- [vLLM DCP and PCP: Decode and Prefill Context Parallelism](../frameworks/vllm/vllm-context-parallelism.md) — PCP size participates in MoE sharding and process-world sizing.

## Related Terms

- [All-Gather](all-gather.md) — Common distributed primitive in parallel training systems.
- [All-Reduce](all-reduce.md) — Common gradient-synchronization primitive.
