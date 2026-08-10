---
title: "Sequence Parallelism"
summary: "A distributed training strategy that splits the input sequence along the length dimension across GPUs, enabling longer sequences than any single device can hold."
tooltip: "Sequence parallelism distributes chunks of an input sequence across GPUs, and uses ring-style communication to compute attention across chunks. It is orthogonal to data, pipeline, and tensor parallelism, forming a fourth parallelism dimension. Unlike tensor parallelism (limited by attention head count), sequence parallelism scales with sequence length, which is typically much larger."
layout: default
confidence: high
category: training
sources:
  - raw/training/sequence-parallelism-long-sequence-training--arxiv-2105.13120.pdf
  - raw/algorithms/context-parallelism-scalable-million-token-inference--arxiv-2411.01783v3.pdf
aliases:
  - SP
  - seq-parallel
appears_in:
  - docs/algorithms/flashattention/flashattention-2.md
  - docs/algorithms/linear-attention/index.md
  - docs/frameworks/vllm-ascend/architecture.md
  - docs/frameworks/vllm-ascend/deepseek-v4-inference.md
  - docs/frameworks/vllm-ascend/kimi-k3-moe-forward.md
  - docs/frameworks/vllm/vllm-kimi-k3-code-reading.md
  - docs/hardware/quantization/nvfp4.md
  - docs/training/foundation-models/llama.md
  - docs/training/index.md
  - docs/training/parallelism/index.md
  - docs/training/parallelism/megatron-lm/index.md
  - docs/training/parallelism/sequence-parallelism/index.md
  - docs/algorithms/context-parallelism/index.md
updated: 2026-08-10
---

# Sequence Parallelism

**Sequence parallelism** is a distributed training strategy that splits the input sequence along the length dimension across multiple GPUs, so each device holds only a sub-sequence chunk. Attention computation across chunks is handled via ring-style peer-to-peer communication (Ring Self-Attention), avoiding the need to materialize the full $L \times L$ attention matrix on any single device.

## Why It Exists

Existing parallelism strategies — data, pipeline, and tensor — were designed for model size, not sequence length. Tensor parallelism splits attention heads ($Z$), but $Z$ (e.g., 12) is far smaller than sequence length $L$ (e.g., 2048+), capping scalability. Pipeline parallelism splits layers but each device still stores the full activation tensor for its stage. When the sequence itself is too long for one GPU, neither approach helps. Sequence parallelism fills this gap by splitting along $L$, the largest dimension.

## How It Works

The input sequence of length $L$ is split into $N$ chunks of length $L/N$, one per GPU. All GPUs hold identical model parameters. To compute self-attention, Ring Self-Attention (RSA) circulates key and value embeddings around a logical ring in two passes:

1. **Score pass:** Each GPU circulates its $K$ chunk $N-1$ times, computing partial $QK^T$ with each received $K$. After the ring completes, each GPU has the full attention score matrix for its chunk.
2. **Output pass:** Same ring pattern with $V$ embeddings. Each GPU accumulates partial weighted sums $S_i^n V_i$ to produce the final attention output $O^n$ for its chunk.

MLP layers that follow operate on each chunk independently with **no communication**, unlike tensor parallelism which requires [all-reduce](all-reduce.md) in MLP blocks.

Memory per device scales as $O(L/N)$ instead of $O(L)$, and with sparse attention (already $O(L)$), sequence parallelism can theoretically handle arbitrarily long sequences.

## Where It Appears

- [Sequence Parallelism: Splitting Sequences Across GPUs](../training/parallelism/sequence-parallelism/index.md) — Li et al. (NUS, 2021) introduce the concept with Ring Self-Attention, achieving 13.7× larger batch size and 3.0× longer sequences than tensor parallelism on 64 GPUs.
- [Megatron-LM: GPU-Cluster Training Parallelism](../training/parallelism/megatron-lm/index.md) — Megatron-LM later adopted sequence parallelism as part of its TP-SP combination within transformer layers, splitting along the sequence dimension inside tensor-parallel regions to reduce activation memory.
- [vLLM-Ascend Architecture](../frameworks/vllm-ascend/architecture.md) — Identifies sequence-parallel communication at Ascend model-runner boundaries.
- [NVFP4](../hardware/quantization/nvfp4.md) — Uses sequence-parallel collectives in distributed low-precision training.
- [FlashAttention-2: Better Parallelism and Work Partitioning](../algorithms/flashattention/flashattention-2.md) — FlashAttention-2 algorithm: reduced non-matmul overhead, sequence-parallel attention blocks, warp-level work partitioning, and.
- [Transformers Are RNNs: Linear Attention](../algorithms/linear-attention/index.md) — Linear attention replaces softmax similarity with a feature-map kernel, reassociates matrix products, and turns causal attention.
- [vLLM Kimi K3 Code Reading Map](../frameworks/vllm/vllm-kimi-k3-code-reading.md) — Code-reading map for upstream vLLM's real Kimi K3 implementation: request parsing, multimodal wrapper, KimiLinear text model.
- [DeepSeek-V4 Inference on Ascend: The DSA Serving Stack in vllm-ascend](../frameworks/vllm-ascend/deepseek-v4-inference.md) — How vllm-ascend runs DeepSeek-V4 end to end on Ascend NPUs: model override with mHC hyper-connections, hybrid c4/c128 compressor.
- [vLLM-Ascend Kimi K3 MoE Forward Insight](../frameworks/vllm-ascend/kimi-k3-moe-forward.md) — Fresh code-reading insight for how the latest vllm-ascend routed-MoE substrate would execute a Kimi K3-style forward pass.
- [LLaMA: Open and Efficient Foundation Language Models](../training/foundation-models/llama.md) — Introduces the original LLaMA model family, showing that smaller decoder-only Transformers trained longer on public data can.
- [Training](../training/index.md) — Training and fine-tuning pages covering optimization behavior, transfer learning, and generalization in large models.
- [Training Parallelism](../training/parallelism/index.md) — Data, tensor, pipeline, and sequence parallelism techniques for large-model training.
- [Context Parallelism for Scalable Million-Token Inference](../algorithms/context-parallelism/index.md) — Extends the sequence-sharding and ring-communication idea to exact multi-turn inference and distributed KV state.

## Related Terms

- [All-Gather](all-gather.md) — The split/all-gather pattern at pipeline boundaries that sequence parallelism avoids entirely.
- [Scatter/Gather](scatter-gather.md) — Pipeline-parallel communication optimization that complements sequence parallelism in 4D training.
- [Microbatch](microbatch.md) — Pipeline-parallel work unit; sequence parallelism is orthogonal and composes naturally with pipeline schedules.
