---
title: "FP8"
summary: "An 8-bit floating-point family used to reduce model and activation memory traffic at a controlled numerical cost."
tooltip: "FP8 stores values in eight bits, commonly with E4M3 or E5M2 encodings and separate scaling. It can make large-model weights and communication cheaper, but accuracy and kernel behavior depend on the encoding and scale policy."
layout: default
confidence: high
category: hardware
sources:
  - raw/algorithms/context-parallelism-scalable-million-token-inference--arxiv-2411.01783v3.pdf
aliases:
  - 8-bit floating point
  - float8
mention_lint: off
appears_in:
  - docs/algorithms/context-parallelism/index.md
  - docs/algorithms/attention-variants/deepseek-v2-mla.md
  - docs/algorithms/deepseek-v3.2/index.md
  - docs/algorithms/flashattention/flashattention-2.md
  - docs/algorithms/flashattention/flashattention-3.md
  - docs/algorithms/flashattention/index.md
  - docs/algorithms/index.md
  - docs/frameworks/deepseek/v4-attention-code-reading.md
  - docs/frameworks/index.md
  - docs/frameworks/triton/triton-in-vllm.md
  - docs/frameworks/vllm-ascend/architecture.md
  - docs/frameworks/vllm-ascend/deepseek-v4-inference.md
  - docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md
  - docs/frameworks/vllm-ascend/index.md
  - docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md
  - docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md
  - docs/frameworks/vllm/vllm-block-management/index.md
  - docs/frameworks/vllm/vllm-kimi-k3-code-reading.md
  - docs/hardware/quantization/flatquant/index.md
  - docs/hardware/quantization/index.md
  - docs/hardware/quantization/microscaling-mx-formats/index.md
  - docs/hardware/quantization/nvfp4.md
  - docs/hardware/nvidia/ada-lovelace-professional-gpu-architecture/index.md
  - docs/training/deepseek/deepseek-v4/index.md
  - docs/training/efficient-attention/minimax-sparse-attention/index.md
updated: 2026-08-25
---

# FP8

**FP8** is an 8-bit floating-point representation used to reduce the storage and movement cost of neural-network tensors.

## Why It Exists

Large-model inference is constrained by both device memory and bandwidth. Replacing 16-bit values with scaled 8-bit values can make a model fit and can leave more bandwidth for attention communication.

## How It Works

An FP8 deployment chooses an encoding such as E4M3 or E5M2 and applies scale factors so the representable range and precision match the tensor. The context-parallel paper uses row-wise FP8 weights for Llama3 405B's feed-forward layers while evaluating the parallel attention system.

## Tradeoffs

FP8 is a numerical optimization, not a context-parallel algorithm. Scale selection, outliers, accumulation precision, and hardware support determine whether it preserves quality and delivers a speedup.

## Common Confusions

- **FP8 vs. INT8:** FP8 preserves a floating-point exponent and fraction; INT8 uses an integer range and a different scaling model.
- **FP8 vs. exact attention:** The paper's attention communication algorithm is exact relative to the evaluated FP8 model; FP8 does not make the attention sparse.

## Where It Appears

- [Context Parallelism for Scalable Million-Token Inference](../algorithms/context-parallelism/index.md) - Uses row-wise FP8 weights in the Llama3 405B benchmark configuration.

- [DeepSeek-V2 Multi-Head Latent Attention](../algorithms/attention-variants/deepseek-v2-mla.md)
- [DeepSeek-V3.2: Sparse Attention, Scaled RL, and Thinking in Tool-Use](../algorithms/deepseek-v3.2/index.md)
- [FlashAttention-2: Better Parallelism and Work Partitioning](../algorithms/flashattention/flashattention-2.md)
- [FlashAttention-3: Hopper Asynchrony and FP8 Attention](../algorithms/flashattention/flashattention-3.md)
- [FlashAttention](../algorithms/flashattention/index.md)
- [Algorithms](../algorithms/index.md)
- [DeepSeek V4 Attention: Code Reading Map](../frameworks/deepseek/v4-attention-code-reading.md)
- [Frameworks](../frameworks/index.md)
- [Triton in Practice: How vLLM and vllm-ascend Use Triton](../frameworks/triton/triton-in-vllm.md)
- [vLLM-Ascend Architecture: How the Ascend NPU Port Integrates with vLLM](../frameworks/vllm-ascend/architecture.md)
- [DeepSeek-V4 Inference on Ascend: The DSA Serving Stack in vllm-ascend](../frameworks/vllm-ascend/deepseek-v4-inference.md)
- [DeepSeek-V4 Lightning Indexer C8 Quantization: INT8/FP8 Indexer Cache in vllm-ascend](../frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md)
- [vLLM Ascend](../frameworks/vllm-ascend/index.md)
- [Qwen3.5 / Qwen3.6 Inference Path on vLLM Ascend](../frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md)
- [MiniMax GQA W4A4 Quantization Path: GPU (vLLM) and NPU (vllm-ascend)](../frameworks/vllm/minimax-gqa-w4a4-quantization-path.md)
- [vLLM Block Table Management: From PagedAttention to the V1 KV Cache Stack](../frameworks/vllm/vllm-block-management/index.md)
- [vLLM Kimi K3 Code Reading Map](../frameworks/vllm/vllm-kimi-k3-code-reading.md)
- [FlatQuant: Fast Learnable Affine Quantization](../hardware/quantization/flatquant/index.md)
- [Quantization](../hardware/quantization/index.md)
- [Microscaling (MX) Formats: Block Floating Point for AI Hardware](../hardware/quantization/microscaling-mx-formats/index.md)
- [NVFP4: Blackwell 4-Bit Floating Point](../hardware/quantization/nvfp4.md)
- [NVIDIA Ada Lovelace Professional GPU Architecture](../hardware/nvidia/ada-lovelace-professional-gpu-architecture/index.md) - Describes FP8 as a fourth-generation Tensor Core capability in a professional GPU.
- [DeepSeek-V4: Million-Token Context via Hybrid Compressed Attention](../training/deepseek/deepseek-v4/index.md)
- [MiniMax Sparse Attention (MSA)](../training/efficient-attention/minimax-sparse-attention/index.md)

## Related Terms

- [Microscaling](microscaling.md) - A block-level scaling family for low-precision formats.
- [NVFP4](nvfp4.md) - A four-bit floating-point deployment format.
