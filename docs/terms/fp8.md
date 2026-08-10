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
appears_in:
  - docs/algorithms/context-parallelism/index.md
updated: 2026-08-10
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

## Related Terms

- [Microscaling](microscaling.md) - A block-level scaling family for low-precision formats.
- [NVFP4](../hardware/quantization/nvfp4.md) - A four-bit floating-point deployment format.
