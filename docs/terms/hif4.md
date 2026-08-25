---
title: "HiFloat4 (HiF4)"
summary: "HiFloat4 (HiF4) is a 4.5-bit-per-value block floating-point format that packs 64 S1P2 payloads with a three-level shared scaling hierarchy."
tooltip: "HiFloat4 uses an E6M2 base scale plus shared one-bit micro-exponents to cover global and local range differences across 64 four-bit values. Its S1P2 payload keeps three bits of significand precision, and its 64-wide dot product can remain mostly integer arithmetic."
layout: default
confidence: medium
category: hardware
sources:
  - raw/hardware/hif4-format-for-language-model-inference--arxiv-2602.11287v1.pdf
aliases:
  - HiF4
mention_lint: canonical
appears_in:
  - docs/hardware/quantization/hif4/index.md
updated: 2026-08-25
---

# HiFloat4 (HiF4)

**HiFloat4 (HiF4)** is a 4.5-bit-per-value block floating-point format that stores 64 four-bit S1P2 values with 32 bits of shared E6M2, E1_8, and E1_16 scaling metadata.

## Why It Exists

Four-bit scalar payloads have too little precision and range for many LLM tensors. HiF4 amortizes exponent information across a 64-value unit so the payload can keep a three-bit significand while local one-bit flags follow smaller range changes.

## How It Works

One E6M2 scale supplies the shared base magnitude. Eight one-bit E1_8 flags cover groups of eight values, and sixteen one-bit E1_16 flags cover groups of four. The payload uses sign-magnitude S1P2, equivalent to E1M2, so local flags can be absorbed as shifts before the dot-product reduction.

## Tradeoffs

HiF4's benefits depend on support for its conversion path and 64-wide integer-heavy dot product. The source reports inference results and hardware estimates, but leaves training evaluation and a public native implementation open.

## Where It Appears

- [HiFloat4 (HiF4): 4-Bit Block Floating Point for LLM Inference](../hardware/quantization/hif4/index.md) — Format layout, BF16 conversion, dot-product hardware flow, and LLM inference comparisons.

## Related Terms

- [Block Floating Point](block-floating-point.md)
- [Microscaling](microscaling.md)
- [NVFP4](nvfp4.md)
