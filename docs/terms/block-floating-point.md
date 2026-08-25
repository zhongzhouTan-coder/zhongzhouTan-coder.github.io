---
title: "Block Floating Point"
summary: "Block floating point shares exponent or scale metadata across a group of narrow values so storage and arithmetic can be cheaper than scalar floating point."
tooltip: "Block floating point gives a group of values one shared scale and keeps smaller payloads for the individual elements. It can reduce metadata, alignment work, and memory traffic, but outliers in a block can reduce the precision available to the rest of the group."
layout: default
confidence: high
category: hardware
sources:
  - raw/hardware/hif4-format-for-language-model-inference--arxiv-2602.11287v1.pdf
  - raw/hardware/microscaling-mx-formats--ocp-v1.0.pdf
aliases:
  - block-floating-point
  - BFP
mention_lint: canonical
appears_in:
  - docs/hardware/quantization/hif4/index.md
  - docs/hardware/quantization/microscaling-mx-formats/index.md
updated: 2026-08-25
---

# Block Floating Point

**Block Floating Point** is a numeric representation that shares one exponent or scale across a group of narrow element payloads.

## Why It Exists

Scalar floating-point values repeat exponent metadata and force hardware to align each product separately. A shared block scale amortizes that metadata and lets a reduction tree use narrower, cheaper arithmetic.

## How It Works

A block stores one scale plus several private values. The scale handles the block's broad magnitude; the private payloads preserve signs and local precision. MX formats use a shared E8M0 scale for 32 private elements, while HiF4 uses an E6M2 base scale and two levels of shared one-bit micro-exponents for 64 S1P2 elements.

## Tradeoffs

The shared scale must cover the whole block. An outlier can therefore waste representable range for smaller values, and the hardware must still define block layout, conversion, accumulation precision, and the wider operations around the narrow dot product.

## Common Confusions

- **Block floating point vs. scalar FP4:** Block floating point describes the shared-scale organization, not simply a four-bit payload.
- **Block floating point vs. microscaling:** Microscaling is a fine-grained block-floating family; different microscaled formats can use different group sizes, scale encodings, and payloads.

## Where It Appears

- [HiFloat4 (HiF4): 4-Bit Block Floating Point for LLM Inference](../hardware/quantization/hif4/index.md) — Uses a three-level scale hierarchy to preserve precision and simplify 64-wide dot products.
- [Microscaling (MX) Formats: Block Floating Point for AI Hardware](../hardware/quantization/microscaling-mx-formats/index.md) — Defines the OCP shared-scale block contract and its hardware boundaries.

## Related Terms

- [Microscaling](microscaling.md)
- [HiFloat4 (HiF4)](hif4.md)
- [NVFP4](nvfp4.md)
