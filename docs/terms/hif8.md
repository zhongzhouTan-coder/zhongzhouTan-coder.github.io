---
title: "HiF8"
summary: "Huawei's 8-bit floating-point format with variable-width exponent encoding and a reported 38-power-of-two combined exponent range."
tooltip: "HiF8 uses a variable-length prefix to indicate exponent width and denormal state rather than pairing every value with a separate microscale. The Ascend 950 white paper positions it as an 8-bit format with more range than FP8 E4M3 while keeping low storage and movement cost."
layout: default
confidence: medium
category: hardware
sources:
  - raw/hardware/ascend-950-npu-architecture-white-paper--paper.pdf
appears_in:
  - docs/hardware/ascend/ascend-950/index.md
updated: 2026-08-31
---

# HiF8

**HiF8** is Huawei's 8-bit floating-point format that uses a variable-width exponent prefix and denormal encoding to provide a wider dynamic range than FP8 E4M3 without an extra per-value microscale.

## Why It Exists

Eight-bit values reduce tensor storage and movement, but a fixed exponent/mantissa split can lose range or precision on neural-network distributions. HiF8 uses the limited bit budget to adapt exponent width across magnitudes.

## How It Works

The prefix field indicates how many exponent bits follow and whether the value is denormal. The white paper reports a combined exponent range of `[-22, 15]`, or 38 powers of two, and describes hidden leading bits plus special encodings for zero, NaN, and infinities.

## Tradeoffs

HiF8 is a vendor-specific format, so software, kernels, calibration, accumulation, and interoperability must all support its encoding. The white paper reports format properties and peak hardware support, not a complete accuracy study across models.

## Common Confusions

- **HiF8 vs. FP8 E4M3:** Both use eight bits, but HiF8 changes how exponent width is encoded instead of using E4M3's fixed split.
- **HiF8 vs. MXFP8:** HiF8 is a scalar encoding in the paper; MXFP8 uses a block-level shared scale as part of a microscaling representation.

## Where It Appears

- [Ascend 950 NPU Architecture: Compute, Memory, and Unified Bus](../hardware/ascend/ascend-950/index.md) — Explains HiF8's exponent encoding and its role in the third-generation Cube datapath.

## Related Terms

- [FP8](fp8.md) — The broader eight-bit floating-point family.
- [Microscaling](microscaling.md) — Shared-scale block formats such as MXFP8 and MXFP4.
