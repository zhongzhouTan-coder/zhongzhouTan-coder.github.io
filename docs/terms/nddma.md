---
title: "NDDMA"
summary: "A multidimensional direct-memory-access engine that copies tensor data while applying layout transformations."
tooltip: "NDDMA (N-dimensional Direct Memory Access) moves data across multidimensional layouts and can combine the copy with reorder or transpose work. On Ascend 950 the white paper describes up to five dimensions, hardware address generation, and larger reads when locality permits."
layout: default
confidence: medium
category: hardware
sources:
  - raw/hardware/ascend-950-npu-architecture-white-paper--paper.pdf
appears_in:
  - docs/hardware/ascend/ascend-950/index.md
updated: 2026-08-31
---

# NDDMA

**NDDMA** is an N-dimensional Direct Memory Access engine that copies tensor data while applying multidimensional layout transformations in hardware.

## Why It Exists

Tensor layouts such as NCHW and NHWC, padding, and strided sub-blocks turn a simple copy into many address calculations and small memory operations. If software performs that work element by element, data movement can dominate the kernel.

## How It Works

The Ascend 950 NDDMA engine receives dimension, stride, and placement parameters, reads a structured region from global memory, and writes the reordered result into Vector Unified Buffer. The white paper reports support for up to five dimensions and an internal cache that can combine fine-grained accesses into 128-byte reads when the pattern has locality.

## Tradeoffs

NDDMA reduces instruction and address-generation work, but the configured shape still has to match alignment, layout, and buffer-capacity constraints. Irregular or low-locality transfers can remain expensive.

## Common Confusions

- **NDDMA vs. ordinary DMA:** Ordinary DMA moves bytes between addresses; NDDMA also describes multidimensional tensor layout and reorder behavior.
- **NDDMA vs. a transpose kernel:** NDDMA is a movement primitive that can perform selected layout transformations; it is not a general arbitrary tensor algorithm.

## Where It Appears

- [Ascend 950 NPU Architecture: Compute, Memory, and Unified Bus](../hardware/ascend/ascend-950/index.md) — Explains NDDMA as the data-movement path for layout-aware AI kernels.

## Related Terms

- [Global Memory](global-memory.md) — The source or destination of many NDDMA transfers.
- [Memory Banking](memory-banking.md) — The on-chip access structure that receives or serves moved tiles.
