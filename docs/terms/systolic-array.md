---
title: "Systolic Array"
summary: "A regular grid of processing elements where data flows rhythmically between neighbors so each weight is reused across many multiply-accumulates without re-fetching."
tooltip: "A systolic array is a 1D or 2D grid of processing elements connected only to their neighbors. Data pulses through the grid (systole-like) while weights stay put or stream in, so every element is reused many times per fetch. This data-reuse pattern is why Google's TPU, the Ascend Cube unit, and many FPGA GEMM designs use systolic-style dataflow for dense matrix multiply."
layout: default
confidence: high
category: hardware
sources:
  - raw/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.html
  - raw/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.metadata.json
  - derived/web-markdown/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.md
aliases:
  - systolic dataflow
appears_in:
  - docs/hardware/spatial-gemm.md
updated: 2026-08-04
---

# Systolic Array

**Systolic Array** is a regular grid of processing elements connected only to their immediate neighbors, through which data flows rhythmically so that each operand is reused across many multiply-accumulates without being re-fetched.

## Why It Exists

In a conventional processor, every multiply needs its two operands delivered to the same ALU, so memory traffic grows with the number of operations. A systolic array inverts this: instead of moving data *to* the compute repeatedly, it moves data *through* the compute once, and each processing element multiplies that data against a locally held value many times. For a dense GEMM this converts a memory-bound problem into a compute-bound one with near-optimal data reuse.

## How It Works

In the classic weight-stationary variant, weights are loaded once into each processing element, one operand matrix streams in from the edges, and partial sums accumulate along the array as data propagates to neighbors. In the output-stationary variant the accumulation registers stay fixed while both operands flow through. Either way the array performs the same multiply-accumulate count as a scalar GEMM but with each fetch amortized over many operations. The Spatial tutorial lists systolic arrays alongside blocking, inner products, and outer products as one of the "many ways to break up" a GEMM, and its later tutorials build systolic-array applications from shift registers.

## Tradeoffs

- Excellent dense-GEMM reuse, but the regular dataflow requires **padding** when matrix dimensions are not multiples of the array size.
- The rigid grid is inefficient for sparse or irregular computation; sparsity-aware hardware needs extra machinery (skipping zeros, compressed rows).
- Array size is fixed at design time, so utilization drops for small matrices; kernels must tile to fill the array.

## Common Confusions

- **Systolic array vs. tensor core / Cube:** Tensor cores and the Ascend Cube are often built with systolic-style internal dataflow but are programmed as one instruction over a whole tile; a systolic array proper is a general dataflow architecture.
- **Systolic array vs. shift register:** Shift registers shift a single data stream; a systolic array is a grid of *processing elements* that compute, not just store, while data moves.
- **Systolic array vs. SIMD:** SIMD applies one instruction to a vector in lockstep; systolic arrays are asynchronous pipelines between neighbors with no single instruction stream.

## Where It Appears

- [Spatial: General Matrix Multiply tutorial](https://spatial-lang.org/gemm/) — Names systolic arrays as one of the GEMM decomposition techniques; later Spatial tutorials implement them from shift registers.
- [Spatial GEMM: Blocked Outer-Product Matrix Multiply](../hardware/spatial-gemm.md) — Lists systolic arrays among the GEMM decomposition options.
- [General Matrix Multiply (GEMM)](gemm.md) — The kernel a systolic array accelerates.
- [Matrix Tiling](matrix-tiling.md) — Tiling feeds the array at its full width; the two strategies compose.

## Related Terms

- [General Matrix Multiply (GEMM)](gemm.md) — The dense product a systolic array computes.
- [Matrix Tiling](matrix-tiling.md) — Supplies the dataflow the array consumes.
- [Memory Banking](memory-banking.md) — Feeds the array's operands from on-chip SRAM without conflicts.
- [Global Memory](global-memory.md) — The off-chip source of the operand streams.
