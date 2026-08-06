---
title: "Matrix Tiling"
summary: "Blocking a GEMM (or any kernel) into tiles that fit on-chip SRAM and registers so operands are loaded from global memory few times and reused many times."
tooltip: "Matrix tiling (blocking) splits a large GEMM into rectangular tiles small enough for on-chip storage. Each tile of A, B, and C is loaded once from global memory and reused across the inner loops, converting a memory-bound kernel into a compute-bound one. Tile sizes and a triple-buffered load–compute–store pipeline decide whether the hardware stays busy."
layout: default
confidence: high
category: algorithms
sources:
  - raw/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.html
  - raw/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.metadata.json
  - derived/web-markdown/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.md
aliases:
  - blocking
  - tile-based computation
  - register blocking
appears_in:
  - docs/algorithms/flashattention/flashattention-4.md
  - docs/algorithms/flashattention/flashattention.md
  - docs/frameworks/triton-ascend/cannbot-skills-workflow.md
  - docs/frameworks/triton/index.md
  - docs/hardware/spatial-gemm.md
updated: 2026-08-06
---

# Matrix Tiling

**Matrix Tiling** (blocking) decomposes a large matrix computation such as GEMM into rectangular tiles that fit on-chip storage — SRAM, shared memory, or registers — so each operand is fetched from global memory a few times and reused many times.

## Why It Exists

Without tiling, the inner loops of a GEMM re-read operands from slow global memory for every multiply, and the kernel is throttled by memory bandwidth instead of compute. The whole point of the memory hierarchy is reuse: if a tile of $A$ is loaded once into SRAM and multiplied against many tiles of $B$, the cost of loading it is amortized across thousands of multiply-accumulates. Every serious GEMM — cuBLAS, Triton, FlashAttention, Spatial's blocked GEMM — is a tiling strategy.

## How It Works

The output is partitioned into $M/bm \times N/bn$ tiles of C, and the shared dimension into $P/bp$ chunks. For each output tile:

1. Prefetch the C tile into SRAM.
2. For each $k$-chunk, load a $bm \times bp$ tile of A and a $bp \times bn$ tile of B in parallel, then run the tile-level multiply-accumulate.
3. Store the accumulated C tile back to global memory.

The Spatial tutorial expresses this with `Foreach(M by bm, N by bn)` over output tiles, a `MemFold` over $P$ chunks that accumulates on top of the prefetched C tile, and a `MemReduce` inside that accumulates the per-$k$ partial tiles. A triple-buffered memory overlaps the load, compute, and store stages of the outermost pipeline so the compute unit never waits on memory. Hierarchical tiling (as in Triton) layers this: shared-memory-sized micro-tiles and register-sized nano-tiles.

## Tradeoffs

- Larger tiles increase arithmetic intensity but must fit on-chip; exceeding SRAM/register capacity causes spills and serialization.
- Tile shape interacts with layout (row-major vs. column-major) and with [memory banking](memory-banking.md): the banking scheme must allow every parallel access into a tile to hit a distinct bank.
- The loop order (which tiling dimension is outermost) changes DRAM traffic; Spatial scans output tiles horizontally to minimize transactions.

## Common Confusions

- **Tiling vs. blocking:** The same idea; "blocking" is the classic high-performance-computing term, "tiling" the accelerator/GPU term.
- **Tiling vs. memory banking:** Tiling decides *which data* is on-chip and how loops are ordered; banking decides *how parallel accesses* to that on-chip data avoid collisions.
- **Tiling vs. kernel fusion:** Fusion avoids writing intermediate tiles to global memory between kernels; tiling keeps individual kernels' operands on-chip.

## Where It Appears

- [Spatial: General Matrix Multiply tutorial](https://spatial-lang.org/gemm/) — The blocked GEMM tutorial is the canonical example: output tiles, $k$-chunks, prefetch, and triple buffering.
- [Spatial GEMM: Blocked Outer-Product Matrix Multiply](../hardware/spatial-gemm.md) — Output tiling and the triple-buffered pipeline are the tutorial's core technique.
- [FlashAttention](../algorithms/flashattention/flashattention.md) — IO-aware tiling is the core trick that makes exact attention memory-bandwidth-optimal.
- [Triton](../frameworks/triton/index.md) — Elevates tiles to first-class language citizens; hierarchical tiling is the compiler's core optimization.
- [CANNBot Skills Workflow](../frameworks/triton-ascend/cannbot-skills-workflow.md) — Tiling is one of the 25 optimization points and a required pre-analysis step before profiling.
- [FlashAttention-4: Blackwell Attention Kernel Co-Design](../algorithms/flashattention/flashattention-4.md) — FlashAttention-4 algorithm and kernel-pipeline techniques for faster exact attention on NVIDIA Blackwell GPUs.

## Related Terms

- [General Matrix Multiply (GEMM)](gemm.md) — The kernel that tiling targets.
- [Memory Banking](memory-banking.md) — Conflict-free on-chip access to tiled data.
- [Inner Product](inner-product.md) / [Outer Product](outer-product.md) — The accumulation strategies used inside each tile.
- [Systolic Array](systolic-array.md) — A spatial alternative that tiles the *dataflow* instead of the loops.
