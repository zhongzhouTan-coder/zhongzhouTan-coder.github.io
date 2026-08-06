---
title: "Memory Banking"
summary: "Partitioning on-chip SRAM into banks so parallel accesses to different addresses hit different banks in the same cycle, avoiding bank conflicts."
tooltip: "Memory banking partitions an on-chip SRAM into several independently addressable banks. When a parallel loop reads or writes several elements of a tile in one cycle, each access must land in a different bank or the memory serializes them. Compilers analyze the access patterns and pick a banking scheme — flat, or hierarchical when flat banking is too costly — that makes the parallel accesses conflict-free."
layout: default
confidence: high
category: hardware
sources:
  - raw/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.html
  - raw/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.metadata.json
  - derived/web-markdown/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.md
aliases:
  - multi-dimensional banking
  - SRAM banking
  - bank assignment
appears_in:
  - docs/hardware/index.md
  - docs/hardware/spatial-gemm.md
updated: 2026-08-06
---

# Memory Banking

**Memory Banking** partitions an on-chip SRAM into multiple independent banks so that parallel accesses to different addresses hit different banks in the same cycle, eliminating bank conflicts.

## Why It Exists

An SRAM has a limited number of read/write ports. When a compiler parallelizes a loop (for example, the innermost GEMM loop reading many elements of a tile at once), the hardware tries to issue several accesses to the same SRAM in one cycle. A single-ported memory can serve only one, so without banking the parallelization collapses back to serialized access. Banking gives the memory structure: several banks, each able to serve one access per cycle, so a parallel access pattern is conflict-free exactly when no two simultaneous accesses target the same bank.

## How It Works

The compiler inspects all accesses to a memory and searches for a bank assignment such that no two accesses in any cycle collide. The Spatial tutorial demonstrates this on a GEMM `tileB`: with `mp = 2` and `ip = 4` parallelization, one possible scheme banks the 2-D SRAM into a 1-D layout; when the writer is additionally parallelized by 8, that scheme breaks and the compiler falls back to a different (flat) scheme. This is based on generalized memory partitioning for high-level synthesis (Wang et al.), with the search ordered to try a flat N-dimensional-to-1-dimensional bank first and only attempt hierarchical (multi-level) banking when flat banking fails or is too resource-expensive.

## Tradeoffs

- Banking granularity wastes capacity: aligning addresses to banks can force padding and unused slots.
- A scheme that is conflict-free for one access pattern may collide for another (e.g., changing the writer's parallelization), so the compiler re-banks on each pattern.
- Hierarchical banking covers more patterns but costs extra multiplexing logic and area.

## Common Confusions

- **Banking vs. cache lines:** Caches group contiguous data into lines for spatial locality; banks are independent parallel-accessible arrays chosen by the compiler for conflict freedom.
- **Banking vs. matrix tiling:** Tiling decides which data lives on-chip and how loops are ordered; banking decides how simultaneous accesses to that on-chip data are served in parallel.
- **Bank conflicts vs. false sharing:** False sharing is a cache coherence cost on CPUs; a bank conflict is a cycle-by-cycle serialization in a compiler-banked SRAM.

## Where It Appears

- [Spatial: General Matrix Multiply tutorial](https://spatial-lang.org/gemm/) — The "Multi-dimensional Banking" section banks GEMM `tileB` under different parallelization factors and shows flat vs. hierarchical schemes.
- [Spatial GEMM: Blocked Outer-Product Matrix Multiply](../hardware/spatial-gemm.md) — The tutorial's banking demonstration is synthesized there.
- [Hardware and Numerics](../hardware/index.md) — Hardware and numerics pages covering accelerator features, precision formats, and related implementation details.

## Related Terms

- [Matrix Tiling](matrix-tiling.md) — The on-chip data layout that banking parallelizes access to.
- [Global Memory](global-memory.md) — The off-chip memory that tiles are loaded from into banked SRAM.
- [Systolic Array](systolic-array.md) — Feeds operands out of banked SRAM into the processing grid.
