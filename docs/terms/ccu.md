---
title: "Collective Communication Unit (CCU)"
summary: "A hardware engine that combines remote data movement and reductions for device-side collective communication."
tooltip: "The Ascend 950 CCU executes programmed collective missions using Memory Slices, Reduce Units, and URMA movement. It targets patterns such as all-reduce, all-gather, reduce-scatter, and all-to-all so AI cores do not have to orchestrate every communication fragment."
layout: default
confidence: medium
category: hardware
sources:
  - raw/hardware/ascend-950-npu-architecture-white-paper--paper.pdf
appears_in:
  - docs/hardware/ascend/ascend-950/index.md
updated: 2026-08-31
---

# Collective Communication Unit (CCU)

**Collective Communication Unit (CCU)** is an Ascend hardware engine that executes programmed collective communication by combining remote movement with local reduction operations.

## Why It Exists

Large-model training and inference repeatedly exchange and reduce tensors across accelerator ranks. If AI cores or the host perform every copy, synchronization, and reduction step, communication consumes compute cycles and system-bus bandwidth.

## How It Works

Software submits a Mission through CCU Management. The instruction implementation dispatches remote movement to [URMA](urma.md) and reduction work to CCU Agents, which pair Memory Slices with Reduce Units. The Ascend 950 white paper lists Broadcast, Reduce Scatter, All Gather, All Reduce, All2All, and All2Allv as supported patterns.

## Tradeoffs

CCU still depends on a programmed algorithm, message sizes, route topology, and available ports. Hardware offload lowers orchestration cost, but it does not guarantee peak bandwidth for an imbalanced or contention-heavy collective.

## Common Confusions

- **CCU vs. All-Reduce:** All-reduce is a communication pattern; CCU is the hardware engine that can execute that and other patterns.
- **CCU vs. a DMA engine:** DMA moves data; CCU can coordinate movement with reductions and collective state.

## Where It Appears

- [Ascend 950 NPU Architecture: Compute, Memory, and Unified Bus](../hardware/ascend/ascend-950/index.md) — Explains CCU Missions, Memory Slices, Reduce Units, URMA movement, and supported collectives.

## Related Terms

- [Unified Bus (UB)](unified-bus.md) — The interconnect family carrying CCU traffic.
- [URMA](urma.md) — The asynchronous movement semantic used by CCU.
- [All-Reduce](all-reduce.md) — A reduction collective.
- [All-to-All](all-to-all.md) — A redistribution collective.
