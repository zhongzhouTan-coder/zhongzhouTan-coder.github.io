---
title: "STARS Scheduler"
summary: "Ascend's device-side task and resource scheduler for coordinating compute, movement, synchronization, and accelerator engines."
tooltip: "STARS (System Task and Resource Scheduler) keeps task streams and resource decisions on the Ascend device. It coordinates AI cores, CPUs, media engines, DMA, Unified Bus queues, and collective engines, reducing host submission overhead but still depending on software-supplied dependencies and partitions."
layout: default
confidence: medium
category: hardware
sources:
  - raw/hardware/ascend-950-npu-architecture-white-paper--paper.pdf
appears_in:
  - docs/hardware/ascend/ascend-950/index.md
updated: 2026-08-31
---

# STARS Scheduler

**STARS Scheduler** is Ascend's device-side System Task and Resource Scheduler for dispatching work across compute, movement, synchronization, and accelerator resources.

## Why It Exists

Issuing every small task from the host adds control latency and makes overlap between compute, data movement, media processing, and communication fragile. A device-side scheduler can see the dependency graph closer to the resources it controls.

## How It Works

The Ascend 950 white paper describes STARS2.0 accepting up to 2,048 host-sunk task streams, prefetching tasks, scheduling AIC/AIV, AI CPU, DVPP, SDMA, UB Jetty, and CCU work, and reporting completion. A dedicated high-speed control bus connects it to AI cores; software can also create groups and resource pools for die affinity and isolation.

## Tradeoffs

STARS reduces submission overhead but does not invent parallelism: unresolved dependencies, poor task granularity, or an overly restrictive resource partition can still leave engines idle. The paper's nanosecond-scale scheduling claim is a control-path claim, not an end-to-end kernel latency guarantee.

## Common Confusions

- **STARS vs. a CPU scheduler:** STARS schedules accelerator tasks and device resources; it is not a general operating-system scheduler.
- **STARS vs. a compiler:** The compiler produces executable tasks and dependencies; STARS dispatches them at runtime.

## Where It Appears

- [Ascend 950 NPU Architecture: Compute, Memory, and Unified Bus](../hardware/ascend/ascend-950/index.md) — Explains device-side scheduling, task streams, groups, and resource slicing.

## Related Terms

- [NDDMA](nddma.md) — A movement engine whose work can be scheduled by STARS.
- [Collective Communication Unit (CCU)](ccu.md) — A communication accelerator coordinated by STARS.
