---
title: "URMA"
summary: "Unified Bus's asynchronous remote-memory access semantic for copies, messages, and atomic operations."
tooltip: "URMA (UB Remote Memory Access) submits asynchronous work through Jetty queues and Doorbells, then uses the transport and UMMU for remote transfer, address translation, and permissions. It supports reliable RTP and lighter CTP modes with different bandwidth and retransmission tradeoffs."
layout: default
confidence: medium
category: hardware
sources:
  - raw/hardware/ascend-950-npu-architecture-white-paper--paper.pdf
appears_in:
  - docs/hardware/ascend/ascend-950/index.md
updated: 2026-08-31
---

# URMA

**URMA** is the Unified Bus Remote Memory Access semantic for asynchronously submitting remote copies, messages, reads, writes, and atomic operations.

## Why It Exists

Distributed AI execution needs communication to overlap with computation. A blocking, host-mediated copy makes the sender wait and forces the CPU to manage every transfer; asynchronous work queues let the device continue until completion is needed.

## How It Works

An AI Core, AI CPU, or STARS rings a Doorbell for a URMA Jetty queue. The operation travels through an interconnect port, while UMMU performs virtual-to-physical translation and permission checks. The paper lists Write, Read, Send, FetchAdd, and CompareAndSwap operations, with RTP for reliable transport and CTP for a lighter mode without end-to-end retransmission.

## Tradeoffs

RTP provides stronger reliability but has different port/bandwidth limits than CTP; CTP can be useful when lower transport overhead is acceptable but cannot be treated as end-to-end reliable by itself. Queue depth, completion handling, and route contention also remain software concerns.

## Common Confusions

- **URMA vs. UB Memory:** URMA is asynchronous queue-based communication; UB Memory is synchronous remote Load/Store/Atomic access.
- **URMA vs. RDMA:** URMA is the Ascend Unified Bus semantic described by this white paper; the shared idea is remote memory access, not identical protocol behavior.

## Where It Appears

- [Ascend 950 NPU Architecture: Compute, Memory, and Unified Bus](../hardware/ascend/ascend-950/index.md) — Traces URMA Jetty submission, transport modes, and its role in CCU and scale-out paths.

## Related Terms

- [Unified Bus (UB)](unified-bus.md) — The interconnect family that exposes URMA.
- [All-to-All](all-to-all.md) — A distributed exchange pattern that can use asynchronous movement.
- [All-Reduce](all-reduce.md) — A reduction pattern that can be offloaded through CCU.
