---
title: "Unified Bus (UB)"
summary: "Huawei's interconnect family for remote memory, asynchronous movement, collectives, and Ethernet-connected scale-out."
tooltip: "Unified Bus (UB) exposes multiple communication semantics under one interconnect family: synchronous remote Load/Store/Atomic access, asynchronous URMA operations, and collective offload through CCU. UBoE extends the scale-out path over ordinary Ethernet switches, while topology and port reuse still determine performance."
layout: default
confidence: medium
category: hardware
sources:
  - raw/hardware/ascend-950-npu-architecture-white-paper--paper.pdf
appears_in:
  - docs/hardware/ascend/ascend-950/index.md
updated: 2026-08-31
---

# Unified Bus (UB)

**Unified Bus (UB)** is Huawei's interconnect family that combines remote-memory semantics, asynchronous transport, collective communication, and Ethernet-connected scale-out for Ascend systems.

## Why It Exists

Large-model systems need more than point-to-point messages: they need bulk transfers, fine-grained atomics, shared-memory-style access, reductions, and scalable topologies. Forcing all of those operations through a host protocol increases software complexity and adds avoidable latency.

## How It Works

UB 2.0 provides synchronous UB Memory Load/Store/Atomic operations, asynchronous [URMA](urma.md) copy/message operations, and [CCU](ccu.md) collective missions. Ascend 950 exposes 18 x4 ports through HiLink SerDes, supports Full Mesh/Clos/nD-Mesh-style topologies, and can use UBoE to carry UB scale-out traffic over Ethernet.

## Tradeoffs

UB's unified programming family does not make remote access local. Link speed, port reuse, route contention, reliability mode, address translation, and switch topology remain visible performance and correctness boundaries.

## Common Confusions

- **Unified Bus vs. Unified Buffer:** UB means the inter-chip Unified Bus in this term; Unified Buffer is the Vector Core's local on-chip storage.
- **Unified Bus vs. PCIe:** PCIe is a host/IO protocol that Ascend 950 also supports; UB adds AI-oriented remote-memory and collective semantics.
- **UB vs. UBoE:** UBoE is the Ethernet attachment of UB, not a separate replacement for the UB protocol.

## Where It Appears

- [Ascend 950 NPU Architecture: Compute, Memory, and Unified Bus](../hardware/ascend/ascend-950/index.md) — Connects UB Memory, URMA, CCU, port reuse, and supernode topology.

## Related Terms

- [URMA](urma.md) — UB's asynchronous remote-memory and message semantic.
- [Collective Communication Unit (CCU)](ccu.md) — Hardware collective offload on the UB path.
- [All-Reduce](all-reduce.md) — One collective pattern that UB/CCU can accelerate.
