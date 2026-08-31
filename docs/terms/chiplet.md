---
title: "Chiplet"
summary: "A package-level design that combines multiple smaller dies into one logical processor or memory system."
tooltip: "A chiplet package distributes compute, IO, and memory functions across smaller dies connected by high-speed links. It improves modularity and scaling, but makes coherence, die-to-die bandwidth, packaging, and locality part of the architecture."
layout: default
confidence: medium
category: hardware
sources:
  - raw/hardware/ascend-950-npu-architecture-white-paper--paper.pdf
appears_in:
  - docs/hardware/ascend/ascend-950/index.md
updated: 2026-08-31
---

# Chiplet

**Chiplet** is a package-level architecture that combines multiple smaller dies into one logical computing or memory system through high-speed die-to-die links.

## Why It Exists

One large die becomes increasingly difficult and expensive to manufacture as compute, cache, IO, and memory-interface requirements grow. Chiplets let a design separate those functions, reuse die types, and scale package capacity while keeping the system boundary inside one package.

## How It Works

Each die owns a subsystem, such as AI compute or IO, and a die-to-die fabric connects the subsystems. In Ascend 950, two AI Dies, two IO Dies, and high-speed on-chip-memory modules form a unified memory-access system; hardware coherence and address translation determine whether that logical unity is efficient in practice.

## Tradeoffs

Chiplets add die-to-die latency, link traffic, packaging constraints, and locality decisions. A unified address space does not make every cross-die access as cheap as a local access.

## Common Confusions

- **Chiplet vs. multicore:** A multicore processor replicates cores inside one die; a chiplet package connects multiple dies that may contain different subsystem types.
- **Chiplet vs. a cluster:** Chiplets are packaged together; a cluster connects separate chips or servers through an external network.

## Where It Appears

- [Ascend 950 NPU Architecture: Compute, Memory, and Unified Bus](../hardware/ascend/ascend-950/index.md) — Combines two AI Dies, two IO Dies, and high-speed memory into a two-die UMA package.

## Related Terms

- [Global Memory](global-memory.md) — The large device-memory pool that the compute dies address.
- [Memory Banking](memory-banking.md) — A finer-grained on-chip parallel-access organization.
