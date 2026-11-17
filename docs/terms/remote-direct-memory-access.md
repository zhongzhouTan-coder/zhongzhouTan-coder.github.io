---
title: "Remote Direct Memory Access (RDMA)"
summary: "A network access mechanism that lets a node read, write, or atomically modify registered remote memory without involving the remote CPU in each one-sided operation."
tooltip: "RDMA lets a node access registered memory on another machine through the network interface. One-sided operations avoid remote CPU handling per operation, but network bandwidth, interface processing, synchronization, and transport semantics still matter."
layout: default
confidence: high
category: frameworks
sources:
  - raw/frameworks/infrastructure/flexkv/flexkv--arxiv-2512.16148v1.pdf
aliases:
  - remote direct memory access
  - RDMA
mention_aliases:
  - RDMA
mention_lint: aliases
appears_in:
  - docs/frameworks/infrastructure/flexkv/index.md
updated: 2026-11-16
---

# Remote Direct Memory Access (RDMA)

**Remote Direct Memory Access (RDMA)** is a network mechanism for accessing registered memory on a remote machine without involving its CPU in each one-sided operation.

## Why It Exists

Distributed systems need to move or update data without paying remote software processing costs for every small operation. RDMA exposes network-interface operations that can access remote memory directly.

## How It Works

A node issues one-sided reads, writes, or supported atomic operations against registered remote memory; the remote network interface handles the operation without an application handler on the remote CPU. RDMA also supports two-sided send/receive operations, where the receiver participates; FlexKV uses these to implement RPCs between compute-node proxies.

## Tradeoffs

Bypassing a remote CPU does not make an operation free: the network interface, link, and remote memory can bottleneck. In FlexKV's testbed, high rates of remote compare-and-swap motivate moving some index work to compute CPUs; that observation is not a universal ranking of RDMA operations.

## Common Confusions

- **One-sided vs. two-sided:** One-sided access does not invoke a remote application handler per operation; send/receive needs a receiver and can carry an RPC.
- **RDMA vs. URMA:** [URMA](urma.md) is a distinct Unified Bus remote-memory semantic; shared concepts do not imply identical protocols.

## Where It Appears

- [FlexKV: Flexible Index Offloading](../frameworks/infrastructure/flexkv/index.md) — Contrasts one-sided remote atomics with two-sided RPCs and local CPU atomics.

## Related Terms

- [URMA](urma.md) — An asynchronous remote-memory and messaging semantic in Huawei's Unified Bus.
