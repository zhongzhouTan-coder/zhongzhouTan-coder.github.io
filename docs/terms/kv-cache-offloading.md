---
title: "KV Cache Offloading"
summary: "Moves request-specific key/value state out of accelerator memory to a slower tier and fetches the needed entries during inference."
tooltip: "KV cache offloading trades scarce accelerator memory for bandwidth and latency by placing some or all request-specific attention state in CPU memory or storage. Selective prediction, transfer granularity, overlap, and reuse determine whether the slower tier is practical."
layout: default
confidence: medium
category: frameworks
sources:
  - raw/frameworks/serving/kvswap/kvswap-disk-aware-kv-cache-offloading-on-device-inference--arxiv-2511.11907v2.pdf
  - raw/frameworks/serving/kv-cache-optimization/kv-cache-optimization-strategies--arxiv-2603.20397v1.pdf
aliases:
  - KV offloading
  - key/value cache offloading
mention_lint: canonical
appears_in:
  - docs/frameworks/serving/kvswap/index.md
  - docs/frameworks/serving/kv-cache-optimization/index.md
  - docs/frameworks/serving/sglang/hicache-mooncake.md
updated: 2026-10-09
---

# KV Cache Offloading

**KV Cache Offloading** is moving request-specific key/value attention state out of accelerator memory to a slower memory or storage tier and fetching it when decoding needs it.

## Why It Exists

The [KV cache](kv-cache.md) grows with context length and batch size, so it can consume the memory needed for model weights, activations, or other requests. Offloading preserves a longer logical history without requiring every key and value to remain in the fastest tier.

## How It Works

An offloading runtime chooses a backing tier—such as CPU memory, NVMe, or eMMC—then coordinates cache placement, reads, writes, and the attention kernel. Systems may move the full cache, select likely-useful entries, compress metadata used for selection, or overlap transfers with computation. KVSwap keeps the complete cache on disk but predicts and reads groups of likely-useful entries for each layer.

Offloading changes where state lives; it does not inherently delete cache entries or change the model's learned weights. The practical cost is the movement of the fetched working set:

$$
T_{step} \approx \max(T_{compute}, T_{load}) + T_{management},
$$

when transfer and computation overlap effectively.

## Tradeoffs

Offloading increases the usable context or batch size, but the slower tier adds bandwidth, latency, read-amplification, energy, and operational costs. Selective loading can reduce movement but makes quality depend on the predictor and workload locality. Persistent storage also introduces endurance, privacy, crash-recovery, and multi-application isolation concerns that a throughput experiment may not cover.

## Common Confusions

- **Offloading vs. eviction:** Offloading moves state to another tier; eviction removes or discards state from the logical cache.
- **Offloading vs. compression:** Offloading changes placement; compression changes the representation or precision. A system can use both.
- **Offloading vs. paging:** Paging provides block-granularity allocation and logical-to-physical mapping; offloading decides which memory tier backs those blocks and when they move.

## Where It Appears

- [KVSwap: Disk-Aware KV Cache Offloading for Long-Context On-Device Inference](../frameworks/serving/kvswap/index.md) — Makes disk-first iterative decoding practical with grouped prediction, overlap, a rolling buffer, and reuse slots.
- [KV Cache Optimization Strategies for Scalable and Efficient LLM Inference](../frameworks/serving/kv-cache-optimization/index.md) — Places offloading in the hybrid-memory family and contrasts placement with eviction, compression, and alternative attention.
- [SGLang HiCache with Mooncake](../frameworks/serving/sglang/hicache-mooncake.md) — Shows GPU and instance-private host tiers backed by a configurable Mooncake L3 for shared prefix reuse.

## Related Terms

- [KV Cache](kv-cache.md) — The request-specific key/value state being moved.
- [PagedAttention](pagedattention.md) — A block-mapping scheme that can provide a logical cache view.
- [Low-Rank Approximation](low-rank-approximation.md) — A representation-reduction technique that can make cache prediction metadata smaller.
