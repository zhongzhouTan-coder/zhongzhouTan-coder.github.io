---
title: "Grouped-Query Attention"
summary: "An attention layout that shares each key/value head across a group of query heads to reduce KV-cache and communication cost."
tooltip: "Grouped-query attention keeps many query heads but uses fewer key/value heads, so several queries reuse one KV projection. It preserves more capacity than multi-query attention while making cached KV reads and distributed KV messages smaller."
layout: default
confidence: high
category: algorithms
sources:
  - raw/algorithms/context-parallelism-scalable-million-token-inference--arxiv-2411.01783v3.pdf
aliases:
  - GQA
  - grouped query attention
appears_in:
  - docs/algorithms/context-parallelism/index.md
updated: 2026-08-10
---

# Grouped-Query Attention

**Grouped-Query Attention** is an attention layout in which multiple query heads share a smaller set of key/value heads.

## Why It Exists

Autoregressive inference stores and reads K/V for every layer and token. Reducing the number of KV heads cuts that memory traffic while retaining more query-head diversity than single-KV-head multi-query attention.

## How It Works

If a model has $N_H$ query heads and $N_{KV}$ KV heads, each KV head serves approximately $N_H/N_{KV}$ query heads. In the paper's Llama3 405B setup, 128 query heads share 8 KV heads, so a CP ring can communicate a much smaller KV payload than a query-shaped payload.

## Tradeoffs

Fewer KV heads reduce cache and communication cost, but the model must be trained or adapted for the grouping. The ratio also changes context-parallel pass selection because it sets the relative size of Q versus KV messages.

## Common Confusions

- **GQA vs. MQA:** GQA uses several KV heads; MQA uses one shared KV head for all query heads.
- **GQA vs. context parallelism:** GQA reduces the payload per token; context parallelism distributes tokens and communicates those payloads across ranks.

## Where It Appears

- [Context Parallelism for Scalable Million-Token Inference](../algorithms/context-parallelism/index.md) - Uses Llama3's 128 query heads and 8 KV heads to make pass-KV communication efficient.
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) - Introduces grouped-query adaptation from multi-head checkpoints.

## Related Terms

- [KV Cache](kv-cache.md) - Stores the grouped key/value states during decoding.
- [Context Parallelism](context-parallelism.md) - Distributes the grouped KV state across ranks.
