---
title: "Tensor Parallelism"
summary: "A model-parallel strategy that splits weight matrices and hidden or head dimensions across accelerator ranks."
tooltip: "Tensor parallelism partitions the arithmetic of a Transformer layer across devices, usually using all-reduce or related collectives between linear layers. It helps a model fit and compute across GPUs, but frequent cross-rank collectives make multi-host scaling expensive."
layout: default
confidence: high
category: training
sources:
  - raw/algorithms/context-parallelism-scalable-million-token-inference--arxiv-2411.01783v3.pdf
aliases:
  - TP
  - tensor model parallelism
appears_in:
  - docs/algorithms/context-parallelism/index.md
updated: 2026-08-10
---

# Tensor Parallelism

**Tensor Parallelism** is a model-parallel strategy that splits a layer's weight matrices or hidden/head dimensions across accelerator ranks.

## Why It Exists

Large models may not fit on one accelerator, and even when they do, splitting matrix multiplications can reduce per-device compute time. The price is communication after partial results must be combined.

## How It Works

Column- and row-parallel linear layers divide the matrix work across ranks and use collectives such as all-reduce to make the next layer see the correct result. In the context-parallel paper, TP8 stays within each host while CP adds hosts along the sequence dimension.

## Tradeoffs

Tensor parallelism shards model weights, which CP does not, but its frequent collectives are costly across slower inter-host links. The useful TP degree is also constrained by hidden dimensions, attention heads, and the interconnect topology.

## Common Confusions

- **Tensor vs. context parallelism:** Tensor parallelism splits model computation; context parallelism splits request tokens and KV state.
- **Tensor vs. pipeline parallelism:** Tensor parallelism splits each layer across ranks; pipeline parallelism assigns different layers to stages.

## Where It Appears

- [Context Parallelism for Scalable Million-Token Inference](../algorithms/context-parallelism/index.md) - Uses TP8 within each host and CP across hosts to balance model fit with long-context latency.

## Related Terms

- [Context Parallelism](context-parallelism.md) - Splits sequence state rather than model weights.
- [Pipeline Parallelism](pipeline-parallelism.md) - Splits layers into stages.
- [All-Reduce](all-reduce.md) - Common collective for combining tensor-parallel partial results.
