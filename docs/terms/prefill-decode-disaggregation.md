---
title: "Prefill-Decode Disaggregation"
summary: "Separating prompt processing and autoregressive generation onto independently scheduled execution resources."
tooltip: "Separating prompt processing and autoregressive generation onto independently scheduled execution resources. Its benefit depends on execution dependencies, resource placement, and communication cost."
layout: default
confidence: high
category: frameworks
sources:
  - raw/frameworks/serving/inference-optimization/llm-inference-optimization-practices-2026--caict.pdf
aliases:
  - PD separation
  - PD disaggregation
mention_lint: canonical
appears_in:
  - docs/frameworks/serving/inference-optimization/caict-2026/index.md
updated: 2026-10-31
---

# Prefill-Decode Disaggregation

**Prefill-Decode Disaggregation** separates prompt processing and autoregressive generation onto independently scheduled execution resources, with attention state transferred or made accessible between them.

## Why It Exists

Long prompt processing and latency-sensitive token generation can interfere when they share the same execution resources. Separate phases can use different batch sizes, parallel strategies, and resource allocations.

## How It Works

A prefill worker processes the prompt and produces [KV cache](kv-cache.md). A decode worker receives or accesses compatible state and continues generation. A controller selects workers and coordinates state readiness; routing can consider cache locality and downstream capacity.

## Tradeoffs

The split adds state-transfer overhead and coordination. It is attractive only when reduced interference or better resource matching compensates for that cost. Shared storage capacity alone does not ensure low-latency access.

## Common Confusions

- [Chunked prefill](chunked-prefill.md) splits prompt work into schedulable pieces and can run with colocated decode.
- [Attention-FFN disaggregation](attention-ffn-disaggregation.md) splits layer modules and exchanges token representations repeatedly, rather than separating request phases.

## Where It Appears

- [CAICT 2026: Inference Optimization as a System](../frameworks/serving/inference-optimization/caict-2026/index.md) — Explains phase separation, cache handoff, and SLO-driven provisioning from report §§III–IV.

## Related Terms

- [KV Cache](kv-cache.md)
- [Prefix Caching](prefix-caching.md)
- [Continuous Batching](continuous-batching.md)
