---
title: "Attention-FFN Disaggregation"
summary: "Placing attention and feedforward modules on separately provisioned resources, exchanging representations across their execution boundary."
tooltip: "Placing attention and feedforward modules on separately provisioned resources, exchanging representations across their execution boundary. Its benefit depends on execution dependencies, resource placement, and communication cost."
layout: default
confidence: high
category: frameworks
sources:
  - raw/frameworks/serving/inference-optimization/llm-inference-optimization-practices-2026--caict.pdf
aliases:
  - AF separation
  - AFD
mention_lint: canonical
appears_in:
  - docs/frameworks/serving/inference-optimization/caict-2026/index.md
updated: 2026-10-31
---

# Attention-FFN Disaggregation

**Attention-FFN Disaggregation** places attention and feedforward modules on separately provisioned execution resources, exchanging token representations and results across their boundary.

## Why It Exists

Attention state access and feedforward matrix computation can favor different batching, parallelism, and hardware choices. This is particularly relevant to large [mixture-of-experts](mixture-of-experts.md) deployments.

## How It Works

Attention workers retain sequence state and compute attention-side operations. Feedforward workers process token representations, including routed expert computation in a MoE design, and return results. These handoffs repeat across layers and tokens; pipelining different work can overlap computation and communication.

## Tradeoffs

Independent provisioning can improve utilization, but frequent exchanges make network bandwidth, latency, batching, and overlap critical. One token's dependent attention and feedforward operations still need ordered execution.

## Common Confusions

[Prefill-decode disaggregation](prefill-decode-disaggregation.md) separates phases of a request and hands off KV state. AF separation splits modules within the model. The two choices can coexist, and neither implies the other.

## Where It Appears

- [CAICT 2026: Inference Optimization as a System](../frameworks/serving/inference-optimization/caict-2026/index.md) — Explains the distinct transfer boundary through the report's MegaScale-Infer and Step-3 summaries (§IV.4).

## Related Terms

- [Mixture of Experts](mixture-of-experts.md)
- [KV Cache](kv-cache.md)
- [Tensor Parallelism](tensor-parallelism.md)
