---
title: "Looped Transformers"
summary: "Transformer variants that reuse one block's parameters across repeated applications, trading parameter count for effective computational depth."
tooltip: "Looped Transformers apply the same block, or a subset of blocks, several times per forward pass with shared weights. The loop adds depth without adding parameters, but it also adds serial latency and, when the loop includes global attention, KV-cache and attention cost per iteration."
layout: default
confidence: high
category: algorithms
sources:
  - raw/algorithms/looped-transformers-length-generalization--arxiv-2409.15647v5.pdf
  - raw/training/universal-yoco-efficient-depth-scaling--arxiv-2604.01220v1.pdf
aliases:
  - looped transformer
  - weight-shared recurrence
  - recursive depth
mention_aliases:
  - weight-shared recurrence
  - recursive depth
appears_in:
  - docs/training/efficient-attention/yoco/universal-yoco.md
  - docs/algorithms/looped-transformers-length-generalization/index.md
updated: 2026-09-10
---

# Looped Transformers

**Looped Transformers** are Transformer architectures that apply the same block parameters repeatedly within a forward pass, so effective computational depth exceeds the number of distinct parameter layers.

## Why It Exists

Standard depth scaling adds parameters and memory with every layer. A loop instead lets a fixed parameter budget express a range of depths: the Universal Transformer introduced the idea by re-applying a weight-shared stack, and later work showed that looping particular blocks, or all blocks, can buy inference-time computation, length generalization, or depth without a proportional parameter increase. The constraint is that recursion multiplies whatever the looped block costs per pass — including, for attention layers, [KV cache](kv-cache.md) growth and attention time.

## How It Works

For a block $f$ with shared weights, one loop iteration computes $X_{t+1} = f(X_t)$; running $T$ iterations produces the output that a $T$-times deeper model of the same block family would produce, using one copy of the parameters. The design space sits on three axes:

| Axis | Options | Consequence |
|---|---|---|
| What is looped | Whole stack, early layers only, or an attention-bounded subset | Determines whether each pass repeats global attention and cache writes. |
| How loops end | Fixed $T$, learned halting, or problem-size-dependent $T$ | Controls latency and how much compute a given input receives. |
| What is supervised | Final output only, per-step outputs, or step-dependent targets | Determines the training signal available to intermediate iterations. |

## Tradeoffs

- Latency is serial: $T$ iterations cannot be parallelized the way $T$ independent layers can.
- Looping attention layers multiplies cache and attention cost per iteration; looping only bounded-state modules keeps the overhead small.
- Gains diminish with $T$, since repeated application tends toward a fixed point.
- Choosing $T$ at inference couples to the task; without a reliable stopping signal, more loops can waste compute.

## Common Confusions

- **Looped Transformers vs. deeper Transformers:** A looped model performs more sequential computation with the same weights; a deeper model adds distinct parameter layers instead.
- **Looped Transformers vs. recurrent networks:** Loops share parameters across depth within one forward pass, while an RNN carries state across token steps.
- **Training-time loops vs. test-time scaling:** Looping a block is an architectural property; chain-of-thought, sampling, and self-consistency spend inference compute without changing the architecture.

## Where It Appears

- [Universal YOCO for Efficient Depth Scaling](../training/efficient-attention/yoco/universal-yoco.md) — Loops the shallow efficient-attention self-decoder so recursion adds depth while the global KV cache is still written once.
- [Looped Transformers for Length Generalization](../algorithms/looped-transformers-length-generalization/index.md) — Reuses one decoder block with input injection and step-dependent supervision so effective depth adapts to algorithmic input length.

## Related Terms

- [KV Cache](kv-cache.md) — The state that full-attention loops multiply, and the reason YOCO-U restricts recursion to efficient-attention blocks.
- [Sliding-Window Attention](sliding-window-attention.md) — A bounded-state attention used inside looped blocks.
- [Linear Attention](linear-attention.md) — A fixed-state attention family compatible with recursive loops.
