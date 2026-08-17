---
title: "Speculative Decoding"
summary: "A lossless generation technique that drafts several tokens cheaply and has the target model verify them in parallel."
tooltip: "Speculative decoding uses a small or specialized drafter to propose a token block, then lets the target model accept a prefix and correct the first mismatch. It reduces serial target-model steps without changing the target distribution when strict rejection sampling is used."
layout: default
confidence: high
category: frameworks
sources:
  - raw/frameworks/eagle-speculative-sampling-feature-uncertainty--arxiv-2401.15077v3.pdf
  - raw/frameworks/eagle-2-dynamic-draft-trees--arxiv-2406.16858v2.pdf
  - raw/frameworks/eagle-3-scaling-inference-acceleration--arxiv-2503.01840v3.pdf
  - raw/frameworks/dspark-confidence-scheduled-speculative-decoding--arxiv-2607.05147v1.pdf
  - raw/frameworks/dflash-block-diffusion-flash-speculative-decoding--arxiv-2602.06036v2.pdf
aliases:
  - speculative sampling
  - assisted generation
mention_lint: off
appears_in:
  - docs/frameworks/eagle/index.md
  - docs/frameworks/eagle-2/index.md
  - docs/frameworks/eagle-3/index.md
  - docs/frameworks/dspark/index.md
  - docs/frameworks/dflash/index.md
updated: 2026-08-17
---

# Speculative Decoding

**Speculative Decoding** is a lossless generation technique that uses a cheap drafter to propose several tokens before the target model verifies them in parallel.

## Why It Exists

Autoregressive decoding makes the full target model produce one token at a time. If a smaller or specialized model can predict a plausible block, the target can score that block in one forward pass and convert several serial steps into one verification cycle.

## How It Works

The drafter generates a token block and records its distribution $\hat{p}_i$. The target model evaluates the same positions and produces $p_i$. Each proposed token is accepted from left to right with probability

$$
\min\left(1, \frac{p_i(\hat{t}_i)}{\hat{p}_i(\hat{t}_i)}\right).
$$

The first rejection is corrected with a residual target distribution, and every later draft token is discarded. With this strict rule, the accepted prefix plus the correction has the same distribution as ordinary target-model sampling.

## Tradeoffs

Speculative decoding helps when the drafter's early tokens are likely to survive and the target model has enough parallel capacity to verify them cheaply. It can hurt throughput when the batch is already saturated, the draft block is usually rejected early, or the runtime spends too much overhead packing and verifying variable-length drafts.

## Common Confusions

- **Speculative decoding vs. draft-model quality:** A high standalone next-token score is not enough; early prefix survival and verification cost determine the serving gain.
- **Speculative decoding vs. relaxed decoding:** The strict rejection rule preserves the target distribution. Relaxed multi-token acceptance methods may improve speed but require a separate quality or distribution analysis.
- **Speculative decoding vs. continuous batching:** Speculation changes how many candidate tokens one target step can process; continuous batching changes which requests share each step.

## Where It Appears

- [EAGLE: Feature-Level Speculative Sampling](../frameworks/eagle/index.md) - Predicts target-aligned features conditioned on the sampled token and verifies a candidate tree losslessly.
- [EAGLE-2: Context-Aware Dynamic Draft Trees](../frameworks/eagle-2/index.md) - Uses calibrated confidence and prefix survival to allocate the draft tree per context.
- [EAGLE-3: Training-Time Test for Speculative Decoding](../frameworks/eagle-3/index.md) - Removes feature regression, fuses target layers, and trains the drafter on self-produced states.
- [DSpark: Confidence-Scheduled Speculative Decoding](../frameworks/dspark/index.md) - Adds semi-autoregressive drafting, calibrated prefix survival, and hardware-aware verification scheduling.
- [DFlash: Block Diffusion for Flash Speculative Decoding](../frameworks/dflash/index.md) - Uses a target-conditioned block-diffusion adapter to draft many tokens in parallel.

## Related Terms

- [Continuous Batching](continuous-batching.md) - Iteration-level scheduling that can carry speculative tokens alongside ordinary decode work.
- [KV Cache](kv-cache.md) - Target-model state that speculative verification reuses during generation.
