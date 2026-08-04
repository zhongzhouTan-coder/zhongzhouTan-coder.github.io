---
title: "Inner Product"
summary: "The scalar dot product Σ xᵢyᵢ; GEMM is the matrix of inner products between rows of A and columns of B, and attention scores are dot products."
tooltip: "An inner product (dot product) reduces two same-length vectors to a scalar by summing element-wise products. In GEMM, each output element is the inner product of a row of A and a column of B. The same operation scores query–key compatibility in attention, which is why the two concepts share the same name and notation."
layout: default
confidence: high
category: algorithms
sources:
  - raw/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.html
  - raw/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.metadata.json
  - derived/web-markdown/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.md
aliases:
  - dot product
  - scalar product
appears_in:
  - docs/algorithms/foundations/transformer.md
  - docs/algorithms/linear-attention/index.md
  - docs/hardware/spatial-gemm.md
  - docs/training/parallelism/sequence-parallelism/index.md
updated: 2026-08-04
---

# Inner Product

**Inner Product** (dot product) is the scalar $\sum_i x_i y_i$ produced by multiplying two same-length vectors element-wise and summing; a GEMM output element is the inner product of a row of $A$ with a column of $B$.

## Why It Exists

Every matrix product is a grid of inner products: $C[i][j] = A[i,:] \cdot B[:,j]$. Viewing GEMM this way makes each output element a self-contained reduction over the shared dimension $K$, which is how a single processing element (a multiply-accumulate unit) computes a result without needing its neighbors' data. The same operation underlies attention, where query–key compatibility is scored by a dot product, so the term spans both matrix multiply and attention.

## How It Works

For a single output element:

$$C[i][j] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]$$

The reduction over $k$ is the "inner" dimension — the one shared by both matrices. In an [inner-product](inner-product.md) GEMM structure, each output element is computed by a full reduction over $k$ before moving to the next element; in an [outer-product](outer-product.md) structure, all output elements advance one $k$-step together. In attention, the scaled dot product $q_i \cdot k_j / \sqrt{d_k}$ measures relevance before softmax.

## Tradeoffs

- Inner-product accumulation keeps each output element in one accumulator, minimizing on-chip storage per element, but serializes the reduction over $k$ per element.
- On parallel hardware the outer-product structure is often preferred because it exposes all pairwise multiplications at once; inner products excel on scalar or vector SIMD units.
- A dot product in a higher dimension is memory-bound when vectors are long relative to the work, which motivates [matrix tiling](matrix-tiling.md).

## Common Confusions

- **Dot product vs. inner product:** The same operation; "dot product" emphasizes the element-wise product-sum, "inner product" the general vector-space notion.
- **Inner-product GEMM vs. outer-product GEMM:** Both compute the same product; they differ in which loop is parallelized. Inner-product keeps the output element stationary; outer-product keeps the shared dimension step stationary.
- **Attention dot product vs. GEMM dot product:** Same arithmetic, different role — attention scores are dot products between a query and a key, then normalized by softmax.

## Where It Appears

- [Spatial: General Matrix Multiply tutorial](https://spatial-lang.org/gemm/) — Mentions inner-product GEMM as the alternative to the outer-product version it builds (linked from the companion Inner Product tutorial).
- [Spatial GEMM: Blocked Outer-Product Matrix Multiply](../hardware/spatial-gemm.md) — Frames inner-product accumulation as one of the GEMM decomposition choices.
- [Transformer Foundations](../algorithms/foundations/transformer.md) — Scaled dot-product attention: compatibility between query and key, with $\sqrt{d_k}$ scaling.
- [Linear Attention](../algorithms/linear-attention/index.md) — Query–key similarity is an ordinary dot product in feature space.
- [Sequence Parallelism](../training/parallelism/sequence-parallelism/index.md) — Each GPU computes dot products between its query chunk and every other rank's keys.

## Related Terms

- [General Matrix Multiply (GEMM)](gemm.md) — The kernel whose output elements are inner products.
- [Outer Product](outer-product.md) — The complementary rank-1 accumulation strategy.
- [Matrix Tiling](matrix-tiling.md) — Breaks the inner-product loop nest into on-chip-sized blocks.
