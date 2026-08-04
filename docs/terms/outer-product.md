---
title: "Outer Product"
summary: "A rank-1 matrix u vᵀ formed from two vectors; GEMM can be computed by accumulating outer products of columns of A with rows of B."
tooltip: "An outer product takes a column u and a row vᵀ and forms the rank-1 matrix u·vᵀ. GEMM is often structured as a sum of outer products: for each shared index k, add column k of A times row k of B. This exposes every pairwise multiplication as independent work, which parallel hardware and systolic-style dataflow exploit."
layout: default
confidence: high
category: algorithms
sources:
  - raw/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.html
  - raw/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.metadata.json
  - derived/web-markdown/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.md
aliases:
  - outer product accumulation
  - rank-1 update
appears_in:
  - docs/algorithms/linear-attention/index.md
  - docs/algorithms/linear-attention/linear-attention-without-softmax.md
  - docs/hardware/spatial-gemm.md
  - docs/terms/delta-rule.md
updated: 2026-08-04
---

# Outer Product

**Outer Product** is the rank-1 matrix $u \otimes v = u v^\top$ formed from a column vector $u$ and a row vector $v^\top$; in GEMM, it is the unit of computation when the product is accumulated as a sum of outer products.

## Why It Exists

The naive GEMM loop iterates over output elements and the shared dimension $k$. For each fixed $k$, the multiplication of column $k$ of $A$ with row $k$ of $B$ produces an entire $M \times N$ block of pairwise products — every element in that block is independent. Structuring the computation as "multiply the whole column by the whole row" turns those pairwise multiplications into parallel work instead of serial accumulation per output element.

## How It Works

The matrix product decomposes into $K$ outer products:

$$C = \sum_{k=0}^{K-1} A[:,k] \cdot B[k,:]$$

Each term is a rank-1 matrix added into the accumulator. In the Spatial GEMM tutorial, the innermost loop selects column `kk` of `tileA` and row `kk` of `tileB`, computes `tileA(ii,kk) * tileB(kk,jj)` for every `(ii,jj)` pair, and a `MemReduce` accumulates the partial tiles. This is also how recurrent linear-attention states are built: each new key–value pair contributes an outer product to the state matrix.

## Tradeoffs

- Outer-product accumulation parallelizes the inner $M \times N$ block but needs an accumulator per tile, which consumes on-chip SRAM/registers.
- It pairs naturally with [systolic-array](systolic-array.md) dataflow, where one operand is stationary and the other streams through.
- The alternative [inner-product](inner-product.md) view keeps one output element stationary and serializes over $k$ — better when the accumulator must stay in a single register.

## Common Confusions

- **Outer product vs. inner product:** An outer product of two length-$K$ vectors is a $K \times K$ matrix; an inner product is a scalar. Both are GEMM building blocks, but they parallelize opposite directions of the loop nest.
- **Outer product vs. Kronecker product:** The [Kronecker product](kronecker-product.md) is a block-structured product of two full matrices; the outer product is the rank-1 special case of two vectors.

## Where It Appears

- [Spatial: General Matrix Multiply tutorial](https://spatial-lang.org/gemm/) — Builds the blocked GEMM explicitly from outer products of `tileA` columns and `tileB` rows.
- [Spatial GEMM: Blocked Outer-Product Matrix Multiply](../hardware/spatial-gemm.md) — The tutorial's main demonstration is outer-product accumulation.
- [Linear Attention](../algorithms/linear-attention/index.md) — The KV state is a prefix sum of key–value outer products.
- [Linear Attention Without Softmax](../algorithms/linear-attention/linear-attention-without-softmax.md) — The parallel training mode materializes all token outer products before a cumulative sum.
- [Delta Rule](delta-rule.md) — The associative memory is updated by adding key–value outer products.

## Related Terms

- [General Matrix Multiply (GEMM)](gemm.md) — The kernel this accumulation strategy implements.
- [Inner Product](inner-product.md) — The complementary element-wise accumulation strategy.
- [Matrix Tiling](matrix-tiling.md) — Blocks the loop nest that outer products fill in.
- [Kronecker Product](kronecker-product.md) — The block-structured product of two full matrices; the outer product is its rank-1 special case.
