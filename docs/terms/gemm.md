---
title: "General Matrix Multiply (GEMM)"
summary: "The dense multiply-accumulate kernel C = A×B whose execution rate is the standard performance reference for linear-algebra workloads on GPUs and NPUs."
tooltip: "GEMM (General Matrix Multiply) computes C = A×B with dense multiply-accumulate work. Almost every neural-network layer reduces to GEMMs — attention projections, MLP weights, and convolution — so its throughput is the roofline reference that kernels like attention are compared against. Implementations break the product into blocks using inner products, outer products, or systolic arrays, and tile it across the memory hierarchy."
layout: default
confidence: high
category: algorithms
sources:
  - raw/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.html
  - raw/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.metadata.json
  - derived/web-markdown/hardware/spatial-gemm--web-2026-08-04-5ae8cd1f3ba2.md
aliases:
  - GEMM
  - general matrix multiplication
  - matrix multiply
  - matmul
appears_in:
  - docs/algorithms/flashattention/flashattention-3.md
  - docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md
  - docs/frameworks/triton-ascend/operator-mechanisms.md
  - docs/hardware/quantization/nvfp4.md
  - docs/hardware/spatial-gemm.md
  - docs/training/parallelism/megatron-lm/index.md
updated: 2026-08-04
---

# General Matrix Multiply (GEMM)

**General Matrix Multiply (GEMM)** is the dense multiply-accumulate kernel that computes $C = A \times B$ for matrices of arbitrary sizes, and whose throughput is the standard performance reference for dense linear algebra.

## Why It Exists

Neural networks are, at their core, a long chain of matrix multiplications: attention projections, MLP weights, and even convolution (via im2col or implicit-GEMM). If the GEMM kernel is slow, the whole model is slow. Optimized GEMM kernels reach 80–90% of peak GPU throughput — the number FlashAttention-3 compares attention against — so GEMM is the roofline target that every other kernel is measured by.

## How It Works

Each output element is a length-$K$ inner product:

$$C[m][n] = \sum_{k=0}^{K-1} A[m][k] \cdot B[k][n]$$

The full product is $M \times N \times K$ multiply-accumulate operations. Because that work dwarfs the data size, the practical problem is **data movement, not FLOPs**: operands must be fetched from global memory into on-chip storage and reused many times. GEMM implementations therefore break the computation into blocks and pick an accumulation strategy. The Spatial tutorial enumerates the main options: [matrix tiling](matrix-tiling.md) to block the loops, [inner-product](inner-product.md) or [outer-product](outer-product.md) accumulation, and [systolic array](systolic-array.md) dataflow. On hardware, tensor cores (NVIDIA), the Ascend Cube unit, and TPU systolic arrays execute GEMM as a single fused instruction with an FP32-accumulate epilogue.

## Tradeoffs

- GEMM is **compute-bound** only when tiles are large enough to amortize memory traffic; small matrices are memory-bound and need kernel fusion or batched dispatch.
- Tiling must match the on-chip capacity ([memory banking](memory-banking.md) and SRAM size); too-large tiles spill, too-small tiles underutilize the compute units.
- **Low-precision GEMMs (FP8, FP4, INT4)** multiply memory savings against accuracy; the [NVFP4](../hardware/quantization/nvfp4.md) format exists precisely to keep GEMM layout constraints tractable.

## Common Confusions

- **GEMM vs. matmul:** The same operation; GEMM emphasizes the general (non-square, non-identity) case with an optional bias/epilogue, as in cuBLAS `GEMM`.
- **GEMM vs. dot product:** A GEMM is a matrix of inner products; a single dot product is one output element.
- **GEMM vs. convolution:** Dense GEMM is a distinct kernel; convolution is often *lowered* to GEMM (implicit-GEMM) rather than being a GEMM itself.

## Where It Appears

- [Spatial: General Matrix Multiply tutorial](https://spatial-lang.org/gemm/) — Builds a blocked, outer-product GEMM in the Spatial DSL and discusses inner-product and systolic-array variants plus SRAM banking.
- [Spatial GEMM: Blocked Outer-Product Matrix Multiply](../hardware/spatial-gemm.md) — The knowledge-base insight page for the same tutorial: output tiling, MemFold/MemReduce, triple buffering, and banking.
- [FlashAttention-3](../algorithms/flashattention/flashattention-3.md) — Measures attention against optimized GEMM kernels that reach 80–90% of H100 peak, and overlaps GEMM with softmax.
- [MiniMax GQA W4A4 Quantization Path](../frameworks/vllm/minimax-gqa-w4a4-quantization-path.md) — W4A4 inference runs FP4 GEMM kernels (FlashInfer MXFP4, CUTLASS NVFP4) with dynamic activation quantization.
- [Triton Ascend Operator Mechanisms](../frameworks/triton-ascend/operator-mechanisms.md) — The Ascend Cube unit executes GEMM / QK·PV matrix multiply-accumulate.
- [NVFP4](../hardware/quantization/nvfp4.md) — Defines GEMM layout constraints (TN-only) for 4-bit float weights and activations.
- [Megatron-LM](../training/parallelism/megatron-lm/index.md) — Splits transformer GEMMs column- and row-wise across GPUs for tensor parallelism.

## Related Terms

- [Matrix Tiling](matrix-tiling.md) — The blocking strategy that makes GEMM compute-bound.
- [Inner Product](inner-product.md) — Element-wise accumulation view of GEMM.
- [Outer Product](outer-product.md) — Rank-1 accumulation view of GEMM.
- [Systolic Array](systolic-array.md) — A spatial dataflow that executes GEMM with maximal weight reuse.
- [Memory Banking](memory-banking.md) — Conflict-free parallel access to the on-chip SRAM holding GEMM tiles.
