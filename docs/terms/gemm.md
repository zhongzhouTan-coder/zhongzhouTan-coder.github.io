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
  - docs/algorithms/flashattention/flashattention-2.md
  - docs/algorithms/flashattention/flashattention-3.md
  - docs/algorithms/flashattention/flashattention-4.md
  - docs/algorithms/flashattention/index.md
  - docs/algorithms/index.md
  - docs/algorithms/kronecker-product.md
  - docs/frameworks/deepseek/v4-attention-code-reading.md
  - docs/frameworks/cuda/index.md
  - docs/frameworks/triton-ascend/cannbot-skills-workflow.md
  - docs/frameworks/triton-ascend/index.md
  - docs/frameworks/triton-ascend/operator-mechanisms.md
  - docs/frameworks/triton/index.md
  - docs/frameworks/triton/triton-in-vllm.md
  - docs/frameworks/tilelang/index.md
  - docs/frameworks/tilelang-ascend/index.md
  - docs/frameworks/vllm-ascend/deepseek-v4-inference.md
  - docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md
  - docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md
  - docs/frameworks/vllm/vllm-kimi-k3-code-reading.md
  - docs/hardware/index.md
  - docs/hardware/quantization/flatquant/index.md
  - docs/hardware/quantization/index.md
  - docs/hardware/quantization/microscaling-mx-formats/index.md
  - docs/hardware/quantization/nvfp4.md
  - docs/hardware/spatial-gemm.md
  - docs/training/efficient-attention/gated-delta-networks/index.md
  - docs/training/kimi/kimi-linear/index.md
  - docs/training/mhc/index.md
  - docs/training/parallelism/index.md
  - docs/training/parallelism/megatron-lm/index.md
updated: 2026-08-10
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

- [CUDA Programming Model: From Host to SM, Warp, and Memory](../frameworks/cuda/index.md) — Uses vector addition and tile programming to show how device-memory traffic and on-chip reuse shape kernel cost.
- [Spatial: General Matrix Multiply tutorial](https://spatial-lang.org/gemm/) — Builds a blocked, outer-product GEMM in the Spatial DSL and discusses inner-product and systolic-array variants plus SRAM banking.
- [Spatial GEMM: Blocked Outer-Product Matrix Multiply](../hardware/spatial-gemm.md) — The knowledge-base insight page for the same tutorial: output tiling, MemFold/MemReduce, triple buffering, and banking.
- [FlashAttention-3](../algorithms/flashattention/flashattention-3.md) — Measures attention against optimized GEMM kernels that reach 80–90% of H100 peak, and overlaps GEMM with softmax.
- [MiniMax GQA W4A4 Quantization Path](../frameworks/vllm/minimax-gqa-w4a4-quantization-path.md) — W4A4 inference runs FP4 GEMM kernels (FlashInfer MXFP4, CUTLASS NVFP4) with dynamic activation quantization.
- [FlatQuant](../hardware/quantization/flatquant/index.md) — Quantizes activations to INT4 inside a fused affine-transform kernel, then multiplies by pre-transformed INT4 weights with a CUTLASS GEMM.
- [Triton Ascend Operator Mechanisms](../frameworks/triton-ascend/operator-mechanisms.md) — The Ascend Cube unit executes GEMM / QK·PV matrix multiply-accumulate.
- [NVFP4](../hardware/quantization/nvfp4.md) — Defines GEMM layout constraints (TN-only) for 4-bit float weights and activations.
- [Megatron-LM](../training/parallelism/megatron-lm/index.md) — Splits transformer GEMMs column- and row-wise across GPUs for tensor parallelism.
- [FlashAttention-2: Better Parallelism and Work Partitioning](../algorithms/flashattention/flashattention-2.md) — FlashAttention-2 algorithm: reduced non-matmul overhead, sequence-parallel attention blocks, warp-level work partitioning, and.
- [FlashAttention-4: Blackwell Attention Kernel Co-Design](../algorithms/flashattention/flashattention-4.md) — FlashAttention-4 algorithm and kernel-pipeline techniques for faster exact attention on NVIDIA Blackwell GPUs.
- [FlashAttention](../algorithms/flashattention/index.md) — The FlashAttention algorithm and kernel family across GPU generations: IO-aware exact attention, parallelism, Hopper asynchrony.
- [Algorithms](../algorithms/index.md) — Algorithm pages covering inference algorithms, attention kernels, and scheduling methods.
- [Kronecker Product](../algorithms/kronecker-product.md) — The block-structured matrix product A⊗B (matrix direct product), foundational to tensor factorization and the Kronecker.
- [DeepSeek V4 Attention: Code Reading Map](../frameworks/deepseek/v4-attention-code-reading.md) — A navigable map of the DeepSeek V4 hybrid compressed attention implementation across vLLM (NVIDIA/AMD/XPU) and vllm-ascend.
- [Triton: Tiled GPU Kernel Language and Compiler](../frameworks/triton/index.md) — The original Triton language and compiler for expressing tiled neural network computations as portable, high-performance GPU.
- [Triton in Practice: How vLLM and vllm-ascend Use Triton](../frameworks/triton/triton-in-vllm.md) — A codebase-driven tour of Triton kernel development in vLLM (NVIDIA GPU) and vllm-ascend (Ascend NPU), covering infrastructure.
- [TileLang Design and Code Learning Path](../frameworks/tilelang/index.md) — Uses a tiled GEMM to connect TileLang's Python DSL to TIRX, target-specific lowering, and runtime adapters.
- [TileLang-Ascend: Ascend Backend and TileLang Integration](../frameworks/tilelang-ascend/index.md) — Uses Ascend GEMM as the concrete path from TileLang IR through Ascend C/PTO codegen and the CANN runtime.
- [CANNBot Skills: Triton Ascend Development Workflow](../frameworks/triton-ascend/cannbot-skills-workflow.md) — How CANNBot's seven Triton-domain skills and the triton-op-generator plugin orchestrate end-to-end Triton Ascend kernel.
- [Triton Ascend: Ascend NPU Backend for Triton](../frameworks/triton-ascend/index.md) — A beginner-friendly tour of triton-ascend: how it bridges the Triton GPU kernel language to Huawei Ascend NPU hardware through a.
- [vLLM Kimi K3 Code Reading Map](../frameworks/vllm/vllm-kimi-k3-code-reading.md) — Code-reading map for upstream vLLM's real Kimi K3 implementation: request parsing, multimodal wrapper, KimiLinear text model.
- [DeepSeek-V4 Inference on Ascend: The DSA Serving Stack in vllm-ascend](../frameworks/vllm-ascend/deepseek-v4-inference.md) — How vllm-ascend runs DeepSeek-V4 end to end on Ascend NPUs: model override with mHC hyper-connections, hybrid c4/c128 compressor.
- [DeepSeek-V4 Lightning Indexer C8 Quantization: INT8/FP8 Indexer Cache in vllm-ascend](../frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md) — How vllm-ascend quantizes the DeepSeek-V4 Lightning Indexer to 8 bits (C8): INT8 keys with FP16 scales on 910B/A2/A3, FP8 e4m3fn.
- [Hardware and Numerics](../hardware/index.md) — Hardware and numerics pages covering accelerator features, precision formats, and related implementation details.
- [Quantization](../hardware/quantization/index.md) — Post-training quantization methods and low-precision numeric formats for LLM inference.
- [Microscaling (MX) Formats: Block Floating Point for AI Hardware](../hardware/quantization/microscaling-mx-formats/index.md) — Uses MX blocks as the narrow-precision operands inside matrix multiplies and describes the format's accumulation and layout boundaries.
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](../training/efficient-attention/gated-delta-networks/index.md) — Gated DeltaNet combines global state decay with key-targeted delta updates to improve fixed-state sequence memory while.
- [Kimi Linear: Expressive Efficient Attention Architecture](../training/kimi/kimi-linear/index.md) — Kimi Linear is a hybrid linear attention architecture that for the first time outperforms full attention across short-context.
- [mHC: Manifold-Constrained Hyper-Connections](../training/mhc/index.md) — DeepSeek's mHC projects Hyper-Connections' residual mixing matrix onto the doubly stochastic manifold with Sinkhorn-Knopp.
- [Training Parallelism](../training/parallelism/index.md) — Data, tensor, pipeline, and sequence parallelism techniques for large-model training.

## Related Terms

- [Matrix Tiling](matrix-tiling.md) — The blocking strategy that makes GEMM compute-bound.
- [Inner Product](inner-product.md) — Element-wise accumulation view of GEMM.
- [Outer Product](outer-product.md) — Rank-1 accumulation view of GEMM.
- [Systolic Array](systolic-array.md) — A spatial dataflow that executes GEMM with maximal weight reuse.
- [Memory Banking](memory-banking.md) — Conflict-free parallel access to the on-chip SRAM holding GEMM tiles.
