---
title: "Global Memory"
summary: "The off-chip device memory on an accelerator (GPU HBM or Ascend GM) that holds full tensors; kernels move tiles from it into on-chip storage to compute."
tooltip: "Global memory (GM) is the large, relatively slow device memory outside the compute units — GPU HBM or the Ascend NPU's GM. Kernels must move data from it into fast on-chip buffers before computing, so traffic to and from global memory usually dominates kernel performance."
layout: default
confidence: high
category: hardware
sources:
  - raw/frameworks/cann-ascendc-basic-architecture--web-2026-07-31-87f687f6b225.html
  - derived/web-markdown/frameworks/cann-ascendc-basic-architecture--web-2026-07-31-87f687f6b225.md
aliases:
  - GM
  - HBM
  - device memory
appears_in:
  - docs/frameworks/triton-ascend/operator-mechanisms.md
  - docs/algorithms/flashattention-3.md
  - docs/hardware/flatquant.md
  - docs/training/minimax-sparse-attention/index.md
updated: 2026-07-31
---

# Global Memory

**Global Memory** is the off-chip device memory on an accelerator — GPU HBM or the Ascend NPU's GM — where full tensors and kernel inputs and outputs live, and from which kernels move data into fast on-chip storage to compute.

## Why It Exists

Accelerator chips hold only a tiny amount of fast on-chip storage (UB, L1, L0, SRAM, registers). All the data a kernel works on — weights, activations, KV caches — must live somewhere large enough, so every accelerator provides a high-capacity device memory that all cores can address.

## How It Works

Kernels read tiles from global memory into on-chip buffers, compute, and write results back. On Ascend, dedicated movement engines handle the transfer: MTE2 (`GM → UB/L1/L0A/L0B`), MTE3 (`UB → GM`), and FixPipe (`L0C → GM/L1`), with L2 caching GM traffic. On NVIDIA GPUs the equivalent is HBM reached through L2/global loads and stores. The typical data paths are `GM → UB → Vector → UB → GM` for vector work and `GM → L1/L0A/L0B → Cube → L0C → FixPipe` for matrix work.

## Tradeoffs

Global memory is the performance bottleneck of most kernels: bandwidth is far lower and latency far higher than on-chip storage. Tiling, double buffering, L1/L2 reuse, and kernel fusion all exist mainly to reduce GM traffic and avoid writing intermediate results back to GM.

## Common Confusions

- **Global memory vs. on-chip storage:** Global memory is the large off-chip pool; on-chip storage (UB, L1, L0, SRAM/shared memory) is small, fast, and per-core.
- **GM vs. host RAM:** Global memory is device memory on the accelerator card; host RAM is separate CPU memory that requires an explicit copy (or unified memory) to access.

## Where It Appears

- [Triton Ascend Operator Mechanisms](../frameworks/triton-ascend/operator-mechanisms.md) — GM is the source and destination of every tile movement on the Ascend NPU.
- [FlashAttention-3](../algorithms/flashattention-3.md) — Labels GPU HBM as global memory holding the Q/K/V/O tiles in its dataflow diagram.
- [FlatQuant](../hardware/flatquant.md) — Fuses affine transformation and quantization so only the quantized result is written to global memory.
- [MiniMax Sparse Attention](../training/minimax-sparse-attention/index.md) — Emits LSE tensors directly to global memory for the KL-loss backward kernel.

## Related Terms

- [KV Cache](kv-cache.md) — A runtime state structure whose footprint is often the dominant consumer of global-memory capacity during decoding.
