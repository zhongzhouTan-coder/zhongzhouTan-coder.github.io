---
title: "Ascend NPU"
summary: "A hardware topic hub for Ascend NPU architecture, AI Core execution, memory hierarchy, and scale-up interconnects."
layout: default
redirect_from:
  - /hardware/ascend/
confidence: medium
sources:
  - raw/hardware/accelerators/ascend/ascend-950-npu-architecture-white-paper--paper.pdf
updated: 2026-09-16
---

# Ascend NPU

This topic collects hardware-level readings of Ascend NPUs and the software or kernel concepts needed to understand their execution model.

- [Ascend 950 NPU Architecture: Compute, Memory, and Unified Bus](ascend-950/index.md) — Third-generation DaVinci Cube/Vector compute, HiF8 and MX formats, chiplet UMA memory, NDDMA, STARS2.0, Unified Bus/URMA/CCU, DVPP, and 8,192-card supernodes.
- [Triton Ascend Operator Mechanisms: Vector, Cube, and CV Fusion](../../../frameworks/compilers-kernels/triton-ascend/operator-mechanisms.md) — Practical mapping from Triton tiles to Ascend AI Core buffers, movement engines, instruction queues, Vector/Cube execution, and CV fusion.
- [vLLM-Ascend Architecture](../../../frameworks/serving/vllm-ascend/architecture.md) — Software integration path from vLLM's model-serving stack to Ascend platforms, attention backends, HCCL, and device execution.
