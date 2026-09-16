---
kind: repository-source
provider: github
clone_url: git@github.com:deepseek-ai/DeepGEMM.git
repository_url: https://github.com/deepseek-ai/DeepGEMM
local_checkout: external-repos/DeepGEMM/
commit: 559d79fb6994a58b8a15b4b93bf13ccc16edf247
ref: main
inspected: 2026-08-26
checkout_state: clean
---

# DeepGEMM Codebase (MegaMoE) Source Record

## Reading Scope

- FP8xFP4 MegaMoE inference path: symmetric-memory setup, weight transformation, JIT dispatch, ring scheduling, fused communication and expert compute
- Public Python and C++ interfaces, shape heuristics, buffer layout, correctness reference, and benchmark/profiling boundaries

## Important Entry Files

- `README.md` — Public MegaMoE contract, grouped-GEMM context, supported GPU/toolchain constraints, and usage example
- `deep_gemm/mega/__init__.py` — Python-facing symmetric-buffer allocation, weight transformation, and MegaMoE launch wrappers
- `tests/test_mega_moe.py` — Multi-process setup, reference MoE computation, correctness checks, and benchmark invocation
- `csrc/apis/mega.hpp` — C++/PyBind validation, symmetric-buffer sizing, architecture dispatch, and debug handling
- `csrc/jit_kernels/heuristics/mega_moe.hpp` — Shape-dependent block, pipeline, cluster, and shared-memory configuration
- `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` — Runtime code generation, JIT compilation, argument packing, and kernel launch
- `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` — Symmetric-memory buffer slices, per-rank views, and inter-rank workspace layout
- `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh` — Ring schedule, communication/compute tile ownership, and barrier coordination
- `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` — Fused FP8xFP4 expert computation, SwiGLU, NVLink dispatch/combine overlap, and output writes
- `deep_gemm/include/deep_gemm/comm/barrier.cuh` — Device-side symmetric-memory barrier primitives used by the communication schedule

## Limitations

- Static code reading only; no NVIDIA GPU, multi-process launch, or runtime benchmark was available in this workspace.
- The inspected checkout is the clean main branch at the pinned public-release commit; later remote branches are outside this evidence snapshot.
