---
kind: repository-analysis
repository_id: github:deepseek-ai/DeepGEMM@559d79fb6994a58b8a15b4b93bf13ccc16edf247
commit: 559d79fb6994a58b8a15b4b93bf13ccc16edf247
source_record: raw/hardware/deepgemm-codebase--github-559d79fb6994.md
generated: 2026-08-26
---

# DeepGEMM Codebase (MegaMoE) Important Files

## Evidence Map

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

## Reproduction Commands

Record exact read-only search or counting commands here when the docs make
quantitative codebase claims.

## Required Code Evidence

The consuming page is `docs/hardware/deepgemm/index.md`.

| docs page | finding | file | symbol | start | end |
|---|---|---|---|---:|---:|
| docs/hardware/deepgemm/index.md | mega-contract | README.md | Mega MoE interface | 114 | - |
| docs/hardware/deepgemm/index.md | python-buffer | deep_gemm/mega/__init__.py | SymmBuffer | 18 | - |
| docs/hardware/deepgemm/index.md | python-weight-layout | deep_gemm/mega/__init__.py | transform_weights_for_mega_moe | 131 | - |
| docs/hardware/deepgemm/index.md | python-launch | deep_gemm/mega/__init__.py | fp8_fp4_mega_moe | 153 | - |
| docs/hardware/deepgemm/index.md | test-setup | tests/test_mega_moe.py | test | 75 | - |
| docs/hardware/deepgemm/index.md | test-fused-call | tests/test_mega_moe.py | run_fused | 181 | - |
| docs/hardware/deepgemm/index.md | test-reference | tests/test_mega_moe.py | run_baseline and correctness | 223 | - |
| docs/hardware/deepgemm/index.md | test-benchmark-boundary | tests/test_mega_moe.py | benchmark and performance summary | 339 | - |
| docs/hardware/deepgemm/index.md | buffer-sizing | csrc/apis/mega.hpp | get_symm_buffer_size_for_mega_moe | 37 | - |
| docs/hardware/deepgemm/index.md | host-validation-dispatch | csrc/apis/mega.hpp | fp8_fp4_mega_moe | 157 | - |
| docs/hardware/deepgemm/index.md | shape-heuristics | csrc/jit_kernels/heuristics/mega_moe.hpp | get_block_config_for_mega_moe | 76 | - |
| docs/hardware/deepgemm/index.md | pipeline-heuristics | csrc/jit_kernels/heuristics/mega_moe.hpp | get_pipeline_config_for_mega_moe | 115 | - |
| docs/hardware/deepgemm/index.md | jit-launch | csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp | sm100_fp8_fp4_mega_moe | 131 | - |
| docs/hardware/deepgemm/index.md | workspace-layout | deep_gemm/include/deep_gemm/layout/mega_moe.cuh | Workspace | 46 | - |
| docs/hardware/deepgemm/index.md | buffer-layout | deep_gemm/include/deep_gemm/layout/mega_moe.cuh | MegaMoEBuffer | 331 | - |
| docs/hardware/deepgemm/index.md | receive-counts | deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh | fetch_expert_recv_count | 249 | - |
| docs/hardware/deepgemm/index.md | routed-task-schedule | deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh | get_next_task | 316 | - |
| docs/hardware/deepgemm/index.md | shared-task-schedule | deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh | mainloop | 383 | - |
| docs/hardware/deepgemm/index.md | nvlink-barrier | deep_gemm/include/deep_gemm/comm/barrier.cuh | nvlink_barrier | 40 | - |
| docs/hardware/deepgemm/index.md | kernel-entry | deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh | sm100_fp8_fp4_mega_moe_impl | 55 | - |
| docs/hardware/deepgemm/index.md | dispatch | deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh | dispatch warps | 333 | - |
| docs/hardware/deepgemm/index.md | remote-pull | deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh | pull token data and SF | 414 | - |
| docs/hardware/deepgemm/index.md | workspace-cleanup | deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh | workspace cleanup | 601 | - |
| docs/hardware/deepgemm/index.md | tensor-core-compute | deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh | TMA and MMA stages | 675 | - |
| docs/hardware/deepgemm/index.md | activation-and-requantization | deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh | GEMM epilogue | 962 | - |
| docs/hardware/deepgemm/index.md | remote-combine-write | deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh | BF16 L2 epilogue | 1205 | - |
| docs/hardware/deepgemm/index.md | topk-reduction | deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh | combine | 1313 | - |
