---
kind: repository-analysis
repository_id: github:deepseek-ai/DeepGEMM@559d79fb6994a58b8a15b4b93bf13ccc16edf247
commit: 559d79fb6994a58b8a15b4b93bf13ccc16edf247
source_record: raw/hardware/deepgemm-codebase--github-559d79fb6994.md
generated: 2026-08-26
purpose: MegaMoE reader contract, runtime-flow evidence, and verification boundary
---

# DeepGEMM MegaMoE Runtime Flow

## Reader Contract

- Audience: a reader who knows matrix multiplication and basic mixture-of-experts (MoE) routing, but not CUDA persistent kernels or symmetric memory.
- Question: how does one routed token move through DeepGEMM's FP8xFP4 MegaMoE path, and where does communication overlap expert computation?
- Code-free mental model: every rank launches the same long-lived kernel; dispatch warps move routed tokens into a bounded local ring, tensor-core warps run the two expert GEMMs, and epilogue warps send weighted results back for top-k reduction.
- Offline: transform weights and allocate one rank-invariant symmetric workspace.
- At launch: validate layouts, choose shape-specific block and pipeline parameters, build TMA descriptors, and JIT-specialize the SM100 kernel.
- At runtime: count routes, publish remote metadata, pull token/scales, schedule L1/L2 work, apply SwiGLU, write remote combine slots, reduce top-k contributions, and recycle counters.
- Limits: this snapshot is a static reading of a clean main-branch checkout; no GPU, multi-process execution, correctness run, or benchmark was available.

## Rich-Content Plan

| Reader question | Representation | Evidence | Teaching job |
|---|---|---|---|
| What is the end-to-end ownership and overlap pattern? | Mermaid sequence diagram | `sm100_fp8_fp4_mega_moe_impl` and `nvlink_barrier` | Show the call/return boundary and the parallel dispatch, compute, and combine roles. |
| What state changes for one token? | Numbered worked trace | Dispatch, scheduler, epilogue, and combine code | Keep logical token identity, expert-local position, ring position, and final output connected. |
| Why is the workspace larger than the current batch? | Small mapping table plus prose | `get_symm_buffer_size_for_mega_moe` and `MegaMoEBuffer` | Explain worst-case ring capacity and reusable storage without inventing a measured allocation. |

## Required Code Evidence

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

## Verification Boundary

The repository test constructs random routed inputs, runs the fused call, and—when optional legacy dependencies load—compares it with a deliberately separated baseline consisting of expert-parallel dispatch, grouped GEMM, SwiGLU, grouped GEMM, and combine. The test compares cumulative expert counts exactly; it compares outputs exactly without shared experts and with a `calc_diff` threshold when shared experts are enabled. Those are executable checks defined by the source, not results observed during this ingest.

The benchmark path invokes `bench_kineto`, estimates FLOPs, HBM bytes, NVLink bytes, and a serial reduction time, then prints a throughput summary. The formulas are useful for understanding the authors' accounting boundary; they are not a measured result in this workspace.

## Reproduction Commands

```bash
rg -n "mega_moe|MegaMoE|fp8_fp4_mega_moe" external-repos/DeepGEMM
nl -ba external-repos/DeepGEMM/tests/test_mega_moe.py | sed -n '223,397p'
nl -ba external-repos/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh | sed -n '333,1451p'
```
