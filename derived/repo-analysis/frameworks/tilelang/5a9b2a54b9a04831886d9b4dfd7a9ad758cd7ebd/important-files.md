---
kind: repository-analysis
repository_id: github:tile-ai/tilelang@5a9b2a54b9a04831886d9b4dfd7a9ad758cd7ebd
commit: 5a9b2a54b9a04831886d9b4dfd7a9ad758cd7ebd
source_record: raw/frameworks/tilelang-codebase--github-5a9b2a54b9a0.md
generated: 2026-08-10
---

# TileLang Codebase Important Files

## Reader Contract

- **Audience:** Python developers who know basic GPU concepts and want to learn
  how TileLang is designed before changing its compiler or writing advanced
  kernels.
- **Question:** How does a Python tile program become a target-specific callable
  kernel, and where should a learner read or extend each stage?
- **Mental model:** TileLang preserves tile-level intent in TIRX long enough for
  a backend pipeline to choose layouts, pipelines, hardware instructions, and an
  execution adapter together.
- **Offline/load/runtime split:** installing TileLang loads its TVM-based native
  extension; the first specialized call elaborates and compiles a kernel; later
  calls reuse the specialization cache and invoke the adapter.
- **Limits:** static source reading only. No backend was compiled or executed,
  and generated code, numerical correctness, hardware coverage, and performance
  were not verified in this workspace.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/tilelang/index.md` | platform-support | `README.md` | Platform and Backend Support | 111 | 127 |
| `docs/frameworks/tilelang/index.md` | quickstart-program | `examples/quickstart.py` | `matmul` | 8 | 48 |
| `docs/frameworks/tilelang/index.md` | jit-surface | `tilelang/jit/__init__.py` | `jit` | 574 | 636 |
| `docs/frameworks/tilelang/index.md` | eager-elaboration | `tilelang/language/eager/builder.py` | `JITFunc._build_tir_template` | 1547 | 1560 |
| `docs/frameworks/tilelang/index.md` | specialization-cache | `tilelang/jit/__init__.py` | `JITImpl.__call__` | 505 | 550 |
| `docs/frameworks/tilelang/index.md` | target-neutral-launch | `tilelang/language/kernel.py` | `Kernel` | 277 | 288 |
| `docs/frameworks/tilelang/index.md` | tile-copy-ir | `tilelang/language/copy_op.py` | `copy` | 110 | 140 |
| `docs/frameworks/tilelang/index.md` | tile-gemm-ir | `tilelang/language/gemm_op.py` | `gemm` | 145 | 194 |
| `docs/frameworks/tilelang/index.md` | backend-contract | `tilelang/backend/module.py` | `BackendModule` | 28 | 99 |
| `docs/frameworks/tilelang/index.md` | backend-resolution | `tilelang/backend/module.py` | `create_backend_context` | 310 | 335 |
| `docs/frameworks/tilelang/index.md` | lower-and-split | `tilelang/engine/lower.py` | `lower_to_host_device_ir` | 103 | 131 |
| `docs/frameworks/tilelang/index.md` | artifact-materialization | `tilelang/engine/lower.py` | `_lower_with_context_impl` | 161 | 200 |
| `docs/frameworks/tilelang/index.md` | cuda-high-level-lowering | `tilelang/cuda/pipeline.py` | `CUDAPassPipelineBodyPrologue` | 68 | 149 |
| `docs/frameworks/tilelang/index.md` | cuda-low-level-lowering | `tilelang/cuda/pipeline.py` | `CUDAPassPipelineBody` | 152 | 265 |
| `docs/frameworks/tilelang/index.md` | tile-op-dispatch | `src/transform/lower_tile_op.cc` | `LowerTileOpPass::VisitStmt_(EvaluateNode)` | 1065 | 1145 |
| `docs/frameworks/tilelang/index.md` | adapter-selection | `tilelang/jit/kernel.py` | `JITKernel._compile_and_create_adapter` | 274 | 379 |
| `docs/frameworks/tilelang/index.md` | cuda-registration | `tilelang/cuda/backend.py` | `BACKEND` | 123 | 143 |

## Runtime Flow Evidence

1. **Entry and specialization** — `jit-surface`, `specialization-cache`.
2. **Python elaboration to TIRX** — `eager-elaboration`,
   `target-neutral-launch`, `tile-copy-ir`, `tile-gemm-ir`.
3. **Backend resolution and lowering** — `backend-contract`,
   `backend-resolution`, `lower-and-split`, `cuda-high-level-lowering`,
   `tile-op-dispatch`, `cuda-low-level-lowering`.
4. **Code and runtime materialization** — `artifact-materialization`,
   `cuda-registration`, `adapter-selection`.
5. **Invocation and reuse** — `specialization-cache`, `adapter-selection`.

## Important Entry Files

- `examples/quickstart.py` — begin with the user-visible tiled GEMM.
- `tilelang/jit/__init__.py` — follow decoration, specialization, and caching.
- `tilelang/language/eager/builder.py` — see how Python becomes TIRX.
- `tilelang/language/kernel.py`, `copy_op.py`, and `gemm_op.py` — inspect the
  target-neutral launch and tile-operation surface.
- `tilelang/backend/module.py` — understand the extension contract shared by
  CUDA, ROCm, Metal, CPU, WebGPU, and CuTe DSL backends.
- `tilelang/engine/lower.py` — see the common compiler orchestration.
- `tilelang/cuda/pipeline.py` and `src/transform/lower_tile_op.cc` — study the
  deepest representative lowering path.
- `tilelang/jit/kernel.py` — finish at generated artifacts and runtime adapters.

## Reproduction Commands

No quantitative codebase claims are used in the consuming page.

## Link Completion

- [x] Every Required Code Evidence row has a matching planned code link.
- [x] Every runtime-flow step names at least one declared finding.
- [x] Major implementation symbols are included in the evidence contract.
- [x] `./scripts/lint-docs.sh` passes for the completed consuming page.
