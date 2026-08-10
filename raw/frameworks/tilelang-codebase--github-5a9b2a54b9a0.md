---
kind: repository-source
provider: github
clone_url: git@github.com:tile-ai/tilelang.git
repository_url: https://github.com/tile-ai/tilelang
local_checkout: external-repos/tilelang/
commit: 5a9b2a54b9a04831886d9b4dfd7a9ad758cd7ebd
ref: main
inspected: 2026-08-10
checkout_state: clean
---

# TileLang Codebase Source Record

## Reading Scope

- TileLang DSL frontend, eager/lazy JIT elaboration, target-neutral tile IR, backend registry, CUDA lowering pipeline, device code generation, execution adapters, and representative GEMM learning path

## Important Entry Files

- `README.md` — Project goals, backend support, and eager GEMM programming model
- `examples/quickstart.py` — Representative end-to-end tiled GEMM using Kernel, memory scopes, Pipelined, copy, gemm, and JIT compilation
- `tilelang/jit/__init__.py` — Public jit decorator, eager/lazy mode inference, specialization cache, and compile entry
- `tilelang/language/eager/builder.py` — Eager Python builder that emits TIRX and preserves source spans
- `tilelang/language/kernel.py` — Target-neutral Kernel launch frame and block/thread binding construction
- `tilelang/language/gemm_op.py` — High-level tile GEMM operation emitted into the IR
- `tilelang/backend/module.py` — Backend component manifest, registry, target resolution, and per-compilation context
- `tilelang/engine/lower.py` — Semantic checks, backend lowering, host/device split, and code-generation handoff
- `tilelang/cuda/pipeline.py` — CUDA pass ordering from launch materialization through layout inference, tile-op lowering, synchronization, and packed API
- `tilelang/cuda/backend.py` — CUDA backend registration, codegen callbacks, and binary compilation cache
- `tilelang/jit/kernel.py` — Compilation orchestration and runtime adapter selection
- `src/transform/lower_tile_op.cc` — C++ lowering of target-neutral tile operations after layout inference

## Limitations

- Static source reading only; GPU/CPU compilation, generated code, numerical correctness, and performance were not executed in this workspace.
