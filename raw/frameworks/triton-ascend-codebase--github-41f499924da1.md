---
kind: repository-source
provider: github
clone_url: https://github.com/triton-lang/triton-ascend.git
repository_url: https://github.com/triton-lang/triton-ascend
local_checkout: external-repos/triton-ascend/
commit: 41f499924da1d58955196c946895597e992127f0
ref: main
inspected: 2026-07-28
checkout_state: clean
---

# Repo Codebase Source Record

## Reading Scope

Full-repository architecture overview: Ascend NPU backend for OpenAI Triton, covering the compilation pipeline, backend registration, Ascend-specific MLIR passes, runtime driver, and CANN extension ops.

## Important Entry Files

- `third_party/ascend/backend/compiler.py` — AscendBackend class defining the full compilation pipeline (TTIR → Linalg → Ascend NPU binary).
- `third_party/ascend/backend/driver.py` — NPUDriver and NPULauncher; loads kernels via ACL runtime.
- `third_party/ascend/backend/__init__.py` — Backend registration, monkey-patches CodeGenerator for Ascend target.
- `third_party/ascend/backend/npu_utils.cpp` — C++ bridge: kernel binary registration and ACL runtime integration.
- `third_party/ascend/language/cann/extension/` — Ascend-specific DSL extension ops (fixpipe, sync_block, conv1d, etc.).
- `python/triton/backends/__init__.py` — Backend discovery mechanism.
- `python/triton/backends/compiler.py` — Abstract BaseBackend and GPUTarget.
- `python/triton/backends/driver.py` — Abstract DriverBase and GPUDriver.
- `third_party/ascend/backend/runtime/autotuner.py` — Ascend autotuner with CV autotune, UB tuning, DSL analysis.
- `third_party/ascend/lib/` — MLIR pass implementations (TritonToHIVM, TritonToLinalg, TritonToLLVM, DynamicCVPipeline, etc.).

## Limitations

- Static reading only; runtime behavior was not executed.
- AscendNPU-IR submodule was not available for local inspection.
- Version 3.6.0-dev (version.txt); inspection may not match the latest pip release (3.2.1).
