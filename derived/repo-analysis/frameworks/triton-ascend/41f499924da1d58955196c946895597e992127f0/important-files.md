---
kind: repository-analysis
repository_id: github:triton-lang/triton-ascend@41f499924da1d58955196c946895597e992127f0
commit: 41f499924da1d58955196c946895597e992127f0
source_record: raw/frameworks/triton-ascend-codebase--github-41f499924da1.md
generated: 2026-07-28
---

# Triton Ascend Reading Notes

## Evidence Map

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
