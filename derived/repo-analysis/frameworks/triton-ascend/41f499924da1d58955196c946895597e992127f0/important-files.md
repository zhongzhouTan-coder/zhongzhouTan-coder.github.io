---
kind: repository-analysis
repository_id: github:triton-lang/triton-ascend@41f499924da1d58955196c946895597e992127f0
commit: 41f499924da1d58955196c946895597e992127f0
source_record: raw/frameworks/triton-ascend-codebase--github-41f499924da1.md
generated: 2026-07-28
---

# Triton Ascend Reading Notes

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/triton-ascend/index.md` | backend-name | `third_party/ascend/backend/name.conf` | `ascend` backend name | 1 | — |
| `docs/frameworks/triton-ascend/index.md` | backend-discovery | `python/triton/backends/__init__.py` | `_discover_backends` | 38 | — |
| `docs/frameworks/triton-ascend/index.md` | abstract-backend | `python/triton/backends/compiler.py` | `BaseBackend` | 23 | — |
| `docs/frameworks/triton-ascend/index.md` | abstract-driver | `python/triton/backends/driver.py` | `DriverBase` | 11 | — |
| `docs/frameworks/triton-ascend/index.md` | monkey-patch | `third_party/ascend/backend/__init__.py` | `_apply_ascend_patch` | 27 | — |
| `docs/frameworks/triton-ascend/index.md` | compilation | `third_party/ascend/backend/compiler.py` | `AscendBackend.add_stages` | 1265 | — |
| `docs/frameworks/triton-ascend/index.md` | launch | `third_party/ascend/backend/driver.py` | `NPULauncher` | 150 | — |
| `docs/frameworks/triton-ascend/index.md` | cpp-bridge | `third_party/ascend/backend/npu_utils.cpp` | `registerKernel` | 55 | — |
| `docs/frameworks/triton-ascend/index.md` | autotune | `third_party/ascend/backend/runtime/autotuner.py` | `AutoTilingTuner` | 205 | — |
| `docs/frameworks/triton-ascend/index.md` | extension-ops | `third_party/ascend/language/cann/extension/__init__.py` | `cann` extension module | 1 | — |

## Runtime Flow Evidence

1. Backend discovery — `backend-name`, `backend-discovery`.
2. Abstract contract — `abstract-backend`, `abstract-driver`.
3. Registration and monkey-patching — `monkey-patch`.
4. Compilation — `compilation`.
5. Kernel launch and C++ bridge — `launch`, `cpp-bridge`.
6. Tuning and DSL extension — `autotune`, `extension-ops`.

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
