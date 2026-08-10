---
kind: repository-analysis
repository_id: github:tile-ai/tilelang-ascend@34a048c19bd762381db0d0f2d5acfdf3527c459f
commit: 34a048c19bd762381db0d0f2d5acfdf3527c459f
source_record: raw/frameworks/tilelang-ascend-codebase--github-34a048c19bd7.md
generated: 2026-08-10
---

# TileLang-Ascend Integration Evidence

## Reader Contract

- **Audience:** TileLang developers who need to understand the Ascend adapter boundary.
- **Question:** Which parts of TileLang-Ascend are shared with TileLang, which parts are Ascend-specific, and how far does upstream integration currently go?
- **Mental model:** TileLang-Ascend is a source-level TileLang variant that keeps the tile DSL shape but owns Ascend lowering, source code generation, and Bisheng/CANN execution.
- **Offline/load/runtime split:** Python constructs TIR; Ascend passes lower it; TVM FFI returns generated source; Bisheng compiles and the Cython adapter loads and calls the library.
- **Limits:** Static reading only. No CANN compilation, NPU execution, numerical validation, or performance measurement was performed.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/tilelang-ascend/index.md` | ascend-overview | `README.md` | TileLang-Ascend overview | 12 | |
| `docs/frameworks/tilelang-ascend/index.md` | ascend-build-switch | `setup.py` | `USE_ASCEND` | 39 | |
| `docs/frameworks/tilelang-ascend/index.md` | ascend-native-sources | `CMakeLists.txt` | `USE_ASCEND` source list | 130 | |
| `docs/frameworks/tilelang-ascend/index.md` | ascend-language-facade | `tilelang/language/__init__.py` | Ascend language exports | 83 | |
| `docs/frameworks/tilelang-ascend/index.md` | ascend-lowering | `tilelang/engine/phase.py` | `LowerAndLegalize` | 49 | |
| `docs/frameworks/tilelang-ascend/index.md` | ascend-optimization | `tilelang/engine/phase.py` | `OptimizeForTarget` | 96 | |
| `docs/frameworks/tilelang-ascend/index.md` | target-selection | `tilelang/utils/target.py` | `determine_target` | 63 | |
| `docs/frameworks/tilelang-ascend/index.md` | codegen-dispatch | `tilelang/engine/lower.py` | `device_codegen` | 159 | |
| `docs/frameworks/tilelang-ascend/index.md` | ascend-lower-entry | `tilelang/engine/lower.py` | `lower` | 193 | |
| `docs/frameworks/tilelang-ascend/index.md` | ascend-codegen-registration | `src/target/rt_mod_ascend.cc` | `target.build.tilelang_ascend` | 31 | |
| `docs/frameworks/tilelang-ascend/index.md` | pto-codegen-registration | `src/target/rt_mod_ascend_pto.cc` | `target.build.tilelang_ascend_pto` | 31 | |
| `docs/frameworks/tilelang-ascend/index.md` | ascend-kernel-prefix | `src/target/codegen_ascend.cc` | `CodeGenTileLangAscend::PrintFuncPrefix` | 100 | |
| `docs/frameworks/tilelang-ascend/index.md` | ascend-host-dispatch | `src/target/codegen_ascend.cc` | `CodeGenTileLangAscend::PrintHostFunc` | 1124 | |
| `docs/frameworks/tilelang-ascend/index.md` | pto-host-dispatch | `src/target/codegen_ascend_pto.cc` | `CodeGenTileLangAscendPto::PrintHostFunc` | 3979 | |
| `docs/frameworks/tilelang-ascend/index.md` | bisheng-build | `tilelang/jit/adapter/libgen.py` | `LibraryGenerator.compile_lib` | 142 | |
| `docs/frameworks/tilelang-ascend/index.md` | npu-wrapper | `tilelang/jit/adapter/wrapper.py` | `TLWrapper.wrap` | 648 | |
| `docs/frameworks/tilelang-ascend/index.md` | cython-adapter | `tilelang/jit/adapter/cython/adapter.py` | `CythonKernelAdapter` | 176 | |
| `docs/frameworks/tilelang-ascend/index.md` | runtime-call | `tilelang/jit/adapter/cython/adapter.py` | `lib.call` | 449 | |
| `docs/frameworks/tilelang-ascend/index.md` | upstream-backend-contract | `tilelang/backend/module.py` | `BackendModule` | 28 | |
| `docs/frameworks/tilelang-ascend/index.md` | upstream-backend-registration | `tilelang/backend/module.py` | `register_backend` | 238 | |
| `docs/frameworks/tilelang-ascend/index.md` | upstream-context-resolution | `tilelang/backend/module.py` | `create_backend_context` | 310 | |
| `docs/frameworks/tilelang-ascend/index.md` | upstream-lower-context | `tilelang/engine/lower.py` | `lower` | 220 | |
| `docs/frameworks/tilelang-ascend/index.md` | upstream-ecosystem-status | `README.md` | Huawei Ascend support table | 123 | |
| `docs/frameworks/tilelang-ascend/index.md` | upstream-trace-awareness | `tilelang/tools/lower_trace/core.py` | Ascend codegen FFI names | 143 | |

## Runtime Flow Evidence

1. **Frontend** — `ascend-language-facade` exports Ascend operations through the TileLang language package.
2. **Lowering** — `ascend-lowering` and `ascend-optimization` apply buffer, layout, pipeline, memory, vector, and synchronization passes.
3. **Target selection** — `target-selection`, `codegen-dispatch`, and `ascend-lower-entry` create the internal target model and call the Ascend FFI.
4. **Source generation** — `ascend-codegen-registration`, `pto-codegen-registration`, `ascend-kernel-prefix`, `ascend-host-dispatch`, and `pto-host-dispatch` produce source with kernel and host-call entry points.
5. **Build and launch** — `bisheng-build`, `npu-wrapper`, `cython-adapter`, and `runtime-call` connect generated source to CANN and NPU tensors.
6. **Upstream comparison** — `upstream-backend-contract`, `upstream-backend-registration`, `upstream-context-resolution`, `upstream-lower-context`, `upstream-ecosystem-status`, and `upstream-trace-awareness` define the current integration boundary.

## Reading Scope

Static inspection of the `ascendc_pto` checkout at commit `34a048c19bd762381db0d0f2d5acfdf3527c459f`, compared with the pinned upstream TileLang checkout at commit `5a9b2a54b9a04831886d9b4dfd7a9ad758cd7ebd`. The comparison covers the Python language facade, target and lowering flow, native Ascend C/PTO codegen registration, Bisheng/CANN source compilation, Cython launch, upstream backend registration, package boundaries, and the separate `npuir` branch claim in the README.

## Limitations

- The `npuir` branch was not inspected; this page describes the checked-out `ascendc_pto` branch.
- No CANN compiler, Ascend NPU, generated library, or runtime launch was available for execution validation.
- The upstream refresh helper could not contact GitHub from the sandbox; the existing clean checkout and pinned evidence record were reused.
