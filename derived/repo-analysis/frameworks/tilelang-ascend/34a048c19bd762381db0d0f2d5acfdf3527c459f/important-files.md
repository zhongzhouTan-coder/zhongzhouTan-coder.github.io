---
kind: repository-analysis
repository_id: github:tile-ai/tilelang-ascend@34a048c19bd762381db0d0f2d5acfdf3527c459f
commit: 34a048c19bd762381db0d0f2d5acfdf3527c459f
source_record: raw/frameworks/tilelang-ascend-codebase--github-34a048c19bd7.md
generated: 2026-08-10
---

# TileLang-Ascend Codebase Important Files

## Evidence Map

- `README.md` — Describes the Ascend variant, Ascend C/PTO and NPU IR routes, installation model, and tested devices
- `setup.py` — Selects USE_ASCEND, configures the CMake build, and packages Ascend dependencies
- `CMakeLists.txt` — Conditionally compiles Ascend C and PTO codegen/runtime sources
- `tilelang/language/__init__.py` — Exports Ascend language operations and tile primitives through the TileLang facade
- `tilelang/engine/phase.py` — Defines the Ascend-specific lowering and optimization passes
- `tilelang/engine/lower.py` — Dispatches target.build.tilelang_ascend and target.build.tilelang_ascend_pto
- `tilelang/utils/target.py` — Detects NPU availability and maps Ascend target aliases
- `tilelang/jit/adapter/libgen.py` — Compiles generated source with Bisheng and CANN libraries
- `tilelang/jit/adapter/cython/adapter.py` — Loads the generated NPU library and invokes the Cython runtime wrapper
- `src/target/codegen_ascend.cc` — Generates Ascend C kernel source from TIR
- `src/target/codegen_ascend_pto.cc` — Generates PTO kernel source from TIR
- `src/target/rt_mod_ascend.cc` — Registers the Ascend C TVM codegen entry point
- `src/target/rt_mod_ascend_pto.cc` — Registers the PTO TVM codegen entry point

## Reproduction Commands

Record exact read-only search or counting commands here when the docs make
quantitative codebase claims.
