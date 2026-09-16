---
kind: repository-source
provider: github
clone_url: git@github.com:tile-ai/tilelang-ascend.git
repository_url: https://github.com/tile-ai/tilelang-ascend
local_checkout: external-repos/tilelang-ascend/
commit: 34a048c19bd762381db0d0f2d5acfdf3527c459f
ref: ascendc_pto
inspected: 2026-08-10
checkout_state: clean
---

# TileLang-Ascend Codebase Source Record

## Reading Scope

- Ascend C and PTO frontend, lowering, native code generation, and CANN execution path
- Integration boundary with upstream TileLang backend registry and package distribution
- Build dependencies, pinned TVM submodule, target selection, and runtime launch

## Important Entry Files

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

## Limitations

- Static code reading only; no CANN compiler or Ascend NPU runtime was available for execution validation.
- The inspected checkout is the ascendc_pto branch; the separate npuir branch was not analyzed.
