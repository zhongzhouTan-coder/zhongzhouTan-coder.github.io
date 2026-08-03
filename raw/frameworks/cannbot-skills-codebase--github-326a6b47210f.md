---
kind: repository-source
repository_url: https://github.com/cann/cannbot-skills
local_checkout: external-repos/cannbot-skills/
commit: 326a6b47210fc31a9c225f1643778a4cc733e57c
ref: main
inspected: 2026-08-03
checkout_state: clean
---

# Repo Codebase Source Record

## Reading Scope

Targeted reading of the seven Triton-domain skills and the triton-op-generator plugin within the CANNBot Skills repository. Focus: understanding the AI-assisted Triton Ascend kernel development workflow — task extraction, algorithm sketch design, code generation, verification, precision debugging, latency optimization, and simulator-driven bottleneck diagnosis.

## Important Entry Files

- `ops/triton-task-extractor/SKILL.md` — PyTorch → standardized task file extraction, single/multi-case mode support.
- `ops/triton-op-designer/SKILL.md` — UnifiedSketch DSL algorithm sketch design, Layer 1 constraint compliance.
- `ops/triton-op-coding/SKILL.md` — Pure Triton Ascend kernel generation, enforced "no PyTorch degradation" constraint.
- `ops/triton-op-verifier/SKILL.md` — Five-category precision decision matrix, benchmark collection.
- `ops/triton-precision-debug/SKILL.md` — Five-stage ULP isolation methodology.
- `ops/triton-latency-optimizer/SKILL.md` — 25 ordered optimization points with reference documentation.
- `ops/triton-simulator-optimizer/SKILL.md` — msprof op simulator bottleneck diagnosis.
- `plugins-official/triton-op-generator/AGENTS.md` — Six-phase orchestration pipeline (Phase 0-6).
- `AGENTS.md` — Project architecture overview: Plugin→Agent→Skill three-layer design.
- `README.md` — Project overview, install instructions, development paths.
- `docs/architecture-design.md` — Detailed architecture with logical views per development path.
- `docs/STANDARDS.md` — Skill classification, naming conventions, structure guidelines.

## Limitations

- Static reading only; runtime behavior was not executed against a live NPU.
- The .claude-plugin/ and .gitcode/ directories contain tooling glue not inspected in detail.
- Community plugins (`plugins-community/`) and non-Triton skills (Ascend C, TileLang, PyPTO, Catlass) were not read.
- Test framework and evaluation data under `tests/` were not inspected.
