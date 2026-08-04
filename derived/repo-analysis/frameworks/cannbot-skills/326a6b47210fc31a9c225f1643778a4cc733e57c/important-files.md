---
kind: repository-analysis
repository_id: gitcode:cann/cannbot-skills@326a6b47210fc31a9c225f1643778a4cc733e57c
commit: 326a6b47210fc31a9c225f1643778a4cc733e57c
source_record: raw/frameworks/cannbot-skills-codebase--gitcode-326a6b47210f.md
generated: 2026-08-03
---

# CANNBot Skills: Important Files

Repository: [cann/cannbot-skills](https://gitcode.com/cann/cannbot-skills)
Revision: `326a6b47210fc31a9c225f1643778a4cc733e57c`

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/triton-ascend/cannbot-skills-workflow.md` | generator-pipeline | `plugins-official/triton-op-generator/AGENTS.md` | `triton-op-generator` workflow | 35 | — |
| `docs/frameworks/triton-ascend/cannbot-skills-workflow.md` | task-extractor-skill | `ops/triton-task-extractor/SKILL.md` | task-extractor skill definition | 14 | — |
| `docs/frameworks/triton-ascend/cannbot-skills-workflow.md` | task-validation | `ops/triton-task-extractor/scripts/validate_task.py` | `main` | 442 | — |
| `docs/frameworks/triton-ascend/cannbot-skills-workflow.md` | sketch-spec | `ops/triton-op-designer/references/sketch-design.md` | UnifiedSketch DSL spec | 1 | — |
| `docs/frameworks/triton-ascend/cannbot-skills-workflow.md` | npu-arch-reference | `ops/npu-arch/references/npu-arch-guide-triton.md` | npu-arch guide | 1 | — |
| `docs/frameworks/triton-ascend/cannbot-skills-workflow.md` | swizzle-case | `ops/triton-op-designer/references/cases/matmul-swizzle2d.md` | matmul-swizzle2d case | 1 | — |
| `docs/frameworks/triton-ascend/cannbot-skills-workflow.md` | benchmark | `ops/triton-op-verifier/scripts/benchmark.py` | `benchmark_implementations` | 774 | — |

## Runtime Flow Evidence

1. Task extraction and validation — `task-extractor-skill`, `task-validation`.
2. Design and reference — `sketch-spec`, `npu-arch-reference`, `swizzle-case`.
3. Code generation pipeline — `generator-pipeline`.
4. Verification and benchmark — `benchmark`.

## Triton-domain skills

| Skill | Path | Role |
|-------|------|------|
| triton-task-extractor | `ops/triton-task-extractor/SKILL.md` | PyTorch → standardized task file |
| triton-op-designer | `ops/triton-op-designer/SKILL.md` | Algorithm sketch in UnifiedSketch DSL |
| triton-op-coding | `ops/triton-op-coding/SKILL.md` | Generate @triton.jit kernel code |
| triton-op-verifier | `ops/triton-op-verifier/SKILL.md` | Precision verification + benchmark |
| triton-precision-debug | `ops/triton-precision-debug/SKILL.md` | Five-stage ULP isolation |
| triton-latency-optimizer | `ops/triton-latency-optimizer/SKILL.md` | 25 ordered optimization points |
| triton-simulator-optimizer | `ops/triton-simulator-optimizer/SKILL.md` | msprof bottleneck diagnosis |

## Plugin

| Plugin | Path | Role |
|--------|------|------|
| triton-op-generator | `plugins-official/triton-op-generator/AGENTS.md` | Six-phase orchestration pipeline |

## Project-level docs

| Doc | Path | Description |
|-----|------|-------------|
| AGENTS.md | `AGENTS.md` | Architecture overview and development guide |
| README.md | `README.md` | Project overview and quick start |
| Architecture design | `docs/architecture-design.md` | Logical views per development path |
| Standards | `docs/STANDARDS.md` | Naming, structure, classification |
| Feature list | `docs/feature-list.md` | Full skill/plugin inventory |
| Skills usage | `docs/skills-usage.md` | Example prompts per skill |
