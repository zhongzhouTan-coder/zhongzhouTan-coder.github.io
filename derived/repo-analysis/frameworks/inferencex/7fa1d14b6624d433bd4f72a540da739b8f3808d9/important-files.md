---
kind: repository-analysis
repository_id: github:SemiAnalysisAI/InferenceX@7fa1d14b6624d433bd4f72a540da739b8f3808d9
commit: 7fa1d14b6624d433bd4f72a540da739b8f3808d9
source_record: raw/frameworks/inferencex-codebase--github-7fa1d14b6624.md
generated: 2026-09-09
---

# InferenceX Important Files

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/inferencex/index.md` | readme-overview | `README.md` | `README` overview | 1 | — |
| `docs/frameworks/inferencex/index.md` | architecture-flow | `docs/architecture.md` | Pipeline architecture | 1 | — |
| `docs/frameworks/inferencex/index.md` | config-schema | `configs/CONFIGS.md` | Master config schema | 1 | — |
| `docs/frameworks/inferencex/index.md` | single-node-matrix | `infx/matrix/validation.py` | `SingleNodeMatrixEntry` | 155 | — |
| `docs/frameworks/inferencex/index.md` | multi-node-matrix | `infx/matrix/validation.py` | `MultiNodeMatrixEntry` | 254 | — |
| `docs/frameworks/inferencex/index.md` | agentic-validation | `infx/matrix/validation.py` | `validate_agentic_matrix_entry` | 433 | — |
| `docs/frameworks/inferencex/index.md` | matrix-validation | `infx/matrix/validation.py` | `validate_matrix_entry` | 446 | — |
| `docs/frameworks/inferencex/index.md` | matrix-generation | `infx/matrix/generate.py` | `generate_full_sweep` | 962 | — |
| `docs/frameworks/inferencex/index.md` | server-readiness | `benchmarks/benchmark_lib.sh` | `wait_for_server_ready` | 464 | — |
| `docs/frameworks/inferencex/index.md` | fixed-sequence-client | `benchmarks/benchmark_lib.sh` | `run_benchmark_serving` | 527 | — |
| `docs/frameworks/inferencex/index.md` | agentx-replay | `benchmarks/benchmark_lib.sh` | `run_agentic_replay_and_write_outputs` | 3227 | — |
| `docs/frameworks/inferencex/index.md` | fixed-result-normalization | `utils/process_result.py` | result identity and topology normalization | 113 | — |
| `docs/frameworks/inferencex/index.md` | result-collection | `utils/collect_results.py` | recursive JSON collection | 9 | — |
| `docs/frameworks/inferencex/index.md` | agentx-aggregation | `utils/agentic/aggregation/process_agentic_result.py` | `build_agg` | 194 | — |
| `docs/frameworks/inferencex/index.md` | agentx-output-gate | `utils/agentic/validation/validate_agentic_result.py` | `validate_result` | 49 | — |

## Runtime Flow Evidence

1. Configuration intent — `readme-overview`, `architecture-flow`, `config-schema`.
2. Matrix validation and expansion — `single-node-matrix`, `multi-node-matrix`, `agentic-validation`, `matrix-validation`, `matrix-generation`.
3. Server readiness and workload execution — `server-readiness`, `fixed-sequence-client`, `agentx-replay`.
4. Result normalization and AgentX gating — `fixed-result-normalization`, `agentx-aggregation`, `agentx-output-gate`.
5. Run-level artifact collection — `result-collection`.

## Evidence Map

- `README.md` — Open-source continuous inference research platform and official benchmark boundary
- `docs/architecture.md` — Config-to-result pipeline and ownership boundaries
- `configs/CONFIGS.md` — Master config and scenario schema
- `infx/matrix/validation.py` — Strict matrix schema and topology invariants
- `infx/matrix/generate.py` — Search-space expansion and derived matrix rows
- `benchmarks/benchmark_lib.sh` — Shared serving benchmark invocation and output contract
- `utils/process_result.py` — Fixed-sequence result normalization and throughput derivation
- `utils/collect_results.py` — Run-level JSON aggregation
- `utils/agentic/aggregation/process_agentic_result.py` — AgentX request filtering and aggregate metrics
- `utils/agentic/validation/validate_agentic_result.py` — AgentX pre-upload result gate

## Reproduction Commands

Record exact read-only search or counting commands here when the docs make
quantitative codebase claims.
