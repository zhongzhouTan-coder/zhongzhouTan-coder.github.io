---
kind: repository-source
provider: github
clone_url: https://github.com/SemiAnalysisAI/InferenceX.git
repository_url: https://github.com/SemiAnalysisAI/InferenceX
local_checkout: external-repos/InferenceX/
commit: 7fa1d14b6624d433bd4f72a540da739b8f3808d9
ref: main
inspected: 2026-09-09
checkout_state: clean
---

# InferenceX Source Record

## Reading Scope

- Config-to-artifact benchmark pipeline: matrix generation, standardized serving runs, AgentX replay, result aggregation, and app handoff

## Important Entry Files

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

## Limitations

- This page is a static code reading at pinned commit 7fa1d14b6624d433bd4f72a540da739b8f3808d9; it does not reproduce GPU or multi-node benchmark runs.
