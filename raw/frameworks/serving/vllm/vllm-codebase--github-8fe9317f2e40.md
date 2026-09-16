---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm
repository_url: https://github.com/vllm-project/vllm
local_checkout: external-repos/vllm-8fe9317f2e40/
commit: 8fe9317f2e401aff6e13044098ac7f59e95dce97
ref: detached
inspected: 2026-08-25
checkout_state: clean
---

# vLLM Data Parallel Deployment Codebase Source Record

## Reading Scope

- Data-parallel deployment topology, internal/hybrid/external load balancing, and MoE DP coordination

## Important Entry Files

- `vllm/engine/arg_utils.py` — Defines data-parallel CLI flags and validates internal, hybrid, and external launch combinations
- `vllm/v1/engine/core_client.py` — Selects the per-mode engine client and computes internal load-balancing choices
- `vllm/v1/engine/coordinator.py` — Collects engine request/KV statistics and coordinates MoE request waves
- `vllm/v1/engine/utils.py` — Launches local/remote core engines and DP coordinator for each topology
- `vllm/v1/engine/core.py` — Runs MoE DP dummy passes, pause consensus, and per-engine load-stat publication
- `vllm/v1/engine/input_processor.py` — Carries an optional externally selected DP rank into EngineCoreRequest
- `vllm/entrypoints/openai/dp_supervisor.py` — Launches one API child per local rank and aggregates health for multi-port external LB
- `docs/serving/data_parallel_deployment.md` — Upstream deployment recipes and mode descriptions at the pinned revision
- `examples/features/data_parallel/multi_instance_data_parallel.py` — Runnable per-rank external-DP example
- `tests/v1/engine/test_engine_core_client.py` — Unit coverage for internal DP routing, KV-pressure scoring, and request lifecycle
- `tests/entrypoints/openai/test_dp_supervisor.py` — Unit and lifecycle coverage for multi-port supervisor behavior

## Limitations

- No multi-GPU or Ray serving run was executed in this workspace; throughput and fabric behavior remain unverified.
