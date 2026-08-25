---
kind: repository-analysis
repository_id: github:vllm-project/vllm@8fe9317f2e401aff6e13044098ac7f59e95dce97
commit: 8fe9317f2e401aff6e13044098ac7f59e95dce97
source_record: raw/frameworks/vllm-codebase--github-8fe9317f2e40.md
generated: 2026-08-25
---

# vLLM Data Parallel Deployment Codebase Important Files

## Evidence Map

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

## Reproduction Commands

Record exact read-only search or counting commands here when the docs make
quantitative codebase claims.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | upstream-mode-guide | `docs/serving/data_parallel_deployment.md` | deployment guide | 1 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | cli-arguments | `vllm/engine/arg_utils.py` | data-parallel CLI definitions | 1095 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | cli-validation | `vllm/engine/arg_utils.py` | engine-argument validation | 2094 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | serve-mode-selection | `vllm/entrypoints/cli/serve.py` | serve mode selection | 78 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | local-engine-boundary | `vllm/config/parallel.py` | local_engines_only | 598 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | client-mode-selection | `vllm/v1/engine/core_client.py` | make_async_mp_client | 128 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | internal-load-score | `vllm/v1/engine/core_client.py` | DPLBAsyncMPClient.get_core_engine_for_request | 1472 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | external-rank-client | `vllm/v1/engine/core_client.py` | DPAsyncMPClient | 1253 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | coordinator-process | `vllm/v1/engine/coordinator.py` | DPCoordinator | 23 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | coordinator-stats | `vllm/v1/engine/coordinator.py` | DPCoordinatorProc.process_input_socket | 189 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | engine-topology | `vllm/v1/engine/utils.py` | launch_core_engines | 1104 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | request-rank-validation | `vllm/v1/engine/input_processor.py` | process_inputs | 301 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | request-materialization | `vllm/v1/engine/input_processor.py` | EngineCoreRequest construction | 418 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | moe-dummy-wave | `vllm/v1/engine/core.py` | DPEngineCoreProc.run_busy_loop | 2184 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | moe-pause-consensus | `vllm/v1/engine/core.py` | DPEngineCoreProc._has_global_unfinished_reqs | 2267 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | output-queue | `vllm/v1/engine/core_client.py` | AsyncMPClient.get_output_async | 1097 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | response-processing | `vllm/v1/engine/async_llm.py` | AsyncLLM output handler | 691 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | multi-port-children | `vllm/entrypoints/openai/dp_supervisor.py` | DPSupervisor._start_children | 401 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | multi-port-health | `vllm/entrypoints/openai/dp_supervisor.py` | DPSupervisor._probe_all_children | 417 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | routing-tests | `tests/v1/engine/test_engine_core_client.py` | DPLBAsyncMPClient routing tests | 206 | — |
| `docs/frameworks/vllm/data-parallel-deployment/index.md` | supervisor-tests | `tests/entrypoints/openai/test_dp_supervisor.py` | DPSupervisor tests | 136 | — |

## Runtime Flow Evidence

1. Request admission and optional rank validation — `request-rank-validation`,
   `request-materialization`.
2. Client selection and dispatch — `client-mode-selection`,
   `internal-load-score`, or `external-rank-client`.
3. Engine/coordinator handoff — `engine-topology`, `coordinator-process`,
   `coordinator-stats`.
4. MoE execution and global state transition — `moe-dummy-wave`,
   `moe-pause-consensus`.
5. Engine output return — `output-queue`.
6. Request output processing and routing-state release — `response-processing`.
7. Optional per-node external supervision — `multi-port-children`,
   `multi-port-health`.
