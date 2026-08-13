---
kind: repository-analysis
repository_id: github:vllm-project/vllm@2d24355eb87b716fc1169e66731dc0386ed1a3a2
commit: 2d24355eb87b716fc1169e66731dc0386ed1a3a2
source_record: raw/frameworks/vllm-codebase--github-2d24355eb87b.md
generated: 2026-08-13
---

# vllm Codebase Important Files

## Evidence Map

- `examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py` — Reference online router that sequences prefill and decode requests
- `examples/disaggregated/disaggregated_serving/disagg_proxy_pushconnector_demo.py` — Push-mode router and request metadata contract
- `vllm/config/kv_transfer.py` — KVTransferConfig connector, role, device, and extra configuration
- `vllm/distributed/kv_transfer/kv_connector/v1/base.py` — Scheduler-side and worker-side KV connector interfaces
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py` — NIXL pull-mode scheduler decisions
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py` — NIXL push-mode scheduler decisions
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py` — Decode-side pull transfer execution
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py` — Prefill-side push transfer execution
- `vllm/v1/core/sched/scheduler.py` — Engine scheduler integration and delayed request completion
- `vllm/v1/worker/kv_connector_model_runner_mixin.py` — Model-runner KV load/store and completion hooks
- `vllm/entrypoints/openai/chat_completion/serving.py` — OpenAI response propagation of KV transfer parameters

## Reader Contract

- **Audience:** operators and vLLM contributors who understand ordinary model serving but have not deployed prefill/decode disaggregation.
- **Question:** how does one request move through the router, prefill pool, KV-transfer plane, and decode pool, and which deployment choices affect correctness and latency?
- **Mental model:** the router turns one client request into two coordinated engine requests; the prefill engine computes and temporarily leases paged KV blocks, while the decode engine imports those blocks before generating the visible stream.
- **Lifecycle:** connector selection and transport setup happen at load time; request pairing, block allocation, transfer, decode, and release happen at runtime.
- **Limits:** static reading only; no GPUs, NIXL fabric, failure injection, throughput benchmark, or multi-node deployment was executed.

## Representation Plan

| Reader question | Evidence | Representation | Teaching job |
|---|---|---|---|
| Where do control-plane decisions and KV bytes move? | vLLM's original high-level connector figure and connector base interface | Preserved original source figure | Separate scheduler metadata planning from worker/data-plane transfer. |
| What changes between co-located and PD deployment? | Feature guide, proxy contract, scheduler integration | Compact comparison table | Make the extra router, transfer, and failure boundaries explicit. |
| What happens to one request over time? | Proxy, scheduler, worker, and completion hooks | Numbered runtime trace | Show state transitions, block ownership, and cleanup in order. |
| When should pull or push mode be chosen? | NIXL pull/push scheduler and worker paths | Decision table | Expose metadata dependencies and operational tradeoffs without implying benchmark results. |
| What must an operator validate before production? | Compatibility guide, lease/failure handling, exported metrics | Deployment checklist | Turn implementation details into testable rollout gates. |

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md` | pd-goals | `docs/features/disagg_prefill.md` | Why disaggregated prefilling | 8 | 16 |
| `docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md` | push-router-contract | `examples/disaggregated/disaggregated_serving/disagg_proxy_pushconnector_demo.py` | `PushProxy._push_completion` | 227 | 272 |
| `docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md` | remote-prefix-admission | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` external KV lookup | 767 | 820 |
| `docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md` | pull-match | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py` | `NixlPullConnectorScheduler.get_num_new_matched_tokens` | 34 | 110 |
| `docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md` | pull-stage-recv | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py` | `NixlPullConnectorScheduler.update_state_after_alloc` | 112 | 179 |
| `docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md` | scheduler-worker-metadata | `vllm/v1/core/sched/scheduler.py` | `Scheduler._build_kv_connector_meta` | 1220 | 1270 |
| `docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md` | worker-transfer-lifecycle | `vllm/v1/worker/kv_connector_model_runner_mixin.py` | `KVConnectorModelRunnerMixin._get_kv_connector_output` | 74 | 112 |
| `docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md` | pull-worker-read | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py` | `NixlPullConnectorWorker.start_load_kv` | 42 | 77 |
| `docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md` | producer-lease-params | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py` | `NixlPullConnectorScheduler.request_finished` | 237 | 280 |
| `docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md` | delayed-block-free | `vllm/v1/core/sched/scheduler.py` | `Scheduler._free_request` | 2338 | 2365 |
| `docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md` | transfer-completion | `vllm/v1/core/sched/scheduler.py` | `Scheduler._update_from_kv_xfer_finished` | 2752 | 2779 |
| `docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md` | openai-kv-params | `vllm/entrypoints/openai/chat_completion/serving.py` | `OpenAIServingChat` response construction | 1092 | 1105 |
| `docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md` | push-registration | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py` | `NixlPushConnectorScheduler.update_state_after_alloc` | 129 | 206 |
| `docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md` | push-worker-match | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py` | `NixlPushConnectorWorker._push_writer_loop` | 210 | 290 |
| `docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md` | load-failure-policy | `docs/features/nixl_connector_usage.md` | KV Load Failure Policy | 389 | 397 |
| `docs/frameworks/vllm/prefill-decode-disaggregated-deployment/index.md` | compatibility-boundary | `docs/features/nixl_connector_compatibility.md` | Configuration Notes | 74 | 110 |

## Runtime Flow Evidence

1. Router splits one client request into a one-token prefill leg and a decode leg with a shared request identity — `push-router-contract`.
2. Decode admission asks the connector how much of the prompt exists remotely and allocates local destination blocks — `remote-prefix-admission`, `pull-match`, `pull-stage-recv`.
3. The scheduler packages opaque transfer metadata for workers — `scheduler-worker-metadata`.
4. Worker connectors bind metadata, start asynchronous transfer, and report completion — `worker-transfer-lifecycle`, `pull-worker-read`.
5. The prefill side exports connection/block metadata and leases blocks instead of freeing them immediately — `producer-lease-params`, `delayed-block-free`.
6. Transfer completion promotes the decode request back to schedulable state or frees producer blocks — `transfer-completion`.
7. The OpenAI-compatible response can carry transfer parameters needed by the router, especially for later turns — `openai-kv-params`.
8. Push mode reverses the data operation: D registers its blocks, then P matches the registration and writes into them — `push-registration`, `push-worker-match`.
9. Production rollout must enforce compatibility and choose explicit transfer-failure behavior — `compatibility-boundary`, `load-failure-policy`.

## Reproduction Commands

Record exact read-only search or counting commands here when the docs make
quantitative codebase claims.
