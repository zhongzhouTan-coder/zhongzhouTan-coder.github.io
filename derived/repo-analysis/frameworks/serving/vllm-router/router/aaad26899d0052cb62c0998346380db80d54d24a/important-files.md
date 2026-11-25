---
kind: repository-analysis
repository_id: github:vllm-project/router@aaad26899d0052cb62c0998346380db80d54d24a
commit: aaad26899d0052cb62c0998346380db80d54d24a
source_record: raw/frameworks/serving/vllm-router/router-codebase--github-aaad26899d00.md
generated: 2026-11-18
---

# vLLM Router Codebase Important Files

## Evidence Map

### Reader Contract

- **Audience:** engineers who already know OpenAI-compatible serving but need
  a code-reading map of the router's control plane.
- **Question:** How does one request enter, get admitted to a worker or P/D
  pair, cross the HTTP or gRPC backend boundary, and return to the caller?
- **Mental model:** the router owns admission and placement while vLLM workers
  own token generation and KV computation.
- **Runtime scope:** static regular and P/D request paths, HTTP and gRPC
  dispatch, discovery, and cleanup.
- **Lifecycle:** worker and policy ownership are established at startup;
  availability, selection, reservations, backend calls, and response cleanup
  happen at request time.
- **Boundary:** no router process, worker, accelerator, transfer fabric, or
  multi-node deployment was executed.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/serving/vllm-router/index.md` | server-routes | `src/server.rs` | `build_app_with_wasm_middleware` | 862 | 913 |
| `docs/frameworks/serving/vllm-router/index.md` | router-factory | `src/routers/factory.rs` | `RouterFactory::create_router` | 15 | 42 |
| `docs/frameworks/serving/vllm-router/index.md` | policy-contract | `src/policies/mod.rs` | `LoadBalancingPolicy` | 58 | 131 |
| `docs/frameworks/serving/vllm-router/index.md` | regular-selection | `src/routers/http/router.rs` | `Router::select_worker_for_model` | 1395 | 1443 |
| `docs/frameworks/serving/vllm-router/index.md` | regular-admission | `src/routers/http/router.rs` | `Router::route_prepared_request` | 1463 | 1788 |
| `docs/frameworks/serving/vllm-router/index.md` | grpc-prepare | `src/backend/frontend.rs` | `EngineFrontend::prepare` | 155 | 178 |
| `docs/frameworks/serving/vllm-router/index.md` | regular-dispatch | `src/routers/http/router.rs` | `Router::send_typed_request` | 1937 | 2120 |
| `docs/frameworks/serving/vllm-router/index.md` | load-cleanup | `src/routers/http/router.rs` | `RequestLoadGuard` | 116 | 188 |
| `docs/frameworks/serving/vllm-router/index.md` | pd-admission | `src/routers/http/pd_router.rs` | `PdRouterBase::select_and_admit_with` | 682 | 756 |
| `docs/frameworks/serving/vllm-router/index.md` | pd-entry | `src/routers/http/vllm_pd_router.rs` | `VllmPDRouter::route_transparent` | 3576 | 3704 |
| `docs/frameworks/serving/vllm-router/index.md` | pd-handoff | `src/routers/http/vllm_pd_router.rs` | `VllmPDRouter::process_vllm_two_stage_request` | 2124 | 2528 |
| `docs/frameworks/serving/vllm-router/index.md` | pd-response | `src/routers/http/vllm_pd_router.rs` | `VllmPDRouter::handle_decode_response` | 1869 | 1925 |
| `docs/frameworks/serving/vllm-router/index.md` | kv-index | `src/kv_index/discovery.rs` | `KvIndexSupervisor::on_worker_added` | 216 | 245 |
| `docs/frameworks/serving/vllm-router/index.md` | worker-discovery | `src/service_discovery.rs` | `handle_pod_event` | 407 | 520 |
| `docs/frameworks/serving/vllm-router/service-discovery-and-igw.md` | cli-routing-mode | `src/main.rs` | `CliArgs::to_router_config` | 489 | 555 |
| `docs/frameworks/serving/vllm-router/service-discovery-and-igw.md` | igw-startup | `src/server.rs` | `startup` | 1060 | 1126 |
| `docs/frameworks/serving/vllm-router/service-discovery-and-igw.md` | igw-worker-registration | `src/routers/router_manager.rs` | `RouterManager::add_worker` | 150 | 244 |
| `docs/frameworks/serving/vllm-router/service-discovery-and-igw.md` | igw-model-routing | `src/routers/router_manager.rs` | `RouterManager::get_router_for_model` | 109 | 140 |
| `docs/frameworks/serving/vllm-router/service-discovery-and-igw.md` | igw-request-selection | `src/routers/router_manager.rs` | `RouterManager::select_router_for_request` | 398 | 475 |
| `docs/frameworks/serving/vllm-router/service-discovery-and-igw.md` | igw-dispatch | `src/routers/router_manager.rs` | `RouterManager::route_chat` | 627 | 648 |
| `docs/frameworks/serving/vllm-router/service-discovery-and-igw.md` | k8s-config | `src/main.rs` | `CliArgs::to_server_config` | 698 | 718 |
| `docs/frameworks/serving/vllm-router/service-discovery-and-igw.md` | discovery-start | `src/server.rs` | `startup` | 1164 | 1184 |
| `docs/frameworks/serving/vllm-router/service-discovery-and-igw.md` | k8s-watcher | `src/service_discovery.rs` | `start_service_discovery` | 226 | 402 |
| `docs/frameworks/serving/vllm-router/service-discovery-and-igw.md` | k8s-reconcile | `src/service_discovery.rs` | `handle_pod_event` | 407 | 520 |
| `docs/frameworks/serving/vllm-router/service-discovery-and-igw.md` | k8s-remove | `src/service_discovery.rs` | `remove_worker_from_router` | 567 | 601 |
| `docs/frameworks/serving/vllm-router/service-discovery-and-igw.md` | vllm-zmq-discovery | `src/routers/http/vllm_service_discovery.rs` | `ServiceRegistry::start_listener` | 170 | 265 |
| `docs/frameworks/serving/vllm-router/service-discovery-and-igw.md` | pd-discovery-construction | `src/routers/http/vllm_pd_router.rs` | `VllmPDRouter::new` | 3030 | 3071 |

## Request Round-Trip Evidence

1. Entry or admission — `server-routes`, then `regular-admission` or
   `pd-entry`.
2. Policy selection and reservation — `regular-selection` or
   `pd-admission`.
3. Worker transport preparation — `grpc-prepare` for gRPC, or direct HTTP
   serialization in `regular-dispatch`.
4. Backend dispatch — `regular-dispatch` or `pd-handoff`.
5. First material result — a worker `Response` reaches
   `regular-dispatch`, while P/D receives the prefill response in
   `pd-handoff` before decode.
6. Result transformation — `pd-response` selects streaming, logprob merge, or
   buffered response assembly for P/D.
7. Response emission and cleanup — `load-cleanup` owns regular capacity until
   body completion; P/D `pd-handoff` and `pd-response` transition and release
   the admitted attempt.

The dedicated IGW and service-discovery document uses two additional lifecycles:

1. IGW worker registration — `igw-worker-registration`; then model/type
   indexing through `igw-model-routing`.
2. IGW request delegation — `igw-request-selection` and `igw-dispatch`.
3. Kubernetes topology updates — `k8s-config`, `discovery-start`,
   `k8s-watcher`, `k8s-reconcile`, and `k8s-remove`.
4. vLLM-native P/D registration — `vllm-zmq-discovery` and
   `pd-discovery-construction`.

## Link Completion

- [x] Every Required Code Evidence row has a matching code link.
- [x] Every non-trivial round-trip step names at least one declared finding.
- [x] The trace follows one request through dispatch, response, and cleanup.
- [x] Producer and consumer sides are linked where state crosses boundaries.

## Reproduction Commands

```bash
git -C external-repos/router status --short
git -C external-repos/router rev-parse HEAD
git -C external-repos/router remote get-url origin
rg -n "pub async fn route_prepared_request|async fn send_typed_request" \
  external-repos/router/src/routers/http/router.rs
rg -n "process_vllm_two_stage_request|handle_decode_response" \
  external-repos/router/src/routers/http/vllm_pd_router.rs
```
