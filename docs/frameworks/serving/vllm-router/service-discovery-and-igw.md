---
title: "vLLM Router: Service Discovery and IGW Code Path"
summary: "A code-backed trace of Kubernetes worker discovery, Inference Gateway multi-router delegation, and how the two mechanisms compose in vLLM Router."
layout: default
confidence: medium
sources:
  - raw/frameworks/serving/vllm-router/router-codebase--github-aaad26899d00.md
  - derived/repo-analysis/frameworks/serving/vllm-router/router/aaad26899d0052cb62c0998346380db80d54d24a/important-files.md
updated: 2026-11-18
code_links: strict
code_evidence: strict
---

# vLLM Router: Service Discovery and IGW Code Path

**Repository:** [vllm-project/router](https://github.com/vllm-project/router)

**Inspected commit:**
`aaad26899d0052cb62c0998346380db80d54d24a`

**Verification boundary:** clean, commit-pinned static reading. A Kubernetes
cluster, vLLM registration sender, running worker fleet, and inference request
were not executed.

## The Short Version

`--enable-igw` and `--service-discovery` solve different control-plane
problems:

- **IGW chooses a router implementation.** It creates a `RouterManager` with
  regular and P/D child routers, then delegates each request using model and
  worker-type metadata.
- **Service discovery populates worker topology.** It watches Kubernetes pods
  and adds or removes workers behind the currently active router.

Neither flag performs model execution. They decide which router sees a request
and which worker endpoints are currently candidates.

## The Two Control Loops

| Control loop | Input | Persistent state it changes | Consumer |
|---|---|---|---|
| IGW registration and request delegation | `/workers` registrations and inference requests | Shared model-indexed worker registry and per-model policy mapping | `RouterManager` and its child routers |
| Kubernetes discovery | Pod add, update, delete, readiness, labels, IP, and UID | Discovery tracking map plus the shared worker registry | Active `RouterTrait` and background health checks |

## `--enable-igw`

### 1. Configuration removes the single-router assumption

<a class="code-link" href="../../../../external-repos/router/src/main.rs#L489" data-code-repo="router-aaad26899d00" data-code-path="src/main.rs" data-code-line="489" data-code-end-line="555"><code>CliArgs::to_router_config()</code></a>
sets `RoutingMode::Regular` with an empty worker list when IGW is enabled.
This mode is only a placeholder: the later server startup branch does not
construct that regular router as the request boundary.

IGW also bypasses the usual CLI choice among static regular, OpenAI, or vLLM
P/D top-level routing. Static `--worker-urls`, `--prefill`, `--decode`, and
normal `--vllm-discovery-address` mode selection are not copied into the
placeholder path.

### 2. Startup creates a meta-router and two child routers

<a class="code-link" href="../../../../external-repos/router/src/server.rs#L1060" data-code-repo="router-aaad26899d00" data-code-path="src/server.rs" data-code-line="1060" data-code-end-line="1126"><code>startup()</code></a>
creates:

- an `http-regular` child with no initial workers;
- an `http-pd` child with no initial P or D workers and no vLLM discovery
  address;
- a `RouterManager` that becomes the server's `RouterTrait`.

All three share the `AppContext` worker and policy registries. The manager is
therefore not a second worker fleet. It is a dispatcher over the same
registered topology.

### 3. Worker registration records enough metadata to select a child

<a class="code-link" href="../../../../external-repos/router/src/routers/router_manager.rs#L150" data-code-repo="router-aaad26899d00" data-code-path="src/routers/router_manager.rs" data-code-line="150" data-code-end-line="244"><code>RouterManager::add_worker()</code></a>
accepts a `WorkerConfigRequest` through `POST /workers`.

The manager:

1. uses an explicit `model_id`, or queries the worker model endpoint;
2. stores `model_id`, priority, cost, and caller labels;
3. creates a regular, prefill, or decode worker;
4. inserts it into the shared `WorkerRegistry`;
5. creates or reuses the model's policy mapping;
6. registers regular workers with an applicable KV-aware policy.

The worker `type` is the first important classification:

- regular → `http-regular`;
- prefill or decode → `http-pd`.

### 4. Request delegation is model-first

<a class="code-link" href="../../../../external-repos/router/src/routers/router_manager.rs#L109" data-code-repo="router-aaad26899d00" data-code-path="src/routers/router_manager.rs" data-code-line="109" data-code-end-line="140"><code>RouterManager::get_router_for_model()</code></a>
looks up the request model in the worker registry. If that model has any
prefill or decode workers, it selects `http-pd`; otherwise it selects
`http-regular`. Missing model matches fall back to the default registered
router.

The broader
<a class="code-link" href="../../../../external-repos/router/src/routers/router_manager.rs#L398" data-code-repo="router-aaad26899d00" data-code-path="src/routers/router_manager.rs" data-code-line="398" data-code-end-line="475"><code>RouterManager::select_router_for_request()</code></a>
adds request-level preference:

- `x-prefer-pd=true` or `x-prefer-pd=1` bonuses the P/D router.
- Without that header, the regular router receives a selection bonus.
- `x-worker-priority` and `x-max-cost` are parsed but are not yet used in the
  score.
- Worker load, health, cost, and priority are explicitly left as future
  cross-router selection inputs.

For a chat request,
<a class="code-link" href="../../../../external-repos/router/src/routers/router_manager.rs#L627" data-code-repo="router-aaad26899d00" data-code-path="src/routers/router_manager.rs" data-code-line="627" data-code-end-line="648"><code>RouterManager::route_chat()</code></a>
then delegates the original headers and typed body to the selected child.
From that point, ordinary regular or P/D selection handles worker placement.

> **Inference:** IGW is a two-level decision. First choose a router family;
> then that child's load-balancing policy chooses a concrete worker or P/D
> pair.

## `--service-discovery`

### 1. Configuration supplies selectors and pod-to-worker mapping

<a class="code-link" href="../../../../external-repos/router/src/main.rs#L698" data-code-repo="router-aaad26899d00" data-code-path="src/main.rs" data-code-line="698" data-code-end-line="718"><code>CliArgs::to_server_config()</code></a>
passes a Kubernetes watcher configuration to startup:

| Setting | Meaning |
|---|---|
| `--selector` | Labels for regular workers |
| `--prefill-selector` | Labels for P/D prefill workers |
| `--decode-selector` | Labels for P/D decode workers |
| `--service-discovery-port` | Port appended to the pod IP |
| `--service-discovery-namespace` | One namespace, or all namespaces when unset |
| `vllm.ai/bootstrap-port` | Optional prefill bootstrap-port annotation |

When discovery is enabled, the regular or P/D router may start with no static
worker URLs.

### 2. Startup attaches the watcher to the active router

The server waits until router construction is complete, then starts discovery in
<a class="code-link" href="../../../../external-repos/router/src/server.rs#L1164" data-code-repo="router-aaad26899d00" data-code-path="src/server.rs" data-code-line="1164" data-code-end-line="1184"><code>startup()</code></a>.
The watcher receives the server-level `router_arc`, so the discovery target is:

- the concrete regular or P/D router in single-router mode;
- the `RouterManager` in IGW mode.

Failure to initialize Kubernetes discovery is logged, and the process
continues with whatever static or already registered workers it has.

### 3. The Kubernetes watcher filters and reconciles pod state

<a class="code-link" href="../../../../external-repos/router/src/service_discovery.rs#L226" data-code-repo="router-aaad26899d00" data-code-path="src/service_discovery.rs" data-code-line="226" data-code-end-line="402"><code>start_service_discovery()</code></a>
creates a Kubernetes client and a long-running pod watcher. It:

1. watches one namespace or all namespaces;
2. filters events to matching selectors or already tracked pods;
3. converts each pod into readiness, IP, role, UID, and bootstrap metadata;
4. routes additions and deletions into reconciliation;
5. retries a failed watcher with exponential backoff and restarts it.

The watcher does not continuously run a placement policy. It only mutates the
worker topology.

### 4. Reconciliation preserves old identity until replacement is safe

<a class="code-link" href="../../../../external-repos/router/src/service_discovery.rs#L407" data-code-repo="router-aaad26899d00" data-code-path="src/service_discovery.rs" data-code-line="407" data-code-end-line="520"><code>handle_pod_event()</code></a>
tracks pods by namespace-qualified name and stores:

- registration-time worker URL;
- regular/prefill/decode role;
- pod UID.

An exact duplicate is ignored. A changed IP, UID, or role is reconciled:

1. replace the tracking record;
2. remove the stale worker using its old registered URL;
3. add the replacement URL.

Only `Ready=True`, `Running` pods that still match a selector are eligible. A
pod that becomes unready or stops matching is removed from both tracking and
the router.

### 5. Removal is role-sensitive

<a class="code-link" href="../../../../external-repos/router/src/service_discovery.rs#L567" data-code-repo="router-aaad26899d00" data-code-path="src/service_discovery.rs" data-code-line="567" data-code-end-line="601"><code>remove_worker_from_router()</code></a>
uses a direct `VllmPDRouter` downcast in P/D mode to invoke
`remove_prefill_server()` or `remove_decode_server()`. Regular discovery calls
the generic worker-removal method.

Using the registration-time URL is important: recomputing an address from a
pod that already changed IP would fail to remove the stale endpoint.

## vLLM-Native P/D Discovery Is Separate

`--vllm-discovery-address` is not the Kubernetes watcher. In P/D mode,
<a class="code-link" href="../../../../external-repos/router/src/routers/http/vllm_pd_router.rs#L3030" data-code-repo="router-aaad26899d00" data-code-path="src/routers/http/vllm_pd_router.rs" data-code-line="3030" data-code-end-line="3071"><code>VllmPDRouter::new()</code></a>
creates a `ServiceRegistry` instead of registering static P/D URLs.

<a class="code-link" href="../../../../external-repos/router/src/routers/http/vllm_service_discovery.rs#L170" data-code-repo="router-aaad26899d00" data-code-path="src/routers/http/vllm_service_discovery.rs" data-code-line="170" data-code-end-line="265"><code>ServiceRegistry::start_listener()</code></a>
binds a ZeroMQ `ROUTER` socket. vLLM workers announce:

- service type `P` or `D`;
- HTTP endpoint;
- ZMQ coordination endpoint;
- tensor/data-parallel sizing;
- connector-specific metadata.

Registrations expire unless workers send fresh announcements. This registry is
the source of per-request P/D candidates.

## Composition Matrix

| Flags and inputs | Result in this revision |
|---|---|
| No discovery, no IGW | Concrete regular, OpenAI, or P/D router uses static URLs |
| Kubernetes discovery, no IGW | The concrete regular or P/D router receives pod-backed workers |
| IGW, no discovery | `RouterManager` accepts typed workers through `/workers` |
| IGW plus regular Kubernetes discovery | Kubernetes calls generic `add_worker()` on `RouterManager`; model information is queried from the worker |
| IGW plus P/D Kubernetes discovery | Unsupported by the current add path: the watcher requires a direct `VllmPDRouter` downcast, but the top-level router is `RouterManager` |
| vLLM ZMQ P/D discovery, no IGW | `VllmPDRouter` owns the live P/D service registry |
| IGW plus `--vllm-discovery-address` | The IGW startup branch constructs an empty P/D child without that discovery address |

> **Important:** Kubernetes discovery and vLLM ZMQ discovery answer different
> questions. Kubernetes supplies pod addresses and roles. vLLM-native P/D
> discovery additionally supplies the coordination addresses and transfer
> dimensions expected by the P/D connector path.

## Failure and Operational Boundaries

| Boundary | Current behavior |
|---|---|
| Kubernetes API becomes unavailable at startup | Startup logs the failure and continues without the watcher |
| Kubernetes watch disconnects | The watcher retries with exponential delay and restarts |
| Pod becomes unready | The tracked worker is removed |
| Pod changes IP or UID | The old registration is removed before the replacement is added |
| Worker model query fails in IGW | Worker model is recorded as `unknown` |
| IGW request has no model or matching worker | Manager selection falls back to registered routers or returns not found |
| P/D Kubernetes discovery behind IGW | Add/remove cannot downcast to `VllmPDRouter` |
| P/D worker stops sending ZMQ registrations | Its service-registry entry expires |
| Cross-router cost or priority routing | Header values are parsed but not yet included in the selection score |

## Where This Page Fits

This page covers router control-plane topology only. Return to the
[vLLM Router repository overview](index.md) for regular and P/D request
round trips. Use the
[vLLM P/D deployment path](../vllm/prefill-decode-disaggregated-deployment/index.md)
for worker-side KV transfer.

**One thing to remember:** IGW routes *between router implementations*;
service discovery routes *topology changes into the worker registry*.
