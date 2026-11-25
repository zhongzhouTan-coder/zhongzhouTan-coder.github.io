---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/router.git
repository_url: https://github.com/vllm-project/router
local_checkout: external-repos/router/
commit: aaad26899d0052cb62c0998346380db80d54d24a
ref: main
inspected: 2026-11-18
checkout_state: clean
---

# vLLM Router Codebase Source Record

## Reading Scope

- Request routing architecture, regular and prefill/decode disaggregation, HTTP and gRPC backends, worker selection, resilience, and discovery

## Important Entry Files

- `src/server.rs` — Axum server construction, ingress routes, middleware, and startup
- `src/routers/factory.rs` — Selects regular, OpenAI, or P/D router implementations from configuration
- `src/routers/http/router.rs` — Regular routing, policy selection, retry dispatch, streaming, and load cleanup
- `src/routers/http/vllm_pd_router.rs` — P/D admission, prefill-to-decode handoff, and response assembly
- `src/policies/mod.rs` — Load-balancing policy contract and optional KV-aware operations
- `src/policies/factory.rs` — Configuration-to-policy construction
- `src/backend/frontend.rs` — gRPC preprocessing and dispatch boundary
- `src/core/worker_registry.rs` — Worker topology, model/type indexes, and availability state
- `src/kv_index/mod.rs` — KV event ingestion and tiered cache placement index
- `src/service_discovery.rs` — Kubernetes worker discovery and reconciliation

## Limitations

- Clean, commit-pinned static code reading; no router, worker, GPU, network, or multi-node runtime was executed.
