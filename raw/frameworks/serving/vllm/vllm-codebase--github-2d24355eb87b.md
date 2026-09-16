---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm
repository_url: https://github.com/vllm-project/vllm
local_checkout: external-repos/vllm-2d24355eb87b/
commit: 2d24355eb87b716fc1169e66731dc0386ed1a3a2
ref: detached
inspected: 2026-08-13
checkout_state: clean
---

# vllm Codebase Source Record

## Reading Scope

- vLLM V1 prefill-decode disaggregated online deployment: request routing, KV-transfer contract, scheduler/worker connector lifecycle, NIXL pull and push modes, scaling and failure boundaries

## Important Entry Files

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

## Limitations

- Static code reading only; runtime behavior was not executed.
