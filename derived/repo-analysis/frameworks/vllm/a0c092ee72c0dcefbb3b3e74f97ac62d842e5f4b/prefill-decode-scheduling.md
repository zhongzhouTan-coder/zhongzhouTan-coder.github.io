---
kind: repository-analysis
repository_id: github:vllm-project/vllm@a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
commit: a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
source_record: raw/frameworks/vllm-codebase--github-a0c092ee72c0.md
generated: 2026-08-06
---

# vLLM Prefill and Decode Scheduling Evidence

Consuming page: `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md`

## Evidence Map

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md` | engine-step | `vllm/v1/engine/core.py` | `EngineCore.step` | 584 | 606 |
| `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md` | scheduler-budget | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 427 | 617 |
| `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md` | request-target | `vllm/v1/request.py` | `Request.num_tokens_with_spec` | 277 | 278 |
| `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md` | kv-admission | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.allocate_slots` | 344 | 410 |
| `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md` | chunk-progress | `vllm/v1/core/sched/scheduler.py` | prefill cursor update and `is_prefill_chunk` | 1318 | 1325 |
| `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md` | worker-persistent-batch | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner._update_states` | 1192 | 1230 |

## Runtime Flow Evidence

1. `EngineCore.step` closes schedule -> execute -> update.
2. `Scheduler.schedule` spends a shared token budget across running and waiting requests.
3. `KVCacheManager.allocate_slots` allocates blocks for only the scheduled token range.
4. `GPUModelRunner._update_states` reconciles the persistent worker batch.
5. The scheduler's per-request token counts become the input query lengths consumed by attention metadata builders.

## Scheduling Invariants

- There is no fundamental prefill queue and decode queue in the V1 scheduler; the scheduler advances `num_computed_tokens` toward `num_tokens_with_spec`.
- Running requests are visited before waiting requests under the default policy, so decode work can claim budget before new prompt admission.
- With chunked prefill enabled, a prompt can consume the remaining budget and continue in a later iteration.
- `max_num_scheduled_tokens` limits total token work per iteration; `max_num_seqs` limits resident request count.
- KV allocation is part of scheduling. A token plan without enough blocks is not runnable.

## Freshness Boundary

Scoped sync on 2026-08-06 found upstream changes after this pinned snapshot but returned `decision: defer` because the previous snapshot is younger than the 14-day promotion interval; eligible date is 2026-08-12. This note therefore records only the pinned checkout.

## Limitations

Static reading only; no GPU or NPU execution was performed. Speculative decoding, pipeline parallelism, data-parallel prefill balancing, multimodal encoder work, and KV connectors add branches beyond the ordinary decoder-only example.
