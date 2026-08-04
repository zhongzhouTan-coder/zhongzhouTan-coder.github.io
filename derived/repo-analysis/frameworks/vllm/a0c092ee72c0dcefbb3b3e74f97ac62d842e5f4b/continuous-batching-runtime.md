---
kind: repository-analysis
repository_id: github:vllm-project/vllm@a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
commit: a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
source_record: raw/frameworks/vllm-codebase--github-a0c092ee72c0.md
generated: 2026-08-02
---

# vLLM V1 Continuous-Batching Runtime Evidence

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm/vllm-continuous-batching/index.md` | engine-step | `vllm/v1/engine/core.py` | `EngineCore.step` | 584 | — |
| `docs/frameworks/vllm/vllm-continuous-batching/index.md` | scheduler-schedule | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 427 | — |
| `docs/frameworks/vllm/vllm-continuous-batching/index.md` | request-state | `vllm/v1/request.py` | `Request` | 59 | — |
| `docs/frameworks/vllm/vllm-continuous-batching/index.md` | kv-admission | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.allocate_slots` | 344 | — |
| `docs/frameworks/vllm/vllm-continuous-batching/index.md` | worker-batch | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner._update_states` | 1192 | — |
| `docs/frameworks/vllm/vllm-continuous-batching/index.md` | update-from-output | `vllm/v1/core/sched/scheduler.py` | `Scheduler.update_from_output` | 1653 | — |

## Runtime Flow Evidence

1. Entry — `engine-step`.
2. Coordination and dispatch — `scheduler-schedule`, `request-state`.
3. Core state transition (KV admission) — `kv-admission`.
4. Materialization and backend handoff — `worker-batch`.
5. Completion and cleanup — `update-from-output`.

## Reading Scope

Static reading of the V1 online-generation loop: per-iteration scheduling,
token and sequence budgets, waiting-request admission, KV-slot allocation,
persistent worker batches, completion, block release, and preemption.

## End-to-End Path

1. `vllm/v1/engine/core.py::EngineCore.step` calls
   `scheduler.schedule()`, submits the returned `SchedulerOutput` to
   `model_executor.execute_model()`, then feeds the result to
   `scheduler.update_from_output()`.
2. `vllm/v1/core/sched/scheduler.py::Scheduler.schedule` creates a fresh token
   budget for every engine step. It schedules existing `running` requests
   first, then admits `waiting` requests while token, sequence, KV-cache,
   encoder, LoRA, and model-length constraints allow.
3. The scheduler does not maintain separate prompt and decode queues. A
   request exposes the gap between `num_computed_tokens` and
   `num_tokens_with_spec`; filling that gap covers prefill chunks, ordinary
   one-token decode, and speculative tokens with the same accounting model.
4. `vllm/v1/core/kv_cache_manager.py::KVCacheManager.allocate_slots` checks and
   allocates only the blocks needed for cached-prefix tokens, new work, and
   lookahead. Returning `None` is an admission failure that can trigger
   preemption of a running request.
5. `vllm/v1/worker/gpu_model_runner.py::_update_states` removes finished and
   unscheduled rows, updates cached requests and block tables, adds new or
   resumed requests, and preserves overlapping request rows across iterations.
6. `Scheduler.update_from_output` appends sampled tokens, evaluates stopping
   conditions, removes stopped requests from the running set, and calls
   `_free_request`; the freed request ID is carried in a later scheduler output
   so workers can remove request-scoped state.

## Scheduling Invariants

- `max_num_scheduled_tokens` (normally `max_num_batched_tokens`) limits total
  tokens issued in one iteration.
- `max_num_seqs` limits requests held in the running set.
- Existing running work is considered before waiting work under the default
  first-come-first-served policy.
- With chunked prefill enabled, a prompt can consume only the remaining token
  budget and continue in a later iteration; decode work and prompt chunks can
  therefore share an iteration.
- A decode request commonly contributes one token of work, but speculative
  decoding can contribute more. The batch is therefore token-budgeted, not
  accurately described by a single fixed request count.
- If KV slots cannot be allocated, the default FCFS path removes a request from
  the tail of `running`, frees its request state through `_preempt_request`, and
  later resumes it from the waiting queue by recomputation or cache reuse.

## Important Files and Symbols

- `vllm/v1/engine/core.py::EngineCore.step` — synchronous schedule/execute/update
  loop; `step_with_batch_queue` is the overlapped-batch variant.
- `vllm/v1/core/sched/scheduler.py::Scheduler.schedule` — per-step construction
  of the token-level batch.
- `vllm/vllm/config/scheduler.py::SchedulerConfig` — token, sequence, chunked
  prefill, FCFS/priority, watermark, and asynchronous-scheduling controls.
- `vllm/v1/request.py::Request` — request state, including
  `num_computed_tokens`, `num_tokens_with_spec`, and `is_prefill_chunk`.
- `vllm/v1/core/kv_cache_manager.py::KVCacheManager.allocate_slots` — KV-block
  admission and allocation.
- `vllm/v1/core/sched/output.py::SchedulerOutput` — scheduled new/cached
  requests, per-request token counts, new blocks, finished IDs, and preempted
  IDs sent toward workers.
- `vllm/v1/worker/gpu_model_runner.py::_update_states` — worker-side persistent
  batch reconciliation.
- `vllm/v1/core/sched/scheduler.py::Scheduler.update_from_output` — token,
  finish, queue, and block-lifecycle updates after model execution.

## Reproduction Commands

```bash
git check-ignore external-repos/vllm
git -C external-repos/vllm remote get-url origin
git -C external-repos/vllm rev-parse HEAD
git -C external-repos/vllm branch --show-current
git -C external-repos/vllm status --porcelain
rg -n '^class Scheduler|^    def schedule\(' external-repos/vllm/vllm/v1/core/sched -g '*.py'
rg -n 'def step|scheduler.schedule|execute_model' external-repos/vllm/vllm/v1/engine external-repos/vllm/vllm/v1/core -g '*.py'
rg -n 'class KVCacheManager|def allocate_slots|def free' external-repos/vllm/vllm/v1/core -g '*.py'
rg -n 'max_num_batched_tokens|max_num_seqs|enable_chunked_prefill|policy:' external-repos/vllm/vllm/config external-repos/vllm/vllm/engine/arg_utils.py -g '*.py'
```

## Limitations

- Static reading only; no GPU throughput or latency benchmark was executed.
- The checkout is a newer V1 implementation than the 2023 web articles. The
  articles explain the durable scheduling idea, while current symbol names and
  edge cases come from this pinned revision.
- Platform-specific model runners, distributed pipeline timing, multimodal
  encoder scheduling, speculative-decoding variants, and external KV
  connectors add branches beyond the ordinary decoder-only path summarized
  here.
