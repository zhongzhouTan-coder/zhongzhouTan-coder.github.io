---
kind: repository-analysis
repository_id: github:vllm-project/vllm@a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
commit: a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b
source_record: raw/frameworks/vllm-codebase--github-a0c092ee72c0.md
generated: 2026-08-05
---

# vLLM Repository Overview — Reading Notes

Scope: whole-repository layout and the V1 serving path, from entrypoints to
native kernels, at the pinned mainline revision
`a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b` (2026-07-29). Static reading of a
clean detached checkout (inspected 2026-08-05); no runtime execution.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm/vllm-overview.md` | api-server | `vllm/entrypoints/openai/api_server.py` | `build_app` | 189 | — |
| `docs/frameworks/vllm/vllm-overview.md` | openai-chat-serving | `vllm/entrypoints/openai/chat_completion/serving.py` | `OpenAIServingChat` | 112 | — |
| `docs/frameworks/vllm/vllm-overview.md` | offline-llm | `vllm/entrypoints/llm.py` | `LLM` | 66 | — |
| `docs/frameworks/vllm/vllm-overview.md` | async-frontend | `vllm/v1/engine/async_llm.py` | `AsyncLLM` | 72 | — |
| `docs/frameworks/vllm/vllm-overview.md` | input-processing | `vllm/v1/engine/input_processor.py` | `InputProcessor` | 37 | — |
| `docs/frameworks/vllm/vllm-overview.md` | output-processing | `vllm/v1/engine/output_processor.py` | `OutputProcessor` | 429 | — |
| `docs/frameworks/vllm/vllm-overview.md` | detokenizer | `vllm/v1/engine/detokenizer.py` | `IncrementalDetokenizer` | 31 | — |
| `docs/frameworks/vllm/vllm-overview.md` | engine-core | `vllm/v1/engine/core.py` | `EngineCore.step` | 584 | 621 |
| `docs/frameworks/vllm/vllm-overview.md` | request | `vllm/v1/request.py` | `Request` | 59 | — |
| `docs/frameworks/vllm/vllm-overview.md` | scheduler | `vllm/v1/core/sched/scheduler.py` | `Scheduler` | 69 | — |
| `docs/frameworks/vllm/vllm-overview.md` | scheduler-output | `vllm/v1/core/sched/output.py` | `SchedulerOutput` | 193 | — |
| `docs/frameworks/vllm/vllm-overview.md` | kv-cache-manager | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager` | 117 | — |
| `docs/frameworks/vllm/vllm-overview.md` | block-pool | `vllm/v1/core/block_pool.py` | `BlockPool` | 143 | — |
| `docs/frameworks/vllm/vllm-overview.md` | executor | `vllm/v1/executor/abstract.py` | `Executor.execute_model` | 210 | — |
| `docs/frameworks/vllm/vllm-overview.md` | worker-base | `vllm/v1/worker/worker_base.py` | `WorkerBase` | 39 | — |
| `docs/frameworks/vllm/vllm-overview.md` | gpu-worker | `vllm/v1/worker/gpu_worker.py` | `Worker` | 128 | — |
| `docs/frameworks/vllm/vllm-overview.md` | gpu-model-runner | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner` | 453 | — |
| `docs/frameworks/vllm/vllm-overview.md` | worker-execute | `vllm/v1/worker/gpu_worker.py` | `Worker.execute_model` | 1019 | — |
| `docs/frameworks/vllm/vllm-overview.md` | worker-memory-profile | `vllm/v1/worker/gpu_worker.py` | `Worker.determine_available_memory` | 460 | — |
| `docs/frameworks/vllm/vllm-overview.md` | runner-execute | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.execute_model` | 4166 | — |
| `docs/frameworks/vllm/vllm-overview.md` | runner-load | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.load_model` | 5303 | — |
| `docs/frameworks/vllm/vllm-overview.md` | runner-kv-init | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.initialize_kv_cache` | 7612 | — |
| `docs/frameworks/vllm/vllm-overview.md` | runner-graph-capture | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.capture_model` | 6806 | — |
| `docs/frameworks/vllm/vllm-overview.md` | model-registry | `vllm/model_executor/models/registry.py` | `_VLLM_MODELS` | 711 | — |
| `docs/frameworks/vllm/vllm-overview.md` | vllm-config | `vllm/config/vllm.py` | `VllmConfig` | 308 | — |
| `docs/frameworks/vllm/vllm-overview.md` | attention-backend | `vllm/v1/attention/backend.py` | `AttentionBackend` | 56 | — |
| `docs/frameworks/vllm/vllm-overview.md` | attention-selector | `vllm/v1/attention/selector.py` | `get_attn_backend` | 101 | — |
| `docs/frameworks/vllm/vllm-overview.md` | model-attention-layer | `vllm/model_executor/layers/attention/attention.py` | `Attention` | 223 | — |
| `docs/frameworks/vllm/vllm-overview.md` | mla-attention-layer | `vllm/model_executor/layers/attention/mla_attention.py` | `MLAAttention` | 346 | — |
| `docs/frameworks/vllm/vllm-overview.md` | attention-forward-delegate | `vllm/model_executor/layers/attention/attention.py` | `Attention.forward` | 488 | — |
| `docs/frameworks/vllm/vllm-overview.md` | sampler | `vllm/v1/sample/sampler.py` | `Sampler` | 20 | — |
| `docs/frameworks/vllm/vllm-overview.md` | platform-interface | `vllm/platforms/interface.py` | `Platform` | 134 | — |
| `docs/frameworks/vllm/vllm-overview.md` | parallel-state | `vllm/distributed/parallel_state.py` | `init_model_parallel_group` | 1319 | — |
| `docs/frameworks/vllm/vllm-overview.md` | compiler-interface | `vllm/compilation/compiler_interface.py` | `CompilerInterface` | 27 | — |
| `docs/frameworks/vllm/vllm-overview.md` | multimodal-registry | `vllm/multimodal/registry.py` | `MultiModalRegistry` | 98 | — |

## Runtime Flow Evidence

1. API and CLI entry — `api-server`, `openai-chat-serving`, `offline-llm`.
2. Frontend admission — `async-frontend`, `input-processing`,
   `output-processing`, `detokenizer`.
3. Engine core iteration — `engine-core`, `request`, `scheduler-output`.
4. Scheduling and KV allocation — `scheduler`, `kv-cache-manager`,
   `block-pool`.
5. Worker dispatch — `executor`, `worker-base`, `gpu-worker`,
   `gpu-model-runner`, `sampler`.
6. Worker vs model runner split — `worker-execute`, `worker-memory-profile`,
   `runner-execute`, `runner-load`, `runner-kv-init`, `runner-graph-capture`.
7. Model and kernel substrate — `model-registry`, `attention-backend`,
   `attention-selector`, `multimodal-registry`.
8. Cross-cutting layers — `vllm-config`, `platform-interface`,
   `parallel-state`, `compiler-interface`.

## Layout

Top-level repository: `vllm/` (Python package), `csrc/` (C++/CUDA kernels and
torch bindings), plus `benchmarks/`, `examples/`, `tests/`, `docs/`, `docker/`,
`tools/`, `scripts/`, `rust/`, `pyproject.toml`, `setup.py`, `CMakeLists.txt`.

`vllm/` package — one directory per subsystem:

- `v1/` — the current engine (frontend, EngineCore, scheduler, KV cache,
  executor, worker, attention, sampling, spec decode, structured output).
- `model_executor/` — model definitions (`models/`, 289 files), reusable
  `layers/` (attention, fused_moe, quantization, rotary_embedding, ...),
  `model_loader/`, `kernels/`, `warmup/`, `offloader/`, `custom_op.py`.
- `models/` — newer hardware-isolated model packages (deepseek_v4, kimi_k3,
  minimax_m3, ...), distinct from the classic `model_executor/models/` registry.
- `config/` — one module per config domain plus the `VllmConfig` aggregate.
- `entrypoints/` — OpenAI/Anthropic/pooling/speech_to_text/scale_out servers,
  `cli/`, MCP server, and the offline `LLM` class.
- `engine/` — thin V0-compat shim: `llm_engine.py` re-exports the V1
  `LLMEngine`; the standalone V0 engine no longer exists at this revision.
- `compilation/` — compiler interface, CUDA graph backends, piecewise backend,
  custom passes.
- `distributed/` — parallel state (`GroupCoordinator`), device communicators,
  KV transfer, elastic expert-parallel load balancing.
- `platforms/` — `Platform` abstraction: cuda, rocm, xpu, tpu, cpu, zen_cpu.
- `multimodal/`, `inputs/`, `transformers_utils/`, `tokenizers/`,
  `tool_parsers/`, `parser/`, `reasoning/`, `renderers/` — input and content
  processing.
- `lora/` — LoRA adapters; `triton_utils/`, `kernels/` — Triton helpers and
  custom-op shims; `plugins/` — plugin loading; `usage/` — telemetry.
- `csrc/` mirrors this split natively: `attention/`, `moe/`, `quantization/`,
  `core/`, `custom_all_reduce.cuh`, `torch_bindings.cpp`, etc.

`vllm/v1/` — engine internals:

- `engine/` — `async_llm.py` (frontend), `core.py` (`EngineCore`),
  `input_processor.py`, `output_processor.py`, `detokenizer.py`,
  `core_client.py`, `llm_engine.py` (offline), `coordinator.py`.
- `core/` — `sched/` (scheduler, async scheduler, request queue, output),
  `kv_cache_manager.py`, `kv_cache_coordinator.py`,
  `single_type_kv_cache_manager.py`, `block_pool.py`, `kv_cache_utils.py`,
  `encoder_cache_manager.py`.
- `executor/` — `uniproc_executor.py`, `multiproc_executor.py`,
  `ray_executor.py`, `ray_executor_v2.py`, `abstract.py`.
- `worker/` — `worker_base.py`, `gpu_worker.py`, `gpu_model_runner.py`
  (largest file, ~7.9k lines), `cpu_worker.py`, `xpu_worker.py`,
  `block_table.py`, `dp_utils.py`, `ubatching.py`, `startup_plan.py`,
  `workspace.py`, and `gpu/` (input_batch, model_runner, model_states,
  spec_decode, mm, pool, sample, metrics, ...).
- `attention/` — `backend.py` (`AttentionBackend`), `selector.py`, `backends/`
  (flash_attn, flashinfer, triton_attn, mla/, mamba*, flex_attention,
  rocm_*, hpc_attn, ...), `ops/`.
- `sample/` — sampler, logits processors, rejection sampler, metadata, ops.
- `spec_decode/` — draft models (eagle, medusa, ngram_proposer, draft_model,
  llm_base_proposer, dynamic/, gemma4, dflash, step3p5, suffix_decoding).
- `structured_output/` — xgrammar, outlines, guidance, lm-format-enforcer.
- `pool/` — embedding/pooling (late interaction).
- `outputs.py` (SamplerOutput, ModelRunnerOutput), `request.py` (Request,
  RequestStatus), `kv_cache_interface.py` + `kv_cache_spec_registry.py`,
  `metrics/`, `fault_tolerance/`, `kv_offload/`, `simple_kv_offload/`.

## Reproduction Commands

- Model file count at this revision: `ls vllm/model_executor/models | wc -l` →
  289 entries including `__init__.py` and helper modules.
- `vllm/engine/llm_engine.py` is a one-line alias:
  `from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine`,
  `LLMEngine = V1LLMEngine`.
- Attention backends listed under `vllm/v1/attention/backends/` (26 modules
  at this revision including `__init__.py` and `registry.py`).
- `EngineCore.step` body confirmed as schedule → execute (non-blocking) →
  grammar bitmask → sample → `update_from_output` (lines 584-621).
