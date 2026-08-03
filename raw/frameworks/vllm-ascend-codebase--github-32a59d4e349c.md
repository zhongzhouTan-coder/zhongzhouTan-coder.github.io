---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm-ascend.git
repository_url: https://github.com/vllm-project/vllm-ascend
local_checkout: external-repos/vllm-ascend/
commit: 32a59d4e349c12c32cdbc1916436c16e39939afc
ref: main
inspected: 2026-08-03
checkout_state: clean
---

# vLLM Ascend Codebase Source Record

## Reading Scope

Static inspection of the vllm-ascend plugin architecture and integration with
upstream vLLM, covering:

- plugin entry-point registration and platform abstraction;
- NPUPlatform, NPUWorker, NPUModelRunner classes;
- attention backends (MLA, DSA, SFA, GQA, context-parallel variants);
- HCCL communication and custom collective communicators;
- ACL graph capture as CUDA graph replacement;
- monkey-patch system for adapting CUDA-coupled vLLM internals;
- custom ops (linear, RoPE, layernorm, MoE, activation, embedding);
- quantization (Ascend fp8, compressed-tensors);
- device allocator (CaMemAllocator with sleep mode);
- model registry overrides for DeepSeek V4, DSpark, MiniMax M3, etc.

## Important Entry Files

- `vllm_ascend/__init__.py` — plugin entry points: `register()`, `register_model()`, `register_connector()`, `register_model_loader()`, `register_service_profiling()`
- `vllm_ascend/platform.py` — `NPUPlatform(Platform)` core abstraction
- `vllm_ascend/worker/worker.py` — `NPUWorker(WorkerBase)` with CaMemAllocator, sleep/wakeup
- `vllm_ascend/worker/model_runner_v1.py` — `NPUModelRunner(GPUModelRunner)` with ACL graphs
- `vllm_ascend/attention/` — Ascend attention backends (mla_v1, dsa_v1, sfa_v1, attention_v1, context_parallel/)
- `vllm_ascend/ops/` — Custom Ascend ops (mla, dsa, linear, rotary_embedding, layernorm, activation, fused_moe/)
- `vllm_ascend/compilation/` — ACL graph wrapper, AscendCompiler, graph fusion passes
- `vllm_ascend/distributed/` — HCCL communicator, parallel state, KV/weight transfer
- `vllm_ascend/patch/` — Monkey-patches for vLLM internals (platform/, worker/)
- `vllm_ascend/device_allocator/` — CaMem pluggable allocator
- `vllm_ascend/quantization/` — Ascend-specific quantization methods
- `vllm_ascend/core/` — KV cache interface, custom schedulers
- `vllm_ascend/models/` — Ascend-specific model overrides via ModelRegistry
- `setup.py` — entry_points declaration for plugin discovery

## Limitations

- Static code reading only; no Ascend NPU execution or performance validation was run.
- The architecture analysis covers the plugin registration, platform abstraction, monkey-patch layer, custom backends, and execution flow; it does not exhaustively trace every code path (e.g., speculative decoding proposers, quantization calibration, device-specific compiler paths).
- This revision was inspected at `32a59d4e349c12c32cdbc1916436c16e39939afc` (main, clean). Later commits may change the integration surface.
