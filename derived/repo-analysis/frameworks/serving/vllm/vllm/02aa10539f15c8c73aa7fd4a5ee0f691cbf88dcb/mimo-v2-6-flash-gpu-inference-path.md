---
kind: repository-analysis
repository_id: github:vllm-project/vllm@02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb
commit: 02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb
source_record: raw/frameworks/serving/vllm/vllm-codebase--github-02aa10539f15.md
generated: 2026-10-19
---

# MiMo-V2.6-Flash GPU Inference Evidence

## Reader Contract

- **Audience:** GPU serving engineers familiar with paged KV cache and MoE.
- **Question:** how does a plain MiMo-V2.6-Flash decode step traverse vLLM, and where do SM100 and SM120 differ?
- **Mental model:** each scheduled token crosses configured local/global attention and dense/expert layers; attention selects a KV reader separately from the expert GEMM selector.
- **Offline/load time:** architecture registry, checkpoint quantization config, layer pattern, attention and expert backend selection, and cache specs.
- **Runtime:** runner prepares slots and positions, model writes/reads KV, routes experts, produces logits, samples, and returns token IDs.
- **Limits:** static source reading of a pinned revision. The captured model card describes the family but does not prove the exact loaded HF config. No GPU, checkpoint, kernel capability, or performance test was run.

## Representation Plan

| Reader question | Evidence | Representation | Teaching job |
|---|---|---|---|
| What differs by GPU? | FA version and MXFP4 selectors | Compact table | Keep attention and expert decisions separate. |
| What happens to one decode token? | Runner, model, attention, sampler | Numbered trace and sequence diagram | Show state write, read, and return. |
| What occupies cache memory? | Checkpoint head sizes, attention spec, DiffKV layout | Record-width table | Distinguish local/global lifetime, weight precision, and KV precision. |

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | architecture | `vllm/model_executor/models/registry.py` | `model registry mapping` | 175 | — |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | layer-choice | `vllm/model_executor/models/mimo_v2.py` | `MiMoV2FlashDecoderLayer.__init__` | 375 | 421 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | layer-pattern | `vllm/model_executor/models/mimo_v2.py` | `MiMoV2FlashDecoderLayer.is_compressed_softmax_layer` | 469 | 470 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | attention-choice | `vllm/model_executor/models/mimo_v2.py` | `MiMoV2Attention.__init__` | 302 | 335 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | tp-heads | `vllm/model_executor/models/mimo_v2.py` | `MiMoV2Attention.__init__` | 241 | 255 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | cache-spec | `vllm/model_executor/layers/attention/attention.py` | `Attention.get_kv_cache_spec` | 618 | 677 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | window-reset | `vllm/model_executor/models/mimo_v2.py` | `MiMoV2FlashDecoderLayer.__init__` | 375 | 378 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | cache-bytes | `vllm/v1/kv_cache_interface.py` | `AttentionSpec.state_content_size_bytes` | 545 | 560 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | swa-retention | `vllm/v1/kv_cache_interface.py` | `SlidingWindowSpec.max_admission_blocks_per_request` | 850 | 896 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | swa-eviction | `vllm/v1/core/single_type_kv_cache_manager.py` | `SlidingWindowManager.remove_skipped_blocks` | 687 | — |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | swa-hit-window | `vllm/v1/core/single_type_kv_cache_manager.py` | `SlidingWindowManager._contiguous_blocks_for_hit` | 967 | 978 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | prefix-cacheability | `vllm/v1/kv_cache_interface.py` | `AttentionSpec.prefix_cacheable` / `prefix_replay_tokens` | 196 | 203 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | slot-map | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner._get_slot_mappings` | 4055 | 4072 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | diffkv-probe | `vllm/v1/attention/backends/flash_attn_diffkv.py` | `FlashAttentionDiffKVBackend.is_supported_on_current_device` | 43 | 65 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | fa-version | `vllm/v1/attention/backends/fa_utils.py` | `get_flash_attn_version` | 98 | 107 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | fp8-gate | `vllm/v1/attention/backends/fa_utils.py` | `flash_attn_supports_kv_cache_dtype` | 306 | 312 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | triton-dtypes | `vllm/v1/attention/backends/triton_attn_diffkv.py` | `TritonAttentionDiffKVBackend.supported_kv_cache_dtypes` | 69 | 78 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | moe-priority | `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` | `_get_priority_backends` | 425 | 449 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | trtllm-gate | `vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py` | `TrtLlmMxfp4ExpertsBase._supports_current_device` | 120 | 123 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | deepgemm-gate | `vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py` | `DeepGemmFP4Experts._supports_current_device` | 447 | 494 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | moe-select | `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` | `select_deepseek_v4_mxfp4_moe_backend` | 785 | 813 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | quant-route | `vllm/model_executor/layers/quantization/fp8.py` | `Fp8Config.get_quant_method` | 177 | 207 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | router-setup | `vllm/model_executor/models/mimo_v2.py` | `MiMoV2MoE.__init__` | 158 | 189 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | model-call | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.execute_model` | 4367 | 4392 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | cache-enumeration | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.get_kv_cache_spec` | 7394 | 7429 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | model-layers | `vllm/model_executor/models/mimo_v2.py` | `MiMoV2Model.forward` | 688 | 719 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | layer-forward | `vllm/model_executor/models/mimo_v2.py` | `MiMoV2FlashDecoderLayer.forward` | 440 | 459 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | qkv | `vllm/model_executor/models/mimo_v2.py` | `MiMoV2Attention.forward` | 341 | 356 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | attention-handoff | `vllm/model_executor/layers/attention/attention.py` | `Attention.forward` | 540 | 579 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | cache-write | `vllm/v1/attention/backends/flash_attn_diffkv.py` | `FlashAttentionDiffKVImpl.do_kv_cache_update` | 92 | 117 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | cache-read | `vllm/v1/attention/backends/flash_attn_diffkv.py` | `FlashAttentionDiffKVImpl.forward` | 218 | 271 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | triton-diffkv | `vllm/v1/attention/backends/triton_attn_diffkv.py` | `TritonAttentionDiffKVImpl.do_kv_cache_update / forward` | 129 | 220 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | moe | `vllm/model_executor/models/mimo_v2.py` | `MiMoV2MoE.forward` | 191 | 215 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | logits | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.execute_model` | 4420 | 4421 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | execute-return | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.execute_model` | 4446 | 4465 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | sample-state | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.sample_tokens` | 4496 | 4510 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | sample | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner._sample` | 3606 | 3620 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | sample-update | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.sample_tokens` | 4518 | 4523 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | return | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.sample_tokens` | 4697 | 4738 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | dflash-setup | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.__init__` | 649 | 655 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | dflash-propose | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.sample_tokens` | 4541 | 4555 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | fa-shape | `vllm/v1/attention/backends/fa_utils.py` | `get_flash_attn_version` | 193 | 209 |
| `docs/frameworks/serving/vllm/mimo-v2-6-gpu-inference-path/index.md` | weight-loader | `vllm/model_executor/models/mimo_v2.py` | `MiMoV2Model.load_weights` | 733 | 864 |

## Request Round-Trip Evidence

1. `GPUModelRunner.execute_model` accepts scheduled rows and builds group-specific physical slot mappings: `slot-map`.
2. The runner installs forward context and invokes MiMo: `model-call`.
3. MiMo embeds the token, projects Q/K/V, and advances configured local/global layers: `model-layers`, `qkv`, `layer-forward`.
4. The attention wrapper orders per-layer cache writes before reads; the chosen DiffKV backend consumes the block table and visible history: `attention-handoff`, `cache-write`, `cache-read`, `triton-diffkv`.
5. Dense/expert FFNs produce the next hidden state: `moe`.
6. The runner computes logits, stores `execute_model_state`, and returns from phase one: `logits`, `execute-return`.
7. `sample_tokens` consumes that state, samples, updates sampled-token state, and returns `ModelRunnerOutput`: `sample-state`, `sample`, `sample-update`, `return`.
8. Only with DFlash enabled, the runner creates a proposer and hands it target results after sampling: `dflash-setup`, `dflash-propose`.

## Scope and Verification

Inspected read-only `external-repos/vllm-02aa10539f15` at the clean pinned SHA above. Freshness sync on 2026-10-19 returned `decision: defer` because the latest snapshot is nine days old; next normal promotion is 2026-10-24. Commands used: `rg -n` over `mimo_v2.py`, `flash_attn_diffkv.py`, `fa_utils.py`, `mxfp4.py`, `gpu_model_runner.py`, followed by `nl -ba` line inspection. No device execution or numeric benchmark.

## Checkpoint-shape calculation

Captured Flash-MOPD model-card table: 48 layers, 39 local/9 global, Q/K width 192, V width 128, local/global KV heads 8/4, window 128. At TP4, local/global KV heads per rank are 2/1. The per-state payload calculation follows `AttentionSpec.state_content_size_bytes`: local `(192+128) × 2 = 640` elements and global `(192+128) × 1 = 320` elements. Multiply by 2 bytes for BF16 or 1 byte for FP8. At BF16 the nine global layers add `9 × 640 = 5,760` payload bytes per new token per rank; the 39 local layers write `39 × 1,280 = 49,920` payload bytes per step but reclaim old blocks. These are derived logical payloads, not measured or page-padded VRAM.
