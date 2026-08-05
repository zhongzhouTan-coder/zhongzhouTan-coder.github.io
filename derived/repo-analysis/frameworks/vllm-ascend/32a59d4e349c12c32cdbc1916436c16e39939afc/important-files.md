---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@32a59d4e349c12c32cdbc1916436c16e39939afc
commit: 32a59d4e349c12c32cdbc1916436c16e39939afc
source_record: raw/frameworks/vllm-ascend-codebase--github-32a59d4e349c.md
generated: 2026-08-03
---

# vLLM-Ascend 32a59d4e349c Important Files

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm-ascend/architecture.md` | plugin-entry | `setup.py` | `entry_points` | 543 | — |
| `docs/frameworks/vllm-ascend/architecture.md` | platform-switchboard | `vllm_ascend/platform.py` | `NPUPlatform` | 127 | — |
| `docs/frameworks/vllm-ascend/architecture.md` | fused-moe-patch | `vllm_ascend/patch/platform/patch_fused_moe.py` | `_ascend_FusedMoE` | 45 | — |
| `docs/frameworks/vllm-ascend/architecture.md` | distributed-adaptation | `vllm_ascend/patch/platform/patch_distributed.py` | `communication_adaptation_310p` | 33 | — |
| `docs/frameworks/vllm-ascend/architecture.md` | parallel-groups | `vllm_ascend/distributed/parallel_state.py` | `init_model_parallel_group` | 86 | — |
| `docs/frameworks/vllm-ascend/architecture.md` | compile-backend | `vllm_ascend/compilation/compiler_interface.py` | `compile_fx` | 39 | — |
| `docs/frameworks/vllm-ascend/architecture.md` | mla-op | `vllm_ascend/ops/mla.py` | `AscendMultiHeadLatentAttention` | 66 | — |
| `docs/frameworks/vllm-ascend/architecture.md` | dsa-op | `vllm_ascend/ops/dsa.py` | `AscendDeepseekSparseAttention` | 61 | — |
| `docs/frameworks/vllm-ascend/architecture.md` | linear-op | `vllm_ascend/ops/linear.py` | `unquantized_gemm` | 53 | — |
| `docs/frameworks/vllm-ascend/architecture.md` | rotary-op | `vllm_ascend/ops/rotary_embedding.py` | `set_cos_and_sin` | 63 | — |
| `docs/frameworks/vllm-ascend/architecture.md` | rope-v4-op | `vllm_ascend/ops/rope_dsv4.py` | `RopeGlobalState` | 12 | — |
| `docs/frameworks/vllm-ascend/architecture.md` | rmsnorm-op | `vllm_ascend/ops/layernorm.py` | `AscendRMSNorm` | 28 | — |
| `docs/frameworks/vllm-ascend/architecture.md` | activation-op | `vllm_ascend/ops/activation.py` | `AscendQuickGELU` | 29 | — |
| `docs/frameworks/vllm-ascend/architecture.md` | vocab-embedding-op | `vllm_ascend/ops/vocab_parallel_embedding.py` | `AscendVocabParallelEmbedding` | 45 | — |
| `docs/frameworks/vllm-ascend/architecture.md` | attn-backend-selection | `vllm_ascend/platform.py` | `get_attn_backend_cls` | 796 | 822 |
| `docs/frameworks/vllm-ascend/architecture.md` | sfa-sparse-flag | `vllm_ascend/utils.py` | `model_uses_sfa_sparse` | 111 | 119 |
| `docs/frameworks/vllm-ascend/architecture.md` | use-sparse-flag | `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner.__init__` | 351 | 356 |
| `docs/frameworks/vllm-ascend/architecture.md` | sfa-backend | `vllm_ascend/attention/sfa_v1.py` | `AscendSFABackend` | 112 | 113 |
| `docs/frameworks/vllm-ascend/architecture.md` | dsa-backend | `vllm_ascend/attention/dsa_v1.py` | `AscendDSABackend` | 191 | 192 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | v4-indexer-cache | `vllm_ascend/models/deepseek_v4.py` | `AscendDeepseekV4IndexerCache` | 143 | 166 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | v4-indexer-module | `vllm_ascend/models/deepseek_v4.py` | `Indexer` | 531 | 605 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | v4-indexer-layers | `vllm_ascend/models/deepseek_v4.py` | `DeepseekV4Attention.__init__` (indexer creation) | 820 | 855 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | c4-c128-groups | `vllm_ascend/patch/platform/patch_kv_cache_utils.py` | `_get_kv_cache_groups_uniform_groups` | 110 | 130 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | c8-dtype | `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner.__init__` (c8 dtype) | 357 | 366 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | c8-config-flags | `vllm_ascend/ascend_config.py` | `AscendConfig.__init__` (li/sfa c8) | 249 | 258 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | c8-layer-filter | `vllm_ascend/ascend_config.py` | `_parse_sparse_li_c8_layers_from_quant_config` | 376 | 399 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | c8-is-layer | `vllm_ascend/ascend_config.py` | `is_sparse_li_c8_layer` | 401 | 419 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | sfa-c8-setup | `vllm_ascend/attention/sfa_v1.py` | `AscendSFAImpl.__init__` (c8 dtypes) | 602 | 611 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | quantize-query | `vllm_ascend/device/device_op.py` | `indexer_quantize_query` (non-A5) | 703 | 708 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | quant-scatter | `vllm_ascend/device/device_op.py` | `indexer_quant_scatter` (non-A5) | 711 | 728 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | a5-quant-path | `vllm_ascend/device/device_op.py` | `indexer_quantize_query`/`indexer_quant_scatter` (A5) | 1486 | 1510 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | indexer-weights-fp16 | `vllm_ascend/device/device_op.py` | `prepare_dsa_indexer_weights` | 767 | 769 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | indexer-key-scale | `vllm_ascend/device/device_op.py` | `prepare_dsa_indexer_key_scale` | 777 | 778 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | sfa-select-postprocess | `vllm_ascend/device/device_op.py` | `indexer_select_post_process` | 459 | 519 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | qli-metadata-prefill | `vllm_ascend/attention/dsa_v1.py` | `build_prefill_metadata` (qli metadata) | 895 | 923 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | qli-prefill | `vllm_ascend/attention/dsa_v1.py` | `_forward_prefill` (quant indexer) | 2166 | 2184 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | qli-decode | `vllm_ascend/attention/dsa_v1.py` | `_forward_decode` (quant indexer) | 2464 | 2482 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | qli-postdecode | `vllm_ascend/attention/dsa_v1.py` | `_indexer_qli` (post-decode) | 2732 | 2750 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | op-registration | `csrc/torch_binding.cpp` | `npu_vllm_quant_lightning_indexer` registration | 2400 | 2415 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | li-kernel | `csrc/attention/lightning_indexer_vllm/op_kernel/lightning_indexer_vllm.cpp` | `lightning_indexer_vllm` | 38 | 57 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | li-metadata-kernel | `csrc/attention/vllm_quant_lightning_indexer_metadata/op_kernel_aicpu/vllm_quant_lightning_indexer_metadata_aicpu.cpp` | `VllmQuantLightningIndexerMetadataCpuKernel::CalcSplitInfo` | 300 | 320 |
| `docs/frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md` | dense-c8-kv | `vllm_ascend/quantization/methods/kv_c8.py` | `AscendC8KVCacheAttentionMethod` | 108 | 160 |

## Runtime Flow Evidence

1. Discovery and registration — `plugin-entry`.
2. Platform switchboard — `platform-switchboard`.
3. Model and patch integration — `fused-moe-patch`, `distributed-adaptation`, `parallel-groups`.
4. Graph capture and compilation — `compile-backend`.
5. Custom kernel ops — `mla-op`, `dsa-op`, `linear-op`, `rotary-op`, `rope-v4-op`, `rmsnorm-op`, `activation-op`, `vocab-embedding-op`.
6. DeepSeek-V4 Lightning Indexer C8 runtime flow — `v4-indexer-cache`, `v4-indexer-module`, `v4-indexer-layers` → `c8-dtype`, `sfa-c8-setup` → `quantize-query`, `quant-scatter`, `a5-quant-path`, `indexer-weights-fp16`, `indexer-key-scale` → `qli-metadata-prefill`, `qli-prefill`, `qli-decode`, `qli-postdecode` → `op-registration`, `li-kernel`, `li-metadata-kernel`. The SFA (V3.2) branch additionally routes through `sfa-select-postprocess` and the per-layer gate `c8-config-flags`, `c8-layer-filter`, `c8-is-layer`; the dense-attention C8 KV cache scheme `dense-c8-kv` is a separate feature kept out of scope.

## Reading Scope

Static inspection of the vllm-ascend plugin architecture and integration
surface with upstream vLLM, covering plugin registration, platform
abstraction, monkey-patches, custom backends (attention, communication, ops),
ACL graph capture, and model registry overrides. A second scoped pass covers
the DeepSeek-V4 Lightning Indexer 8-bit ("C8") quantization path: model-side
indexer cache construction and dtypes, operator-level quant/scatter helpers
for both A5 and non-A5 device types, the quantized top-k custom operators
(`lightning_indexer_vllm`, `vllm_quant_lightning_indexer_metadata`), the
per-layer C8 gate driven by quant description, and the DSV4 DSA prefill/decode
call sites. SFA (DeepSeek-V3.2) indexer C8 and dense-model C8 KV cache
(`kv_c8.py`) are mapped for contrast but are not the focus.

## Entry Files

- `vllm_ascend/__init__.py` — five plugin entry points: `register()`,
  `register_model()`, `register_connector()`, `register_model_loader()`,
  `register_service_profiling()`; all call `_ensure_global_patch()` first.
- `vllm_ascend/platform.py` — `NPUPlatform(Platform)` with `PlatformEnum.OOT`,
  `device_type="npu"`, `simple_compile_backend="eager"`, `AscendCompiler`
  backend, `CaMemAllocator` support, and Ascend-specific SP/config defaults.
- `vllm_ascend/worker/worker.py` — `NPUWorker(WorkerBase)` with
  `CaMemAllocator` init, sleep/wakeup lifecycle, custom op registration, and
  weight transfer engine setup.
- `vllm_ascend/worker/model_runner_v1.py` — `NPUModelRunner(GPUModelRunner)`
  with `ACLGraphWrapper`, Ascend attention backends, contiguous 2M-aligned KV
  cache, and custom sampler.
- `vllm_ascend/attention/attention_v1.py` — `AscendAttentionBackend` for GQA/MQA
  using `torch_npu.npu_fused_infer_attention_score` (FIA).
- `vllm_ascend/attention/mla_v1.py` — `AscendMLABackend` for DeepSeek MLA using
  MLAPO (MLA Prefill Operator).
- `vllm_ascend/attention/dsa_v1.py` — `AscendDSABackend` for DeepSeek Sparse
  Attention (V4 sparse tokens).
- `vllm_ascend/attention/sfa_v1.py` — `AscendSparseFlashAttention` for V4 dense
  query tokens.
- `vllm_ascend/attention/context_parallel/` — decode context-parallel variants
  (`AscendAttentionDCP`, `AscendDSACP`, `AscendSFADCP`).
- `vllm_ascend/compilation/acl_graph.py` — `ACLGraphWrapper` around
  `torch_npu.npu.warp_graph` with FULL/PIECEWISE/NONE modes and
  weak-reference workspace cleanup.
- `vllm_ascend/compilation/compiler_interface.py` — `AscendCompiler` custom
  compile backend using `aot_autograd` + Ascend `PassManager`.
- `vllm_ascend/distributed/device_communicators/pyhccl.py` —
  `PyHcclCommunicator` wrapping HCCL C API for custom all-reduce.
- `vllm_ascend/distributed/device_communicators/npu_communicator.py` —
  `NPUCommunicator` with HCCL `all_to_all`.
- `vllm_ascend/distributed/parallel_state.py` — Ascend-specific TP groups:
  `_MC2`, `_MLP_TP`, `_OTP`, `_LMTP`, `_EMBED_TP`, `_P_TP`, `_DYNAMIC_EPLB`.
- `vllm_ascend/ops/mla.py`, `vllm_ascend/ops/dsa.py` — DeepSeek MLA and sparse
  attention kernel ops.
- `vllm_ascend/ops/linear.py`, `vllm_ascend/ops/linear_op.py` — fractal-format
  Ascend linear layers.
- `vllm_ascend/ops/rotary_embedding.py`, `vllm_ascend/ops/rope_dsv4.py` — RoPE
  with DeepSeek scaling and V4 complex-exponential kernels.
- `vllm_ascend/ops/layernorm.py`, `vllm_ascend/ops/activation.py` — custom
  RMSNorm, AscendQuickGELU, AscendSiluAndMul.
- `vllm_ascend/ops/fused_moe/fused_moe.py` — `AscendMoERunner` replacing
  upstream `FusedMoE`.
- `vllm_ascend/ops/register_custom_ops.py` — `torch.ops._C_ascend.*` custom
  op registry with dummy fusion op fallbacks.
- `vllm_ascend/patch/platform/` — FusedMoE factory replacement, KV connector,
  parallel state extension, distributed init, KV cache coordinator, speculative
  config.
- `vllm_ascend/patch/worker/` — CUDA graph → ACL graph redirection, DeepSeek
  MTP, Eagle3, FP8, Triton, model-specific paths (Qwen3, MiniMax M2, Kimi
  K2.5).
- `vllm_ascend/device_allocator/cumem_allocator.py` — `CaMemAllocator`
  pluggable NPU memory allocator with sleep/wakeup support.
- `vllm_ascend/quantization/` — Ascend fp8 and compressed-tensors methods.
- `vllm_ascend/models/deepseek_v4.py` — `AscendDeepseekV4ForCausalLM` model
  override via `ModelRegistry`.
- `setup.py` — `entry_points` declaring all five plugin hooks.

## Reproduction Commands

```bash
git check-ignore external-repos/vllm-ascend
git -C external-repos/vllm-ascend remote get-url origin
git -C external-repos/vllm-ascend rev-parse HEAD
git -C external-repos/vllm-ascend branch --show-current
git -C external-repos/vllm-ascend status --porcelain
rg -n "class NPUPlatform|class NPUWorker|class NPUModelRunner" external-repos/vllm-ascend/vllm_ascend
rg -n "def register\b|def register_model\b|def _ensure_global_patch" external-repos/vllm-ascend/vllm_ascend/__init__.py
rg -n "class Ascend.*Backend" external-repos/vllm-ascend/vllm_ascend/attention
rg -n "class ACLGraphWrapper|class AscendCompiler" external-repos/vllm-ascend/vllm_ascend/compilation
rg -n "class PyHcclCommunicator|class NPUCommunicator" external-repos/vllm-ascend/vllm_ascend/distributed
rg -n "entry_points" external-repos/vllm-ascend/setup.py
```
