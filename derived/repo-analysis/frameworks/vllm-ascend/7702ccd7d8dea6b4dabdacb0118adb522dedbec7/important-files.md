---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@7702ccd7d8dea6b4dabdacb0118adb522dedbec7
commit: 7702ccd7d8dea6b4dabdacb0118adb522dedbec7
source_record: raw/frameworks/vllm-ascend-codebase--github-7702ccd7d8de.md
generated: 2026-08-27
---

# vLLM Ascend Codebase (Qwen3.8-27B FP8 to MXFP8 on Ascend 950) Important Files

## Evidence Map

The consuming page is `docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md`.

| docs page | finding | file | symbol | start | end |
|---|---|---|---|---:|---:|
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | fp8-config | vllm_ascend/quantization/fp8_config.py | AscendFp8Config | 30 | 113 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | source-format | vllm_ascend/quantization/methods/fp8_block.py | module docstring | 17 | 33 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | resolve-source-scales | vllm_ascend/quantization/methods/fp8_block.py | resolve_block_scales | 59 | 97 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | mx-requantize | vllm_ascend/quantization/methods/fp8_block.py | _mx_quantize | 100 | 106 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | 950-gate | vllm_ascend/quantization/methods/fp8_block.py | AscendFp8BlockLinearMethod.__init__ | 123 | 126 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | linear-postload | vllm_ascend/quantization/methods/fp8_block.py | AscendFp8BlockLinearMethod.process_weights_after_loading | 146 | 175 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | mxfp8-linear-runtime | vllm_ascend/quantization/methods/w8a8_mxfp8.py | AscendW8A8MXFP8DynamicLinearMethod.apply | 79 | 120 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | mxfp8-linear-layout | vllm_ascend/quantization/methods/w8a8_mxfp8.py | AscendW8A8MXFP8DynamicLinearMethod.process_weights_after_loading | 122 | 175 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | moe-postload | vllm_ascend/quantization/methods/fp8_block.py | AscendFp8BlockFusedMoEMethod.process_weights_after_loading | 254 | 319 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | mxfp8-moe-runtime | vllm_ascend/quantization/methods/w8a8_mxfp8.py | AscendW8A8MXFP8DynamicFusedMoEMethod.apply | 269 | 308 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | mxfp8-moe-layout | vllm_ascend/quantization/methods/w8a8_mxfp8.py | AscendW8A8MXFP8DynamicFusedMoEMethod.process_weights_after_loading | 319 | 358 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | mx-scale-algorithm | vllm_ascend/quantization/utils.py | get_dynamic_mx_quant_scale_alg | 40 | 67 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | direct-mxfp8-config | vllm_ascend/quantization/modelopt_mxfp8_config.py | AscendModelOptMxFp8Config | 35 | 91 |

## Reproduction Commands

Record exact read-only search or counting commands here when the docs make
quantitative codebase claims.

## Additional consuming page: GLM-5.1 W4A4C8 on Ascend 950

The page reuses this immutable revision for the sparse SFA/LI C8 path. The
upstream freshness check on 2026-08-27 found relevant changes after this
revision, but deferred a new snapshot until 2026-09-10 under the repository's
14-day revision interval. The rows below therefore describe only the pinned
implementation, not the deferred upstream candidate.

| docs page | finding | file | symbol | start | end |
|---|---|---|---|---:|---:|
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | sparse-c8-config | vllm_ascend/ascend_config.py | AscendConfig sparse-C8 derivation | 482 | 496 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | li-c8-layer-filter | vllm_ascend/ascend_config.py | LI C8 quant-metadata parser and layer filter | 655 | 706 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | a5-hardware-alias | vllm_ascend/device/device_config.py | is_950 | 57 | 63 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | glm-a5-c8-dtype | vllm_ascend/attention/sfa_v1.py | AscendSFAImpl initialization | 503 | 520 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | sparse-cache-layout | vllm_ascend/attention/sfa_v1.py | kv_cache_indexer_k_idx / kv_cache_indexer_scale_idx | 534 | 553 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | v1-cache-spec | vllm_ascend/worker/model_runner_v1.py | NPUModelRunner.get_kv_cache_spec | 4717 | 4775 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | v2-cache-spec | vllm_ascend/worker/v2/attn_utils.py | get_kv_cache_spec | 74 | 135 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | packed-sfa-write | vllm_ascend/attention/sfa_v1.py | _store_parallel_kv | 1335 | 1355 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | li-key-quant | vllm_ascend/attention/sfa_v1.py | indexer_select_pre_process | 1148 | 1156 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | li-query-quant | vllm_ascend/attention/sfa_v1.py | indexer_select_post_process | 1220 | 1240 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | a5-kv-indexer-scatter | vllm_ascend/device/device_op.py | dsa_kv_compress_scatter / indexer_quant_scatter | 1275 | 1316 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | a5-indexer-dtypes | vllm_ascend/device/device_op.py | Lightning Indexer dtype preparation | 1352 | 1367 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | a5-indexer-topk | vllm_ascend/device/device_op.py | indexer_select_post_process | 1460 | 1527 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | sfa-forward | vllm_ascend/attention/sfa_v1.py | AscendSFAImpl.forward | 1452 | 1515 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | cache-recomposition | vllm_ascend/attention/sfa_v1.py | _compose_sfa_kv_cache | 1394 | 1450 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | li-c8-aligned-allocation | vllm_ascend/worker/v2/attn_utils.py | _allocate_sparse_c8_indexer_tensors | 492 | 540 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | dense-c8-metadata | vllm_ascend/quantization/modelslim_config.py | _add_kvcache_quant_metadata | 1050 | 1068 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | dense-c8-method | vllm_ascend/quantization/methods/kv_c8.py | AscendC8KVCacheAttentionMethod | 108 | 139 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | sparse-c8-release-contract | docs/source/user_guide/release_notes.md | v0.23.0 configuration changes and known issues | 106 | 147 |
| docs/frameworks/vllm-ascend/glm-5.1-c8-950.md | sparse-c8-doc-contract | docs/source/user_guide/configuration/additional_config.md | sparse C8 options | 63 | 65 |
