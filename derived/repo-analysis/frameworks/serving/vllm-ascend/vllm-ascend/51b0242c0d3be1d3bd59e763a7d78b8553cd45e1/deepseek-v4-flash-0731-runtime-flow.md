---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@51b0242c0d3be1d3bd59e763a7d78b8553cd45e1
commit: 51b0242c0d3be1d3bd59e763a7d78b8553cd45e1
source_record: raw/frameworks/serving/vllm-ascend/vllm-ascend-codebase--github-51b0242c0d3b.md
generated: 2026-10-26
---

# DeepSeek-V4-Flash-0731 Runtime Flow

## Reader Contract

- **Audience:** vLLM-Ascend serving engineers comparing the V4 target path and DSpark draft handoff on A3 and 950/A5.
- **Question:** How does a scheduled target-verification batch pass through the V4 DSA-CP path, device-selected cache plan, DSpark proposer, and worker output?
- **Mental model:** The target decoder uses per-layer sliding-window and compressed KV state; C4 layers additionally select compressed records. DSpark consumes configured target auxiliary states to propose tokens, while the target sampler determines the accepted output IDs.
- **Offline/load time:** The model registry maps the V4 target and DSpark draft architectures; attention construction assigns C4/C128 roles and cache owners.
- **Runtime:** The A3 recipe enables DSA context parallel, which updates sequence cache state and computes local-query attention; the shared KV plan chooses the physical operator/cache branch.
- **Limits:** This is static reading of the pinned 2026-10-10 commit. No A3/950 runtime was executed. A scoped upstream refresh was attempted on 2026-10-26, but the GitHub fetch timed out; this note does not claim current-main freshness.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | worker-entry | `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner.execute_model` | 2228 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | target-registration | `vllm_ascend/models/__init__.py` | V4 target registration | 32 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | dspark-registration | `vllm_ascend/models/__init__.py` | `DSparkDraftModel` registration | 52 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | layer-compression-role | `vllm_ascend/models/deepseek_v4/model.py` | `DeepseekV4Attention` compression-ratio selection | 548 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | dsa-cp-selection | `vllm_ascend/attention/dsa_v1.py` | `AscendDSABackend.get_impl_cls` | 262 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | cp-cache-attention | `vllm_ascend/attention/context_parallel/dsa_cp.py` | `AscendDSACPImpl._forward` | 1801 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | c4-topk-selection | `vllm_ascend/attention/context_parallel/dsa_cp.py` | `AscendDSACPImpl._indexer_select_topk` | 2116 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | compressed-index-handoff | `vllm_ascend/attention/context_parallel/dsa_cp.py` | `cmp_sparse_indices` argument to sparse attention | 2000 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | hardware-kv-plan | `vllm_ascend/attention/dsa_attn_kv_plan.py` | `get_dsa_attn_kv_plan` | 206 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | a3-sparse-op | `vllm_ascend/attention/dsa_attn_kv_plan.py` | non-A5 sparse-attention plan | 228 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | a5-compressor-epilog | `vllm_ascend/attention/dsa_attn_kv_plan.py` | A5 KV-compression epilog | 195 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | a5-quant-sparse-op | `vllm_ascend/attention/dsa_attn_kv_plan.py` | A5 quantized shared-KV sparse op | 258 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | a5-bf16-sparse-op | `vllm_ascend/attention/dsa_attn_kv_plan.py` | explicit A5 BF16 sparse-flash plan | 245 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | slot-coordinate-conversion | `vllm_ascend/attention/dsa_attn_kv_plan.py` | `DsaAttnKvPlan.format_dsa_slot_mapping` | 168 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | indexer-cache-dtype | `vllm_ascend/models/deepseek_v4/indexer.py` | `get_indexer_k_dtype` | 98 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | indexer-scale-dtype | `vllm_ascend/models/deepseek_v4/indexer.py` | indexer cache scale dtype | 155 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | swa-cache-spec | `vllm_ascend/models/deepseek_v4/model.py` | `AscendDeepseekV4SWACache.get_kv_cache_spec` | 129 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | compressor-state-dtype | `vllm_ascend/models/deepseek_v4/compressor.py` | FP32 compressor-state dtype | 154 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | target-auxiliary-state | `vllm_ascend/models/deepseek_v4/model.py` | `DeepseekV4Model.forward` auxiliary-state emission | 961 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | dspark-target-handoff | `vllm_ascend/worker/model_runner_v1.py` | `target_hidden_states` from auxiliary outputs | 2068 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | dspark-hidden-combine | `vllm_ascend/models/deepseek_v4/dspark.py` | `DeepseekV4DSparkModel.combine_hidden_states` | 174 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | mtp-only-hidden-override | `vllm_ascend/worker/model_runner_v1.py` | MTP target hidden-state override | 2058 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | target-rejection-sampling | `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner._sample` | 2972 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | worker-token-return | `vllm_ascend/worker/model_runner_v1.py` | `ModelRunnerOutput` construction | 2925 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | a5-prefill-oproj-gate | `vllm_ascend/attention/context_parallel/dsa_cp.py` | `enable_dsa_cp_full_o_proj` hardware gate | 1455 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | a5-prefill-oproj-branch | `vllm_ascend/attention/context_parallel/dsa_cp.py` | prefill-only full-weight gather choice | 1721 | — |

## Scope and Verification

The A3 DSA-CP selection is taken from the pinned recipe in the 2026-08-27 documentation snapshot; this note records the corresponding implementation in the 2026-10-10 code pin. The non-BF16 A5 plan, explicit-BF16 alternative, and C4 indexer path are static branch conclusions. They are not proof that the 0731 W8A8 DSpark checkpoint was run on 950.
