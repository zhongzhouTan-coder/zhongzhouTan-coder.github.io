---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm-ascend
repository_url: https://github.com/vllm-project/vllm-ascend
local_checkout: external-repos/vllm-ascend-9a52ca5fc36c/
commit: 9a52ca5fc36c1852241822863c50717bee5dc761
ref: detached
inspected: 2026-08-06
checkout_state: clean
---

# vLLM Ascend Codebase (Qwen3.5/Qwen3.6 inference path) Source Record

## Revision Note

This revision was created with `worktree.py sync --force-new-revision` as an
explicitly requested, release-specific inspection of the Qwen3.5/Qwen3.6
inference path, overriding the 14-day revision interval (previous snapshot
inspected 2026-08-03; normally eligible 2026-08-17). Qwen3.6-35B-A3B weekly
test configs and the Qwen3.6 support-matrix/docs updates exist only on this
upstream tip. Because Qwen3.6 keeps the Qwen3.5 architecture, the inspection
covers the shared `qwen3_5` / `qwen3_5_moe` family: Qwen3.5-27B, Qwen3.6-27B,
Qwen3.5/Qwen3.6-35B-A3B, and Qwen3.5-397B-A17B.

## Reading Scope

- Qwen3.5/Qwen3.6 (Qwen3.5-27B, Qwen3.6-27B, Qwen3.5/Qwen3.6-35B-A3B, Qwen3.5-397B-A17B) inference path: qwen3_5/qwen3_5_moe model-type reuse, hybrid GDN+full-attention forward, FIA and GDN attention backends, ModelSlim W8A8 quantization, qwen3_5_mtp speculative decoding, NPUModelRunner execution, and e2e tests.
- The model substrate (upstream vLLM `qwen3_5.py` model classes and the ModelRegistry architecture mapping) was verified against the pinned vLLM checkout `external-repos/vllm` at `a0c092ee72c0dcefbb3b3e74f97ac62d842e5f4b`; vllm-ascend itself contains no Qwen3.5/Qwen3.6-specific model class.

## Important Entry Files

- `vllm_ascend/patch/worker/patch_qwen3_5.py` — Qwen3.5/Qwen3.6 hybrid decoder-layer forward, AscendQwen3NextAttention, GDN split/state patches, MTP forward backport
- `vllm_ascend/ops/gdn.py` — AscendGatedDeltaNetAttention: GDN linear-attention forward via torch.ops.vllm.qwen_gdn_attention_core and Triton FLA chunks
- `vllm_ascend/ops/gdn_attn_builder.py` — AscendGDNAttentionBackend and AscendGDNAttentionMetadataBuilder for GDN linear-attention layers
- `vllm_ascend/attention/attention_v1.py` — AscendAttentionBackend: FIA full-attention op (npu_fused_infer_attention_score) and full_graph_fia/full_graph_fia_v2 ACL-graph capture
- `vllm_ascend/quantization/modelslim_config.py` — AscendModelSlimConfig with qwen3_5/qwen3_5_moe packed-module mappings and per-layer quant methods
- `vllm_ascend/patch/platform/patch_speculative_config.py` — Rewrites qwen3_5/qwen3_5_moe model_type to qwen3_5_mtp with Qwen3_5MTP/Qwen3_5MoeMTP architectures
- `vllm_ascend/platform.py` — NPUPlatform.get_attn_backend_cls: attention backend selection (FIA/MLA/SFA/DSA/FA3)
- `vllm_ascend/worker/model_runner_v1.py` — NPUModelRunner.execute_model/load_model/capture_model: step-level execution and ACL graph capture
- `tests/e2e/pull_request/two_card/test_qwen3_6_27b_fia.py` — Qwen3.6-27B multimodal e2e test with FIA op (eager and FULL_AND_PIECEWISE graph)
- `tests/e2e/weekly/single_node/configs/Qwen3.6-35B-A3B-w4a8-A3.yaml` — Qwen3.6-35B-A3B weekly perf config: ModelSlim W8A8, EP, qwen3_5_mtp spec decode, FULL_DECODE_ONLY graphs

## Limitations

- Static code reading plus upstream tutorial/test evidence; no Ascend NPU execution or runtime validation was run in this environment.
