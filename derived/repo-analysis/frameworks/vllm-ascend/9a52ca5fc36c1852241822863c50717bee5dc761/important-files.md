---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@9a52ca5fc36c1852241822863c50717bee5dc761
commit: 9a52ca5fc36c1852241822863c50717bee5dc761
source_record: raw/frameworks/vllm-ascend-codebase--github-9a52ca5fc36c.md
generated: 2026-08-06
---

# vLLM Ascend Codebase (Qwen3.6 inference path) Important Files

Consuming page: `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md`

Inspection date: 2026-08-06. Checkout `external-repos/vllm-ascend-9a52ca5fc36c`
is clean at `9a52ca5fc36c1852241822863c50717bee5dc761` (main, upstream tip).
This revision was created with an explicit release-specific override of the
14-day revision interval because the user requested a Qwen3.6 inference-path
doc while the latest evidence snapshot (2026-08-03) was younger than the
interval; Qwen3.6-35B-A3B weekly test configs and the support-matrix/docs
updates land only on this upstream tip.

## Evidence Map

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

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | model-type-reuse | `vllm_ascend/quantization/modelslim_config.py` | `packed_modules_model_mapping` (qwen3_5 / qwen3_5_moe) | 69 | 100 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | spec-rewrite | `vllm_ascend/patch/platform/patch_speculative_config.py` | qwen3_5/qwen3_5_moe → qwen3_5_mtp rewrite | 106 | 114 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | decoder-patch | `vllm_ascend/patch/worker/patch_qwen3_5.py` | `AscendQwen3_5DecoderLayer.forward` | 117 | 160 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | attn-patch | `vllm_ascend/patch/worker/patch_qwen3_5.py` | `AscendQwen3NextAttention.forward` | 65 | 114 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | gdn-patch | `vllm_ascend/patch/worker/patch_qwen3_5.py` | GDN patch targets (`_split_ba_for_tp`, `get_state_shape`, `get_attn_backend`, `forward`, `_forward_core`) | 215 | 228 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | mtp-forward | `vllm_ascend/patch/worker/patch_qwen3_5.py` | `qwen3_5_mtp_forward` | 165 | 212 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | gdn-forward | `vllm_ascend/ops/gdn.py` | `AscendGatedDeltaNetAttention.forward` | 67 | 148 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | gdn-core | `vllm_ascend/ops/gdn.py` | `AscendGatedDeltaNetAttention._forward_core` | 149 | 457 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | gdn-backend | `vllm_ascend/ops/gdn_attn_builder.py` | `AscendGDNAttentionBackend` | 804 | 807 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | gdn-metadata | `vllm_ascend/ops/gdn_attn_builder.py` | `AscendGDNAttentionMetadataBuilder` | 193 | 196 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | fia-backend | `vllm_ascend/attention/attention_v1.py` | `AscendAttentionBackend` | 72 | 136 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | fia-forward | `vllm_ascend/attention/attention_v1.py` | `AscendAttentionBackendImpl.forward` | 1604 | 1669 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | fia-op | `vllm_ascend/attention/attention_v1.py` | `forward_fused_infer_attention` | 1268 | 1400 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | fia-graph | `vllm_ascend/attention/attention_v1.py` | `full_graph_fia` | 845 | 1015 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | backend-select | `vllm_ascend/platform.py` | `NPUPlatform.get_attn_backend_cls` | 216 | 244 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | quant-config | `vllm_ascend/quantization/modelslim_config.py` | `AscendModelSlimConfig.get_quant_method` | 665 | 735 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | runner-exec | `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner.execute_model` | 1756 | 2181 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | runner-load | `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner.load_model` | 3517 | — |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | runner-capture | `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner.capture_model` | 4845 | — |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | e2e-fia | `tests/e2e/pull_request/two_card/test_qwen3_6_27b_fia.py` | `test_qwen3_6_27b_multimodel_fia_eager` | 28 | 64 |
| `docs/frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md` | weekly-config | `tests/e2e/weekly/single_node/configs/Qwen3.6-35B-A3B-w4a8-A3.yaml` | test case `Qwen3.6-35B-A3B-in16k-out1k-0-128-32` | 6 | 65 |

## Runtime Flow Evidence

End-to-end path for a Qwen3.6 request (Qwen3.6-27B multimodal or Qwen3.6-35B-A3B MoE):

1. Model identity — the HF `config.json` carries a `qwen3_5` / `qwen3_5_moe` model type, so vLLM's ModelRegistry resolves the upstream `qwen3_5.py` model class (e.g. `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForCausalLM`) with no vllm-ascend model override. Evidence: `model-type-reuse`, `spec-rewrite`.
2. Quantization wiring — `--quantization ascend` routes to `AscendModelSlimConfig`, whose `packed_modules_model_mapping` provides the qwen3_5/qwen3_5_moe packed-module layout and whose `get_quant_method` picks per-layer Ascend quant methods. Evidence: `model-type-reuse`, `quant-config`.
3. Speculative rewrite (optional) — `--speculative-config '{"method":"qwen3_5_mtp"}'` rewrites the model type to `qwen3_5_mtp` and sets `Qwen3_5MTP`/`Qwen3_5MoeMTP` architectures. Evidence: `spec-rewrite`.
4. Model construction and patch application — the patched hybrid decoder layer (`AscendQwen3_5DecoderLayer.forward`) routes GDN layers to Ascend GDN attention and full-attention layers to the Ascend attention layer; `AscendQwen3NextAttention.forward` handles the fused qkv+rmsnorm+mrope path. Evidence: `decoder-patch`, `attn-patch`, `gdn-patch`.
5. GDN linear-attention compute — `AscendGatedDeltaNetAttention.forward` runs input projections, the `torch.ops.vllm.qwen_gdn_attention_core` custom op, and output projection; `_forward_core` drives the Triton FLA chunked rule; `AscendGDNAttentionBackend` + `AscendGDNAttentionMetadataBuilder` provide the vLLM V1 attention-backend interface. Evidence: `gdn-forward`, `gdn-core`, `gdn-backend`, `gdn-metadata`.
6. Full-attention compute — `AscendAttentionBackend` (selected by `NPUPlatform.get_attn_backend_cls`) executes the FIA op (`npu_fused_infer_attention_score`) via `forward_fused_infer_attention`; during ACL-graph capture it uses `full_graph_fia`/`full_graph_fia_v2`. Evidence: `fia-backend`, `fia-forward`, `fia-op`, `fia-graph`, `backend-select`.
7. Step execution — `NPUModelRunner.execute_model` runs the model forward and sampling; `load_model` builds the runner and model; `capture_model` captures ACL graphs (e.g. `FULL_DECODE_ONLY` for the 35B-A3B weekly configs, `FULL_AND_PIECEWISE` for the 27B FIA graph test). Evidence: `runner-exec`, `runner-load`, `runner-capture`.
8. Validation — multimodal FIA e2e tests (`e2e-fia`) and weekly perf/accuracy configs (`weekly-config`) exercise the path.

## Reproduction Commands

- `git log --oneline 32a59d4e349c12c32cdbc1916436c16e39939afc..9a52ca5fc36c1852241822863c50717bee5dc761` — 99 commits between the previously pinned vllm-ascend revision and this one.
- `grep -ril 'qwen3.6' vllm_ascend/` — no matches: vllm-ascend contains no Qwen3.6-specific model code; support reuses the `qwen3_5` family.
- `git ls-tree -r --name-only 9a52ca5fc36c… | grep -i 'qwen3\.6'` — Qwen3.6 artifacts are docs, the FIA e2e test, and weekly configs only.
- `git diff --stat 32a59d4e349c… 9a52ca5fc36c… -- vllm_ascend/ | grep -i qwen` — Qwen3.6-relevant code deltas are `patch_qwen3_5.py` (+51) and `modelslim_config.py` (+2); both are vLLM-version/compat refactors, not new Qwen3.6 logic.
- `--speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}'` and `--enable-expert-parallel` appear verbatim in `Qwen3.6-35B-A3B-w4a8-A3.yaml`.
