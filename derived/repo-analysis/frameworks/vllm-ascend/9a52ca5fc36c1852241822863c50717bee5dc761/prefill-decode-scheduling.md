---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@9a52ca5fc36c1852241822863c50717bee5dc761
commit: 9a52ca5fc36c1852241822863c50717bee5dc761
source_record: raw/frameworks/vllm-ascend-codebase--github-9a52ca5fc36c.md
generated: 2026-08-06
---

# vLLM-Ascend Prefill and Decode Evidence

Consuming page: `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md`

## Evidence Map

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md` | ascend-state | `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner._build_attn_state` | 1289 | 1316 |
| `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md` | ascend-execute | `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner.execute_model` | 1756 | 1935 |
| `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md` | attention-state | `vllm_ascend/attention/attention_v1.py` | `AscendAttentionState` | 138 | 143 |
| `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md` | metadata-build | `vllm_ascend/attention/attention_v1.py` | `AscendAttentionMetadataBuilder.build` | 287 | 375 |
| `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md` | metadata-split | `vllm_ascend/attention/attention_v1.py` | `AscendAttentionMetadataBuilder._split_decodes_and_prefills` | 260 | 264 |
| `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md` | fia-forward | `vllm_ascend/attention/attention_v1.py` | `AscendAttentionBackendImpl.forward` | 1604 | 1669 |
| `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md` | c8-mixed | `vllm_ascend/attention/attention_v1.py` | `AscendC8AttentionBackendImpl._forward_c8_chunked_prefill` | 1920 | 2019 |
| `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md` | qwen-hybrid | `vllm_ascend/patch/worker/patch_qwen3_5.py` | `AscendQwen3_5DecoderLayer.forward` | 117 | 160 |
| `docs/frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md` | gdn-forward | `vllm_ascend/ops/gdn.py` | `AscendGatedDeltaNetAttention.forward` | 67 | 148 |

## Runtime Flow Evidence

1. `NPUModelRunner.execute_model` receives the upstream `SchedulerOutput` and prepares the persistent batch and model inputs.
2. `_build_attn_state` classifies the token shape and computed-token state as prefill, decode, speculative, or chunked-prefill.
3. `AscendAttentionMetadataBuilder.build` calls the shared decode/prefill splitter and records token counts, query lengths, sequence lengths, block tables, and slots.
4. FIA attention consumes the metadata; the C8 mixed path uses paged BNSD for decode and TND for prefill.
5. The Qwen3.5 decoder patch sends full-attention layers to FIA and linear-attention layers to the Ascend GDN implementation.

## Qwen3.5 GQA Boundary

The Qwen3.5 family is hybrid. GQA describes its full-attention layers, which use the Ascend FIA backend with separate query-head and KV-head counts. GDN layers use recurrent state and a separate metadata/backend path; they are not ordinary GQA KV-cache attention.

## Limitations

Static reading of the clean pinned checkout only; no Ascend NPU execution or graph-capture validation was performed. Exact kernel behavior depends on device generation, CANN version, KV-cache dtype, graph mode, and the upstream vLLM contract paired with this revision.
