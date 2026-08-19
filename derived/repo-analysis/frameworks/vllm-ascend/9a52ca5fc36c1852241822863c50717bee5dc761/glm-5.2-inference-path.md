---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@9a52ca5fc36c1852241822863c50717bee5dc761
commit: 9a52ca5fc36c1852241822863c50717bee5dc761
source_record: raw/frameworks/vllm-ascend-codebase--github-9a52ca5fc36c.md
generated: 2026-08-18
---

# GLM-5.2 vLLM-Ascend Backend Evidence

Consuming page: `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md`

Inspection date: 2026-08-18. The checkout is clean at
`9a52ca5fc36c1852241822863c50717bee5dc761`. A scoped refresh compared the GLM
tutorial, SFA backend, NPU runner, patches, quantization, distributed paths,
speculative decoding, and tests with upstream
`2515e80d46843812063e176fd73fd0fe1644c71b`. Relevant changes exist, but the
repository's 14-day evidence interval deferred a new revision until
2026-08-20, so this note deliberately remains pinned to `9a52ca5fc36c`.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | deployment-contract | `docs/source/tutorials/models/GLM5.2.md` | single-node GLM-5.2 command | 113 | 145 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | backend-selection | `vllm_ascend/platform.py` | `NPUPlatform.get_attn_backend_cls` | 216 | 242 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | shared-indexer-init | `vllm_ascend/patch/worker/patch_deepseek_v2.py` | `_should_skip_indexer_init` and indexer construction | 36 | 54 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | sfa-construction | `vllm_ascend/attention/sfa_v1.py` | `AscendSFAImpl.__init__` GLM and cache flags | 533 | 647 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | sfa-metadata | `vllm_ascend/attention/sfa_v1.py` | `AscendSFAMetadataBuilder._build` | 337 | 481 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | runner-forward | `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner.execute_model` forward and logits | 2079 | 2179 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | sfa-forward | `vllm_ascend/attention/sfa_v1.py` | `AscendSFAImpl.forward` | 1811 | 2056 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | lightning-indexer | `vllm_ascend/device/device_op.py` | `DeviceOperator.indexer_select_post_process` | 454 | 521 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | sparse-attention-op | `vllm_ascend/device/device_op.py` | `DeviceOperator.execute_sparse_flash_attention_process` | 523 | 584 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | ascend-moe | `vllm_ascend/ops/fused_moe/routed_experts.py` | `AscendFusedMoE.forward_impl` | 443 | 524 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | target-sampling | `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner.sample_tokens` | 2230 | 2340 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | mtp-adaptation | `vllm_ascend/patch/worker/patch_deepseek_mtp.py` | GLM MTP rotation and weight loading | 26 | 80 |
| `docs/frameworks/vllm-ascend/glm-5.2-inference-path.md` | quant-dispatch | `vllm_ascend/quantization/modelslim_config.py` | `AscendModelSlimConfig.get_quant_method` | 665 | 734 |

## Runtime Findings

1. `glm_moe_dsa` resolves to the upstream DeepSeek-V2-family model shell, while
   the Ascend platform selects SFA for the `(use_mla=True, use_sparse=True,
   use_compress=False)` capability tuple.
2. GLM-5.2's `indexer_types` can mark a layer as `shared`. Those layers omit a
   local Indexer and consume top-k indices written into a shared buffer by an
   owning layer. This differs from GLM-5.1's runtime IndexCache override, where
   each checkpoint layer still has Indexer weights.
3. SFA builds query, sequence-length, block-table, slot, and RoPE metadata;
   projects compressed queries/KV; writes paged KV and optional C8 indexer
   caches; computes or reuses 2,048 top-k token indices; and invokes the sparse
   attention operator before value-up and output projection.
4. The Ascend MoE path dispatches tokens to routed experts, applies the selected
   quantized or unquantized expert scheme, and finalizes the configured
   collective; shared experts may overlap on a separate NPU stream.
5. The common tutorial configuration uses TP=8, EP enabled, DSA context
   parallelism, LI C8, full-decode graph mode, and three-token DeepSeek MTP.

## Verification Boundary

Static code and checked-in deployment/test evidence only. No model weights,
Ascend NPU, CANN runtime, HCCL fabric, ACL graph, sparse kernel, quantized
checkpoint, or multi-node deployment was executed. The deferred freshness
decision means implementation claims describe the pinned revision, not current
`main`.
