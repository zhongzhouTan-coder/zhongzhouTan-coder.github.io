---
kind: repository-analysis
repository_id: github:vllm-project/vllm@02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb
commit: 02aa10539f15c8c73aa7fd4a5ee0f691cbf88dcb
source_record: raw/frameworks/serving/vllm/vllm-codebase--github-02aa10539f15.md
generated: 2026-11-26
---

# DeepSeek V4.1 SM120 HiSparse Boundary (1017 Snapshot)

Static evidence map for the DeepSeek-V4.1-Flash attention branch inspected at
the 2026-10-17 snapshot. The question is whether the model's SM120 path enters
vLLM's generic HiSparse host-resident cache. It does not.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | 1017-v41-selector | `vllm/models/deepseek_v41/nvidia/model.py` | `_select_dsv4_attn_cls` | 197 | 243 |
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | 1017-v41-layer | `vllm/models/deepseek_v41/attention.py` | `DeepseekV4Attention` | 176 | 186 |
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | 1017-sm120-layer | `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` | `DeepseekV4FlashInferSM120Attention` | 590 | 595 |
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | 1017-generic-binding | `vllm/model_executor/layers/attention/mla_attention.py` | `MLAAttention.__init__` HiSparse binding | 595 | 600 |
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | 1017-sm120-generic-branch | `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` | `FlashInferMLASparseSM120Impl.forward_mqa` | 122 | 142 |
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | 1017-v41-index-topk | `vllm/models/deepseek_v41/attention.py` | `DeepseekV4Attention` indexer `index_topk` read | 1300 | 1300 |
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | 1017-config-gate | `vllm/config/vllm.py` | `VllmConfig.__post_init__` HiSparse DSA validation | 1862 | 1867 |

## Findings

- The V4.1 selector chooses `DeepseekV4FlashInferSM120Attention` when the
  device capability major version is 12. The class is a `DeepseekV4Attention`
  subclass, not the generic `MLAAttention` layer.
- The generic `MLAAttention` constructor binds `hisparse_cache` only when its
  backend implementation exposes a `HiSparseMLAIndexGroup` index group.
- The generic SM120 sparse-MLA implementation contains a HiSparse branch, but
  the V4.1 SM120 attention class does not use that implementation.
- The global config validator only checks for `index_topk`; DeepSeek V4.1 has
  that field, so the configuration check alone does not prove HiSparse applies.

## Reproduction Commands

```bash
git grep -n -i hisparse HEAD -- vllm/models/deepseek_v41
git grep -n 'HiSparseMLAIndexGroup' HEAD -- \
  vllm/model_executor/layers/attention \
  vllm/v1/attention/backends/mla
```

## Link Completion

- [ ] Every Required Code Evidence row has a matching code link.
- [ ] The consuming page distinguishes the V4.1 custom layer from the generic
      sparse-MLA HiSparse path.
