---
kind: repository-analysis
repository_id: github:vllm-project/vllm@9e8b3fb83ed4f3da1ed5376083027b14994939fc
commit: 9e8b3fb83ed4f3da1ed5376083027b14994939fc
source_record: raw/frameworks/serving/vllm/vllm-codebase--github-9e8b3fb83ed4.md
generated: 2026-11-26
---

# DeepSeek V4.1 SM120 HiSparse Boundary (2026-11-25 Pin)

Static evidence map for the later vLLM pin. HiSparse now describes DSA and QSA
models, but its integration still belongs to the generic sparse-MLA layer and
the SM120 backend explicitly declines it.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | current-hisparse-scope | `vllm/config/attention.py` | `HiSparseConfig` | 21 | 22 |
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | current-config-gate | `vllm/config/vllm.py` | `VllmConfig.__post_init__` HiSparse DSA/QSA validation | 2031 | 2040 |
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | current-v41-sm120 | `vllm/models/deepseek_v41/nvidia/model.py` | `_select_dsv4_attn_cls` SM12 branch | 298 | 299 |
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | current-v41-own-layer | `vllm/models/deepseek_v41/sparse_mla.py` | `DeepseekV4SparseMLABackend` | 98 | 131 |
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | current-v41-sm120-class | `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py` | `DeepseekV4FlashInferSM120Attention` | 575 | 580 |
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | current-default-false | `vllm/v1/attention/backend.py` | `AttentionBackend.supports_hisparse` | 229 | 231 |
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | current-sm120-disabled | `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py` | `FlashInferMLASparseSM120Backend.supports_hisparse` | 207 | 209 |
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | current-sm120-latent-branch | `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` | `FlashInferMLASparseSM120Impl.forward_mqa` HiSparse branch | 127 | 142 |
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | current-platform | `vllm/platforms/cuda.py` | `CudaPlatformBase.get_hisparse_backend_cls` | 274 | 276 |
| `docs/frameworks/serving/vllm/deepseek/v4.1-hisparse-sm120/index.md` | current-kernel-scope | `csrc/hisparse_kernels.cu` | HiSparse kernel header | 4 | 4 |

## Findings

- The later config docstring broadens HiSparse to DSA and QSA, and the global
  gate accepts DSA models with `index_topk`; DeepSeek V4.1 satisfies that check.
- DeepSeek V4.1 still selects its own attention layer and backend base; the
  backend base inherits the default `supports_hisparse() == False`.
- The generic SM120 sparse-MLA backend explicitly returns `False` from
  `supports_hisparse`, so the generic selector rejects HiSparse on SM120 even
  though the SM120 implementation file contains a HiSparse branch.
- CUDA advertises a HiSparse backend class platform-wide; the model and backend
  layers, not the platform check, determine whether the path is reachable.

## Reproduction Commands

```bash
git grep -n -i hisparse HEAD -- vllm/models/deepseek_v41
git grep -n 'supports_hisparse' HEAD -- vllm/v1/attention
git grep -n 'hisparse' HEAD -- vllm/config vllm/platforms
```

## Link Completion

- [ ] Every Required Code Evidence row has a matching code link.
- [ ] The consuming page distinguishes the platform capability from the
      model-layer and backend-layer gates.
