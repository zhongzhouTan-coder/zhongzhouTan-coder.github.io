---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@cc10dddf5b5bba8b81a5c80d657ca388c2fd9de2
commit: cc10dddf5b5bba8b81a5c80d657ca388c2fd9de2
source_record: raw/frameworks/serving/vllm-ascend/vllm-ascend-codebase--github-cc10dddf5b5b.md
generated: 2026-11-25
---

# vLLM Ascend HiSparse Runtime Important Files

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/serving/vllm-ascend/deepseek-v4.1-hisparse.md` | ascend-platform-registration | `vllm_ascend/platform.py` | `NPUPlatform.get_hisparse_backend_cls` | 305 | 307 |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4.1-hisparse.md` | sfa-hisparse-builder | `vllm_ascend/attention/sfa_v1.py` | `AscendSFABackend.get_builder_cls` | 378 | 382 |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4.1-hisparse.md` | sfa-hisparse-impl | `vllm_ascend/attention/sfa_v1.py` | `AscendSFABackend.get_impl_cls` | 444 | 448 |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4.1-hisparse.md` | v41-dsa-selection | `vllm_ascend/models/layer/attention/layer.py` | `DSAAttention.__init__` | 120 | 163 |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4.1-hisparse.md` | dsa-no-hisparse-branch | `vllm_ascend/attention/dsa_v1.py` | `AscendDSABackend.get_impl_cls` | 351 | 357 |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4.1-hisparse.md` | v41-test-boundary | `tests/ut/models/test_deepseek_v41_cache.py` | `runtime` fixture | 150 | 158 |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4.1-hisparse.md` | ascend-hisparse-init | `vllm_ascend/attention/sfa_hisparse.py` | `AscendHiSparseImpl.__init__` | 212 | 254 |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4.1-hisparse.md` | ascend-swap-in-path | `vllm_ascend/attention/sfa_hisparse.py` | `AscendHiSparseImpl._execute_sparse_flash_attention_process` | 701 | 760 |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4.1-hisparse.md` | ascend-backend-adapter | `vllm_ascend/hisparse/backend.py` | `AscendHiSparseBackend` | 109 | 215 |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4.1-hisparse.md` | mapped-host-registration | `vllm_ascend/hisparse/host_pool.py` | `MappedHostTensor.__init__` | 29 | 89 |

## Request Round-Trip Evidence

This page is a compatibility trace rather than a request round trip.

1. Configuration reaches the attention backend — sfa-hisparse-builder.
2. Generic SFA selects HiSparse — sfa-hisparse-impl and ascend-hisparse-init.
3. DeepSeek V4.1 instead selects its dedicated DSA implementation — v41-dsa-selection and dsa-no-hisparse-branch.
4. The supported SFA branch resolves and stages selected rows — ascend-swap-in-path.
5. The platform backend supplies NPU and mapped-host operations — ascend-platform-registration, ascend-backend-adapter, and mapped-host-registration.

## Reproduction Commands

```bash
git log --all --reverse --format='%H %cI %s' -- \
  vllm_ascend/hisparse \
  vllm_ascend/attention/sfa_hisparse.py
rg 'hisparse_config|AscendHiSparse|get_impl_cls' \
  vllm_ascend/attention/sfa_v1.py \
  vllm_ascend/attention/dsa_v1.py \
  vllm_ascend/models/layer/attention/layer.py
```

## Link Completion

- [ ] Every Required Code Evidence row has a matching code link.
- [ ] Every selected implementation boundary is represented in the docs page.
- [ ] The page states that no NPU runtime was exercised.
