---
kind: repository-analysis
repository_id: github:vllm-project/vllm@9e8b3fb83ed4f3da1ed5376083027b14994939fc
commit: 9e8b3fb83ed4f3da1ed5376083027b14994939fc
source_record: raw/frameworks/serving/vllm/vllm-codebase--github-9e8b3fb83ed4.md
generated: 2026-11-25
---

# vLLM HiSparse Runtime Important Files

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/serving/vllm-ascend/deepseek-v4.1-hisparse.md` | hisparse-config | `vllm/config/attention.py` | `HiSparseConfig` | 21 | 35 |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4.1-hisparse.md` | hisparse-config-inference | `vllm/config/vllm.py` | `VllmConfig.__post_init__` HiSparse validation | 1998 | 2022 |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4.1-hisparse.md` | hisparse-index-group | `vllm/v1/attention/backends/mla/index_group.py` | `HiSparseMLAIndexGroup` | 165 | 240 |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4.1-hisparse.md` | hisparse-group-swap | `vllm/v1/hisparse/runtime.py` | `HiSparseRuntime.swap_in` | 1016 | 1057 |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4.1-hisparse.md` | hisparse-layout | `vllm/v1/hisparse/layout.py` | `get_hisparse_kv_cache_config` | 443 | 495 |

## Reproduction Commands

```bash
git grep -n 'hisparse_config\|class HiSparse' HEAD -- \
  vllm/config \
  vllm/v1
```

## Link Completion

- [ ] Every Required Code Evidence row has a matching code link.
- [ ] The page distinguishes the upstream SFA-compatible runtime from the DeepSeek-V4.1 DSA selector.
