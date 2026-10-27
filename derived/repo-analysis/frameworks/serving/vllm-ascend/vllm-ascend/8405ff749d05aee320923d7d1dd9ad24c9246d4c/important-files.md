---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@8405ff749d05aee320923d7d1dd9ad24c9246d4c
commit: 8405ff749d05aee320923d7d1dd9ad24c9246d4c
source_record: raw/frameworks/serving/vllm-ascend/vllm-ascend-codebase--github-8405ff749d05.md
generated: 2026-10-27
---

# vLLM Ascend Codebase (DeepSeek-V4 C4/C128 main-KV compression) Important Files

## Evidence Map

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | ratio-assignment | `vllm_ascend/models/deepseek_v4/model.py` | Per-layer `compress_ratio` assignment | 570 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | compressor-for-all-compressed-ratios | `vllm_ascend/models/deepseek_v4/model.py` | Compressor construction when ratio > 1 | 616 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | c4-only-indexer-owner | `vllm_ascend/models/deepseek_v4/model.py` | C4-only Indexer construction | 627 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | compressor-kernel-ratio | `vllm_ascend/models/deepseek_v4/compressor.py` | Compressor custom-op `cmp_ratio` argument | 226 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | c4-main-kv-write | `vllm_ascend/attention/dsa_v1.py` | C4 compressed-main-KV scatter callback | 2652 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | c128-main-kv-write | `vllm_ascend/attention/dsa_v1.py` | Non-C4 compressed-main-KV write | 2687 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | compressed-kv-read | `vllm_ascend/attention/dsa_v1.py` | Compressed `cmp_kv` passed to sparse attention | 2834 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | c4-only-topk-read | `vllm_ascend/attention/dsa_v1.py` | C4-only `cmp_sparse_indices` argument | 2838 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | cp-main-kv-compressor | `vllm_ascend/attention/context_parallel/dsa_cp.py` | DSA-CP main-KV compressor call | 1927 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | cp-main-kv-write | `vllm_ascend/attention/context_parallel/dsa_cp.py` | DSA-CP compressed-main-KV scatter | 1950 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | cp-c4-topk-branch | `vllm_ascend/attention/context_parallel/dsa_cp.py` | C4 branch passes top-k compressed indices | 2002 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | cp-c128-compressed-read | `vllm_ascend/attention/context_parallel/dsa_cp.py` | C128 branch reads compressed main KV without top-k indices | 2022 | — |

## Reproduction Commands

Record exact read-only search or counting commands here when the docs make
quantitative codebase claims.
