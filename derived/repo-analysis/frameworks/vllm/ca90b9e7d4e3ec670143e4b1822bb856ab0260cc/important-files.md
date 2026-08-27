---
kind: repository-analysis
repository_id: github:vllm-project/vllm@ca90b9e7d4e3ec670143e4b1822bb856ab0260cc
commit: ca90b9e7d4e3ec670143e4b1822bb856ab0260cc
source_record: raw/frameworks/vllm-codebase--github-ca90b9e7d4e3.md
generated: 2026-08-27
---

# vLLM Codebase (weight-loading lifecycle for Qwen3.8 FP8 to MXFP8) Important Files

## Evidence Map

The consuming page is `docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md`.

| docs page | finding | file | symbol | start | end |
|---|---|---|---|---:|---:|
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | quant-config | vllm/model_executor/model_loader/weight_utils.py | get_quant_config | 240 | 287 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | checkpoint-load | vllm/model_executor/model_loader/default_loader.py | DefaultModelLoader.load_weights | 414 | 427 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | load-lifecycle | vllm/model_executor/model_loader/base_loader.py | BaseModelLoader.load_model | 42 | 80 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | postload-dispatch | vllm/model_executor/model_loader/utils.py | process_weights_after_loading | 97 | 123 |

## Reproduction Commands

Record exact read-only search or counting commands here when the docs make
quantitative codebase claims.
