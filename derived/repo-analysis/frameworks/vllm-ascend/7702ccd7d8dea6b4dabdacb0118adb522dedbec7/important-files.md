---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@7702ccd7d8dea6b4dabdacb0118adb522dedbec7
commit: 7702ccd7d8dea6b4dabdacb0118adb522dedbec7
source_record: raw/frameworks/vllm-ascend-codebase--github-7702ccd7d8de.md
generated: 2026-08-27
---

# vLLM Ascend Codebase (Qwen3.8-27B FP8 to MXFP8 on Ascend 950) Important Files

## Evidence Map

The consuming page is `docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md`.

| docs page | finding | file | symbol | start | end |
|---|---|---|---|---:|---:|
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | fp8-config | vllm_ascend/quantization/fp8_config.py | AscendFp8Config | 30 | 113 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | source-format | vllm_ascend/quantization/methods/fp8_block.py | module docstring | 17 | 33 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | resolve-source-scales | vllm_ascend/quantization/methods/fp8_block.py | resolve_block_scales | 59 | 97 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | mx-requantize | vllm_ascend/quantization/methods/fp8_block.py | _mx_quantize | 100 | 106 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | 950-gate | vllm_ascend/quantization/methods/fp8_block.py | AscendFp8BlockLinearMethod.__init__ | 123 | 126 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | linear-postload | vllm_ascend/quantization/methods/fp8_block.py | AscendFp8BlockLinearMethod.process_weights_after_loading | 146 | 175 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | mxfp8-linear-runtime | vllm_ascend/quantization/methods/w8a8_mxfp8.py | AscendW8A8MXFP8DynamicLinearMethod.apply | 79 | 120 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | mxfp8-linear-layout | vllm_ascend/quantization/methods/w8a8_mxfp8.py | AscendW8A8MXFP8DynamicLinearMethod.process_weights_after_loading | 122 | 175 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | moe-postload | vllm_ascend/quantization/methods/fp8_block.py | AscendFp8BlockFusedMoEMethod.process_weights_after_loading | 254 | 319 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | mxfp8-moe-runtime | vllm_ascend/quantization/methods/w8a8_mxfp8.py | AscendW8A8MXFP8DynamicFusedMoEMethod.apply | 269 | 308 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | mxfp8-moe-layout | vllm_ascend/quantization/methods/w8a8_mxfp8.py | AscendW8A8MXFP8DynamicFusedMoEMethod.process_weights_after_loading | 319 | 358 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | mx-scale-algorithm | vllm_ascend/quantization/utils.py | get_dynamic_mx_quant_scale_alg | 40 | 67 |
| docs/frameworks/vllm-ascend/qwen3.8-fp8-mxfp8-950.md | direct-mxfp8-config | vllm_ascend/quantization/modelopt_mxfp8_config.py | AscendModelOptMxFp8Config | 35 | 91 |

## Reproduction Commands

Record exact read-only search or counting commands here when the docs make
quantitative codebase claims.
