---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@7702ccd7d8dea6b4dabdacb0118adb522dedbec7
commit: 7702ccd7d8dea6b4dabdacb0118adb522dedbec7
source_record: raw/frameworks/serving/vllm-ascend/vllm-ascend-codebase--github-7702ccd7d8de.md
generated: 2026-10-26
---

# DeepSeek-V4-Flash-0731 Deployment Evidence

## Reader Contract

- **Audience:** serving engineers checking which hardware/model claims are explicit in the pinned vLLM-Ascend guide.
- **Question:** What does this repository snapshot document for the July-31 DSpark checkpoint, and what does it establish only for the V4-Flash family?
- **Mental model:** The A3 guide names the 0731 W8A8 DSpark checkpoint and gives an A3 recipe; the 950 support matrix is a family-level row and does not itself prove that exact DSpark checkpoint was run there.
- **Offline/load time:** The model guide specifies a checkpoint path, quantization, and speculative-decoding method.
- **Runtime:** The single-node A3 recipe enables DSA-CP; the support matrix separately lists device-family capabilities.
- **Limits:** Documentation and test definitions are source evidence, not a runtime execution performed for this reading. The DSpark acceptance test uses a W4A8 test checkpoint.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | 0731-checkpoint-identity | `docs/source/tutorials/models/DeepSeek-V4-Flash.md` | July-31 DSpark weight identification | 28 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | dspark-a2-a3-support-statement | `docs/source/tutorials/models/DeepSeek-V4-Flash.md` | DSpark A2/A3 version statement | 42 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | a3-dspark-launch | `docs/source/tutorials/models/DeepSeek-V4-Flash.md` | A3 DSpark `vllm serve` command | 281 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | a3-dspark-seven-tokens | `docs/source/tutorials/models/DeepSeek-V4-Flash.md` | DSpark speculative config | 298 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | a3-enable-dsa-cp | `docs/source/tutorials/models/DeepSeek-V4-Flash.md` | A3 `enable_dsa_cp` setting | 306 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | 950-family-support | `docs/source/user_guide/support_matrix/supported_models.md` | DeepSeek V4-Flash 950 row | 22 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | a2-a3-experimental-status | `docs/source/user_guide/support_matrix/supported_models.md` | DeepSeek V4-Flash A2/A3 row | 31 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | 950-is-a5 | `vllm_ascend/device/hardware.py` | `device_type_from_soc_version` | 52 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | dspark-test-checkpoint | `tests/e2e/pull_request/four_card/spec_decode/test_dspark_deepseekv4.py` | test-only W4A8 model | 33 | — |
| `docs/frameworks/serving/vllm-ascend/deepseek-v4-inference.md` | dspark-dsa-cp-test-case | `tests/e2e/pull_request/four_card/spec_decode/test_dspark_deepseekv4.py` | DSA-CP DSpark parameter case | 61 | — |

## Scope and Verification

The model guide's explicit DSpark hardware statement names A2/A3. The 950 row establishes V4-Flash family support and notes native mixed MXFP8/MXFP4 weights, while also marking W8A8 in the feature matrix; it does not identify `DeepSeek-V4-Flash-0731-w8a8` as the checkpoint used for a 950 DSpark run. The test definition covers a different W4A8 test checkpoint and was not run during this documentation update.
