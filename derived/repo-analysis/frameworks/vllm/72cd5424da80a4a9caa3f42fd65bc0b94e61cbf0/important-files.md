---
kind: repository-analysis
repository_id: github:vllm-project/vllm@72cd5424da80a4a9caa3f42fd65bc0b94e61cbf0
commit: 72cd5424da80a4a9caa3f42fd65bc0b94e61cbf0
source_record: raw/frameworks/vllm-codebase--github-72cd5424da80.md
generated: 2026-08-04
---

# vllm Codebase Important Files

## Reading Scope

Static code reading of the MiniMax M2/M2.5 GQA attention model and the GPU
W4A4 quantization path (compressed-tensors MXFP4/NVFP4 schemes, the MXFP4
method, and ModelOpt NVFP4 / MXFP8 configs) at commit
`72cd5424da80a4a9caa3f42fd65bc0b94e61cbf0`.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | gqa-attention | `vllm/model_executor/models/minimax_m2.py` | `MiniMaxM2Attention` | 137 | 201 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | moe-factory | `vllm/model_executor/models/minimax_m2.py` | `MiniMaxM2MoE` | 70 | 96 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | packed-mapping | `vllm/model_executor/models/minimax_m2.py` | `MiniMaxM2ForCausalLM` | 430 | 437 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | ct-dispatch | `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` | `_get_scheme_from_parts` | 722 | 748 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | w4a4-mxfp4 | `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxfp4.py` | `CompressedTensorsW4A4Mxfp4` | 23 | 101 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | w4a4-nvfp4 | `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` | `CompressedTensorsW4A4Fp4` | 28 | 148 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | mxfp4-moe-only | `vllm/model_executor/layers/quantization/mxfp4.py` | `Mxfp4Config.get_quant_method` | 82 | 108 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | modelopt-w4a4 | `vllm/model_executor/layers/quantization/modelopt.py` | `ModelOptNvFp4Config` | 1002 | 1098 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | minimax-mxfp8 | `vllm/model_executor/layers/quantization/modelopt.py` | `ModelOptMxFp8Config.from_config` | 1719 | 1744 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | kernel-select | `vllm/model_executor/kernels/linear/__init__.py` | `init_mxfp4_linear_kernel` | 840 | 875 |

## Runtime Flow Evidence

1. Checkpoint config discovery — `minimax-mxfp8`, `modelopt-w4a4`.
2. W4A4 scheme selection on GPU — `ct-dispatch`, `w4a4-mxfp4`, `w4a4-nvfp4`,
   `mxfp4-moe-only`.
3. MiniMax M2 GQA layer adaptation — `gqa-attention`, `moe-factory`,
   `packed-mapping`.
4. W4A4 GEMM kernel handoff — `kernel-select`.

## Entry Files

- `vllm/model_executor/models/minimax_m2.py` — MiniMax M2 model: `MiniMaxM2Attention`
  builds a GQA `QKVParallelLinear` `qkv_proj` and `RowParallelLinear` `o_proj`,
  `MiniMaxM2MoE` builds `FusedMoEFactory` experts (`ckpt_names=("w1","w2","w3")`),
  and `MiniMaxM2ForCausalLM` declares `packed_modules_mapping`
  (`qkv_proj -> q/k/v`) used by the compressed-tensors fused weight loader.
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` —
  `_get_scheme_from_parts` dispatches NVFP4 weights to `CompressedTensorsW4A4Fp4`
  (W4A16 when no input activation quant, else true W4A4) and MXFP4 weights to
  `CompressedTensorsW4A4Mxfp4`.
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxfp4.py` —
  W4A4 MXFP4 linear scheme: packed FP4 (E2M1) uint8 weights (2 values/byte),
  per-group E8M0 scales (group 32), dynamic activation quant on SM100+ with
  FlashInfer, Marlin W4A16 fallback elsewhere.
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` —
  W4A4 NVFP4 linear scheme: packed FP4 weights, global weight scale plus
  per-group FP8 E4M3 scales (group 16), and W4A4-mode `alpha` precompute for
  runtime activation quantization on Blackwell.
- `vllm/model_executor/layers/quantization/mxfp4.py` — canonical `Mxfp4Config`:
  linear layers fall back to `UnquantizedLinearMethod` (no GPU MXFP4 linear
  kernel in this revision); routed experts get `Mxfp4MoEMethod`.
- `vllm/model_executor/layers/quantization/modelopt.py` — `ModelOptNvFp4Config`
  maps `quant_algo: NVFP4` to `ModelOptNvFp4LinearMethod` (W4A4 CUTLASS NVFP4
  GEMM) and `W4A16_NVFP4` to the Marlin path; `ModelOptMxFp8Config.from_config`
  normalizes MiniMax-style `quant_method: "mxfp8"` checkpoints to the ModelOpt
  MXFP8 schema.
- `vllm/model_executor/kernels/linear/__init__.py` — `init_mxfp4_linear_kernel`
  selects the platform MXFP4 linear kernel with `--linear-backend` filtering and
  `VLLM_DISABLED_KERNELS` overrides; `init_nvfp4_linear_kernel` selects the NVFP4
  kernel.

## Reproduction Commands

```bash
git -C external-repos/vllm-72cd5424da80 rev-parse HEAD   # 72cd5424da80a4a9caa3f42fd65bc0b94e61cbf0
git -C external-repos/vllm-72cd5424da80 status --porcelain  # clean
rg -n "class MiniMaxM2Attention|QKVParallelLinear|FusedMoEFactory" \
  external-repos/vllm-72cd5424da80/vllm/model_executor/models/minimax_m2.py
rg -n "_get_scheme_from_parts|CompressedTensorsW4A4Mxfp4|CompressedTensorsW4A4Fp4" \
  external-repos/vllm-72cd5424da80/vllm/model_executor/layers/quantization
rg -n "class ModelOptNvFp4Config|class ModelOptMxFp8Config|MiniMax-style" \
  external-repos/vllm-72cd5424da80/vllm/model_executor/layers/quantization/modelopt.py
```

## Limitations

- Static code reading only; runtime behavior was not executed.
- vllm-ascend and vllm revisions are separate main-branch snapshots; they were
  inspected together as the GPU/NPU pair but were not run against each other.
