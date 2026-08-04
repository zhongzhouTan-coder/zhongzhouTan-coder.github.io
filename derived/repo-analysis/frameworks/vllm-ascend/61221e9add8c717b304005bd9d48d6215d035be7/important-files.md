---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@61221e9add8c717b304005bd9d48d6215d035be7
commit: 61221e9add8c717b304005bd9d48d6215d035be7
source_record: raw/frameworks/vllm-ascend-codebase--github-61221e9add8c.md
generated: 2026-08-04
---

# vllm-ascend Codebase Important Files

## Reading Scope

Static code reading of the MiniMax M2/M2.5 W4A4 quantization path on the
Ascend NPU: ModelSlim config parsing (`quant_model_description.json`),
scheme registration, W4A4 linear/MoE methods (MXFP4, FLATQUANT, LAOS), and the
MiniMax M2 fp8-disable patch at commit
`61221e9add8c717b304005bd9d48d6215d035be7`.

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | modelslim-config | `vllm_ascend/quantization/modelslim_config.py` | `AscendModelSlimConfig` | 508 | 560 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | packed-minimax | `vllm_ascend/quantization/modelslim_config.py` | `packed_modules_model_mapping` | 246 | 254 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | quant-method-dispatch | `vllm_ascend/quantization/modelslim_config.py` | `AscendModelSlimConfig.get_quant_method` | 665 | 759 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | config-load | `vllm_ascend/quantization/modelslim_config.py` | `AscendModelSlimConfig.maybe_update_config` | 820 | 900 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | scheme-factory | `vllm_ascend/quantization/modelslim_config.py` | `create_scheme_for_layer` | 469 | 505 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | scheme-registry | `vllm_ascend/quantization/methods/registry.py` | `get_scheme_class` | 52 | 60 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | npu-w4a4-mxfp4 | `vllm_ascend/quantization/methods/w4a4_mxfp4.py` | `AscendW4A4MXFP4DynamicLinearMethod` | 42 | 123 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | npu-w4a4-flatquant | `vllm_ascend/quantization/methods/w4a4_flatquant.py` | `AscendW4A4FlatQuantDynamicLinearMethod` | 78 | 160 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | npu-w4a4-laos | `vllm_ascend/quantization/methods/w4a4_laos_dynamic.py` | `AscendW4A4LaosDynamicLinearMethod` | 28 | 76 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | adapter-linear | `vllm_ascend/quantization/method_adapters.py` | `AscendLinearMethod` | 36 | 190 |
| `docs/frameworks/vllm/minimax-gqa-w4a4-quantization-path.md` | minimax-fp8-disable | `vllm_ascend/patch/platform/patch_minimax_m2_config.py` | `_should_disable_fp8` | 61 | 90 |

## Runtime Flow Evidence

1. ModelSlim config load — `config-load`, `modelslim-config`.
2. MiniMax M2 layer-to-scheme dispatch — `packed-minimax`, `quant-method-dispatch`,
   `scheme-factory`, `scheme-registry`.
3. W4A4 weight layout creation and packing — `adapter-linear`, `npu-w4a4-mxfp4`,
   `npu-w4a4-flatquant`, `npu-w4a4-laos`.
4. W4A4 forward matmul on NPU — `npu-w4a4-mxfp4`, `npu-w4a4-flatquant`,
   `npu-w4a4-laos`.
5. MiniMax M2 fp8 checkpoint handling — `minimax-fp8-disable`.

## Entry Files

- `vllm_ascend/quantization/modelslim_config.py` — registers
  `AscendModelSlimConfig` as the `"ascend"` quant method, loads
  `quant_model_description.json` in `maybe_update_config`, maps `minimax_m2`
  fused modules (`qkv_proj` q/k/v, `experts` w1/w2/w3), rewrites MiniMax
  prefixes (`mlp` -> `block_sparse_moe`, expert index normalization) in
  `get_quant_method`, and instantiates per-layer schemes via
  `create_scheme_for_layer`.
- `vllm_ascend/quantization/methods/registry.py` — `_SCHEME_REGISTRY` keyed by
  `(quant_type, layer_type)`; `register_scheme` decorator and `get_scheme_class`.
- `vllm_ascend/quantization/methods/w4a4_mxfp4.py` — `W4A4_MXFP4` linear and MoE
  methods: FP4 (E2M1) weights packed into uint8, per-group E8M0 scales (group
  32), dynamic FP4 activation quant, `npu_quant_matmul` forward.
- `vllm_ascend/quantization/methods/w4a4_flatquant.py` — `W4A4_FLATQUANT_DYNAMIC`
  linear method: per-channel INT4 weights packed to int32, FlatQuant Kronecker
  left/right transforms + clip ratio for activation smoothing, then
  `npu_quant_matmul`.
- `vllm_ascend/quantization/methods/w4a4_laos_dynamic.py` — `W4A4_DYNAMIC` (LAOS)
  linear method: per-channel INT4 weights with scale/offset, `quint4x2` dynamic
  activation quant, `npu_quant_matmul` with per-token scale.
- `vllm_ascend/quantization/method_adapters.py` — `AscendLinearMethod` wraps a
  scheme: `create_weights` calls the scheme's `get_weight`/`get_pertensor_param`/
  `get_perchannel_param`/`get_pergroup_param` and registers packing attrs;
  `apply` delegates to the scheme with TP-rank handling for `o_proj`/`down_proj`.
- `vllm_ascend/quantization/methods/base.py` — `AscendLinearScheme` /
  `AscendMoEScheme` / `AscendAttentionScheme` abstract contracts.
- `vllm_ascend/patch/platform/patch_minimax_m2_config.py` — disables fp8
  quantization for `minimax_m2` checkpoints on NPU (loads bf16 dequantized
  weights) and sets `HCCL_OP_EXPANSION_MODE=AIV` for ACL graph capture.

## Reproduction Commands

```bash
git -C external-repos/vllm-ascend-61221e9add8c rev-parse HEAD   # 61221e9add8c717b304005bd9d48d6215d035be7
git -C external-repos/vllm-ascend-61221e9add8c status --porcelain  # clean
rg -n "minimax_m2|class AscendModelSlimConfig|def get_quant_method|def maybe_update_config" \
  external-repos/vllm-ascend-61221e9add8c/vllm_ascend/quantization/modelslim_config.py
rg -n "@register_scheme|class AscendW4A4" \
  external-repos/vllm-ascend-61221e9add8c/vllm_ascend/quantization/methods/w4a4_*.py
rg -n "_should_disable_fp8|minimax_m2" \
  external-repos/vllm-ascend-61221e9add8c/vllm_ascend/patch/platform/patch_minimax_m2_config.py
```

## Limitations

- Static code reading only; runtime behavior was not executed.
- The `W4A4` schemes target A2/A3-class Ascend NPUs (tests live under
  `tests/ut/quantization/methods/a2/`); A5/MXFP4 hardware paths differ in kernel
  selection details not covered here.
- vllm-ascend and vllm revisions are separate main-branch snapshots; they were
  inspected together as the GPU/NPU pair but were not run against each other.
