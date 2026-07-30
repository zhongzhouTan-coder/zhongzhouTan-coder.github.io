---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@e3bb5f570f0b7d7fef9df3190a450052bee090cc
commit: e3bb5f570f0b7d7fef9df3190a450052bee090cc
source_record: raw/frameworks/vllm-ascend-codebase--github-e3bb5f570f0b.md
generated: 2026-07-30
---

# vLLM-Ascend Kimi K3-Style MoE Forward Notes

## Scope

The inspected revision is a clean local checkout of
`vllm-project/vllm-ascend` on remote `main` at
`e3bb5f570f0b7d7fef9df3190a450052bee090cc`. The repository contains Kimi
K2.5 compatibility patches and KDA kernels, but no plugin-local `kimi_k3.py`.
The relevant Kimi K3 insight is therefore the Ascend routed-MoE forward
substrate that an upstream vLLM Kimi K3 model would use after the plugin
patches upstream `FusedMoE`.

Remote verification found that `main` advanced from the original local checkout
`32a59d4e349c12c32cdbc1916436c16e39939afc` to
`e3bb5f570f0b7d7fef9df3190a450052bee090cc`. The intervening commits do not
touch the inspected fused-MoE, EPLB, routed-expert capture, patch-fused-MoE, or
`xlite` paths, so the detailed MoE reading is unchanged while the source pin is
now the true latest commit.

## Evidence Map

- `patch/platform/patch_fused_moe.py` patches both upstream FusedMoE bindings
  before model import. `_ascend_FusedMoE()` chooses `AscendMoERunner`, carries
  EPLB redundant expert allocation, and forwards `tid2eid`.
- `AscendMoERunner.__init__()` copies upstream routing params from
  `routed_experts`, installs Ascend quant methods on that child module,
  initializes TP/DP/EP/MC2 groups, logical and physical expert maps, `log2phy`,
  dynamic EPLB load buffers, and `VllmEplbAdaptor` registration.
- `no_shared_forward_impl()` calls the active communication method's
  `prepare()`, calls the quant method's `apply()` with `self.routed_experts` as
  the weight owner, collects optional EPLB load, and finalizes routed output.
- `AscendUnquantizedFusedMoEMethod.apply()` calls `select_experts()`, captures
  top-k IDs when `enable_return_routed_experts` is enabled, builds typed
  `MoEFusedExpertsInput`, and delegates dispatch/MLP/combine to `MoECommMethod`.
- `experts_selector.py` supports fused softmax/sigmoid routing through
  `DeviceOperator.moe_gating_top_k()` and the `sqrtsoftplus` hash branch
  through `_C_ascend.moe_gating_top_k_hash()`. Unsupported cases fall back to
  native PyTorch scoring and top-k.
- `moe_runtime_args.py` and `moe_stage_contracts.py` define typed payloads for
  prepare output, fused experts input, dispatch input, dispatch output, MLP
  input, and communication-specific combine metadata.
- `moe_comm_method.py` implements map-logical-to-physical, dispatch, build MLP
  input, `unified_apply_mlp()`, combine, and `FusedExpertsResult` return.
- `token_dispatcher.py` dispatches AllGather with
  `DeviceOperator.npu_moe_init_routing()`, All2All with token permute plus EP
  all-to-all, and MC2 with `torch_npu.npu_moe_distribute_dispatch(_v2)` and
  `torch_npu.npu_moe_distribute_combine(_v2)`.
- `FusedMC2CommImpl` can use CANN MegaMoe when supported, allocating its
  symmetric buffer from rank-invariant capacity rather than current per-rank
  token count. Otherwise, `enable_fused_mc2 == 1` calls
  `_C_ascend.dispatch_ffn_combine()`.
- `moe_mlp.py` runs unquantized grouped-matmul gate/up, activation, and
  grouped-matmul down. Quantized branches support W8A8/W4A8/MXFP variants,
  fused GMM+SwiGLU+quant custom ops, `swiglustep`, GELU, clipped
  `swigluoai_uninterleave`, and MoE LoRA deltas.
- `patch_routed_experts_capture.py` repairs top-k capture under DP, padded
  AllGather, All2All, MC2, and sequence-parallel layouts.
- `eplb/` collects `moe_load`, computes new placements, validates expert
  placement, updates expert maps and `log2phy`, and moves expert weights.
- `patch/worker/patch_kimi_k25.py` only patches Kimi K2.5 vision/quantization
  behavior; `xlite/xlite.py` has graph adapters for Qwen MoE, GLM MoE, and
  MiniMax M2, but not Kimi K3.

## Interpretation

For a Kimi K3-style large routed-MoE forward pass, the current vllm-ascend
insight is not "find the Kimi model file." It is "understand the patched
FusedMoE substrate": upstream model code constructs `FusedMoE`, vllm-ascend
replaces the runner, and all high-cardinality expert routing, communication,
grouped MLP compute, capture, and EPLB mechanics happen in the plugin's fused
MoE package.
