---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@e3bb5f570f0b7d7fef9df3190a450052bee090cc
commit: e3bb5f570f0b7d7fef9df3190a450052bee090cc
source_record: raw/frameworks/vllm-ascend-codebase--github-e3bb5f570f0b.md
generated: 2026-07-30
---

# vLLM-Ascend e3bb5f570 Important Files

## Reading Scope

Static code reading of the Kimi K3-style routed-MoE serving substrate in
vllm-ascend at commit `e3bb5f570f0b7d7fef9df3190a450052bee090cc`.

## Entry Files

- `vllm_ascend/patch/platform/patch_fused_moe.py` — monkey-patches upstream
  `FusedMoE`, selects `AscendMoERunner` or `AscendMoERunner310`, pushes EPLB
  redundancy into upstream allocation, and forwards `tid2eid`.
- `vllm_ascend/ops/fused_moe/fused_moe.py` — owns `AscendMoERunner`, installs
  Ascend quant methods on `routed_experts`, initializes expert maps and
  `log2phy`, collects EPLB load, and overlaps shared experts.
- `vllm_ascend/ops/fused_moe/experts_selector.py` — gates between fused
  `DeviceOperator.moe_gating_top_k`, hash routing through
  `_C_ascend.moe_gating_top_k_hash`, and native PyTorch fallback.
- `vllm_ascend/ops/fused_moe/moe_runtime_args.py` and
  `vllm_ascend/ops/fused_moe/moe_stage_contracts.py` — define typed payloads
  across prepare, dispatch, MLP, and combine.
- `vllm_ascend/ops/fused_moe/prepare_finalize.py` — pads, slices, gathers, and
  finalizes tensors differently for AllGather, All2All, and MC2.
- `vllm_ascend/ops/fused_moe/token_dispatcher.py` — implements AllGather
  `npu_moe_init_routing`/`npu_moe_token_unpermute`, All2All token exchange, and
  MC2 distributed dispatch/combine.
- `vllm_ascend/ops/fused_moe/moe_mlp.py` — runs unquantized and quantized expert
  MLPs with grouped matmul, SwiGLU/GELU/Step activations, and fused
  GMM+SwiGLU+quant custom ops.
- `vllm_ascend/patch/worker/patch_routed_experts_capture.py` — adapts upstream
  routed-expert capture to Ascend DP, AllGather padding, All2All, MC2, and
  sequence-parallel layouts.
- `vllm_ascend/eplb/adaptor/vllm_adaptor.py` and `vllm_ascend/eplb/core/` —
  collect per-layer loads, compute new placements, update expert maps, and copy
  expert weights for dynamic EPLB.
- `vllm_ascend/xlite/xlite.py` — exposes optional graph runtime adapters for
  Qwen MoE, GLM MoE, and MiniMax M2, but no Kimi K3 adapter.

## Reproduction Commands

```bash
git check-ignore external-repos/vllm-ascend
git -C external-repos/vllm-ascend remote get-url origin
git -C external-repos/vllm-ascend rev-parse HEAD
git -C external-repos/vllm-ascend branch --show-current
git -C external-repos/vllm-ascend status --porcelain
git ls-remote https://github.com/vllm-project/vllm-ascend.git refs/heads/main
git -C external-repos/vllm-ascend diff --name-only 32a59d4e349c12c32cdbc1916436c16e39939afc..origin/main -- vllm_ascend/ops/fused_moe vllm_ascend/patch/platform/patch_fused_moe.py vllm_ascend/patch/worker/patch_routed_experts_capture.py vllm_ascend/eplb vllm_ascend/xlite
rg -n "def _ascend_FusedMoE|class AscendMoERunner|class AscendUnquantizedFusedMoEMethod" external-repos/vllm-ascend/vllm_ascend
rg -n "moe_gating_top_k_hash|moe_gating_top_k|npu_moe_distribute_dispatch|npu_grouped_matmul|dispatch_ffn_combine|mega_moe" external-repos/vllm-ascend/vllm_ascend/ops/fused_moe external-repos/vllm-ascend/csrc/moe external-repos/vllm-ascend/csrc/mc2
rg -n "enable_return_routed_experts|RoutedExpertsCapturer|routed_experts" external-repos/vllm-ascend/vllm_ascend
```

## Limitations

- No runtime traces were captured.
- The direct target worktree is `external-repos/vllm-ascend-e3bb5f570/`,
  detached at the remote `main` commit. The original `external-repos/vllm-ascend/`
  worktree remains on `32a59d4e349c12c32cdbc1916436c16e39939afc`; relevant MoE
  files are unchanged across that range.
- The analysis maps Kimi K3-style routed MoE onto vllm-ascend's generic Ascend
  MoE runner. It does not claim a plugin-local Kimi K3 model class exists.
