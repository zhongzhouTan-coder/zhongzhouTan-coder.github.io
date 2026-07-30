---
kind: repository-source
repository_url: https://github.com/vllm-project/vllm-ascend
local_checkout: external-repos/vllm-ascend-e3bb5f570/
commit: e3bb5f570f0b7d7fef9df3190a450052bee090cc
ref: origin/main
inspected: 2026-07-30
checkout_state: clean
---

# vLLM Ascend Codebase Source Record

## Reading Scope

Static inspection of the vllm-ascend routed-MoE serving path relevant to a
Kimi K3-style large MoE forward pass:

- patched upstream `FusedMoE` construction;
- `AscendMoERunner` initialization and forward implementation;
- top-k expert routing, including hash/sqrtsoftplus paths;
- prepare/finalize, token dispatch/combine, grouped expert MLP, MC2/Fused MC2;
- dynamic EPLB and routed-expert capture support;
- Kimi-adjacent plugin patches and `xlite` MoE adapter coverage.

## Important Entry Files

- `vllm_ascend/patch/platform/patch_fused_moe.py` — replaces upstream
  `FusedMoE` with Ascend's runner before model import.
- `vllm_ascend/ops/fused_moe/fused_moe.py` — defines `AscendMoERunner`,
  `AscendUnquantizedFusedMoEMethod`, shared-expert overlap, EPLB registration,
  and the main routed-MoE forward path.
- `vllm_ascend/ops/fused_moe/experts_selector.py` — selects routed experts with
  Ascend fused top-k ops or native fallback logic.
- `vllm_ascend/ops/fused_moe/moe_runtime_args.py` and
  `vllm_ascend/ops/fused_moe/moe_stage_contracts.py` — typed payload builders
  and dataclass contracts between MoE stages.
- `vllm_ascend/ops/fused_moe/prepare_finalize.py` — communication-specific
  tensor preparation and finalization for AllGather, All2All, and MC2.
- `vllm_ascend/ops/fused_moe/token_dispatcher.py` — token dispatch/combine
  implementations for AllGather, All2All, and MC2.
- `vllm_ascend/ops/fused_moe/moe_mlp.py` — grouped-matmul expert MLP kernels
  and quantized/fused activation paths.
- `vllm_ascend/patch/worker/patch_routed_experts_capture.py` — DP/SP-aware
  capture of top-k routed expert IDs for API return paths.
- `vllm_ascend/eplb/` — dynamic expert placement and weight transfer support.
- `vllm_ascend/xlite/xlite.py` — optional graph runtime adapters for selected
  MoE architectures.

## Limitations

- Static code reading only; no Ascend NPU execution, Kimi K3 checkpoint load, or
  performance validation was run.
- Remote `main` was checked with `git ls-remote`, fetched, and inspected at
  `e3bb5f570f0b7d7fef9df3190a450052bee090cc`. The three commits after
  `32a59d4e349c12c32cdbc1916436c16e39939afc` did not modify the inspected MoE,
  EPLB, routed-expert capture, patch-fused-MoE, or `xlite` files.
- This vllm-ascend revision does not define a literal `kimi_k3.py` plugin model
  file. Kimi K3 inference is understood through the upstream-compatible
  `FusedMoE` substrate plus Kimi-adjacent patches and supported MoE runtime
  mechanisms.
