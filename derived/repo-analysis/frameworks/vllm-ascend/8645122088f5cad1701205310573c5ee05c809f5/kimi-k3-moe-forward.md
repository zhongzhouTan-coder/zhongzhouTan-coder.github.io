---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@8645122088f5cad1701205310573c5ee05c809f5
commit: 8645122088f5cad1701205310573c5ee05c809f5
source_record: raw/frameworks/vllm-ascend-codebase--github-8645122088f5.md
generated: 2026-07-29
---

# vLLM-Ascend Kimi K3-Style MoE Forward Notes

## Scope

Static reading of the Ascend large-MoE forward path that would run a Kimi
K3-style routed-MoE layer. The pinned revision does not contain a literal
`kimi_k3.py` model file; the implemented path is the upstream-compatible
`FusedMoE` factory plus the vllm-ascend `AscendMoERunner`, with concrete
DeepSeek V4 model wiring in `vllm_ascend/models/deepseek_v4.py`.

## Evidence Map

- `vllm_ascend/patch/platform/patch_fused_moe.py` monkey-patches upstream
  `vllm.model_executor.layers.fused_moe.FusedMoE` before model import. It
  makes `AscendMoERunner` the default runner, propagates Ascend EPLB redundant
  expert capacity into the upstream factory, and passes Ascend-specific
  `tid2eid` router metadata through `runner_args`.
- `vllm_ascend/models/deepseek_v4.py` defines `DeepseekV4MoE`. Its
  constructor builds a gate, optional shared experts, and `FusedMoE(...)` with
  routed expert count, top-k, grouped top-k settings, correction bias, shared
  expert fusion flags, hash routing flags, and EPLB metadata.
- `DeepseekV4MoE.forward()` reshapes token hidden states, optionally sequence
  parallel chunks them, computes router logits with `F.linear()` unless the
  runner owns an internal router, calls `self.experts(...)`, then merges shared
  and routed outputs and restores sequence-parallel/tensor-parallel layout.
- `vllm_ascend/ops/fused_moe/fused_moe.py` defines
  `AscendUnquantizedFusedMoEMethod.apply()`. It calls `select_experts()` to
  produce `topk_weights` and `topk_ids`, builds `MoEFusedExpertsInput`, then
  delegates routed compute to `_EXTRA_CTX.moe_comm_method.fused_experts(...)`.
- `vllm_ascend/ops/fused_moe/experts_selector.py` selects NPU fused routing
  when constraints allow it. The fused `sqrtsoftplus` path calls
  `torch.ops._C_ascend.moe_gating_top_k_hash(...)`; unsupported cases fall back
  to PyTorch top-k/grouped-top-k logic.
- `vllm_ascend/ops/fused_moe/moe_comm_method.py` implements the generic routed
  stage as dispatch, grouped MLP, and combine. Communication methods include
  `AllGatherCommImpl`, `AlltoAllCommImpl`, `MC2CommImpl`, and
  `FusedMC2CommImpl`.
- `vllm_ascend/ops/fused_moe/token_dispatcher.py` implements the dispatch and
  combine details. AllGather uses `DeviceOperator.npu_moe_init_routing()` and
  `DeviceOperator.npu_moe_token_unpermute()`. MC2 uses
  `torch_npu.npu_moe_distribute_dispatch(_v2)` and
  `torch_npu.npu_moe_distribute_combine(_v2)`. All-to-all uses
  `torch_npu.npu_moe_token_permute()`, asynchronous all-to-all, and
  `torch_npu.npu_moe_token_unpermute()`.
- `vllm_ascend/ops/fused_moe/moe_mlp.py` runs the expert MLP with grouped
  matmul kernels. The normal structure is GMM1 for gate/up projection,
  SwiGLU or related activation, then GMM2 for down projection. Quantized paths
  use dynamic quantization and fused grouped-matmul/SwiGLU/quant custom ops
  where available.
- `AscendMoERunner.no_shared_forward_impl()` prepares tensors, calls the
  quant method, optionally records per-expert token load for dynamic EPLB, and
  finalizes the routed output. `shared_forward_impl()` overlaps shared-expert
  computation with routed dispatch/GMM/combine using NPU events and an optional
  shared-expert stream.
- EPLB support is initialized in `AscendMoERunner.__init__()` and connected to
  `VllmEplbAdaptor.register_layer(self)`. Dynamic load collection uses
  `fused_experts_results.expert_tokens` to update `self.moe_load`.

## Reproduction Commands

```bash
rg -n "def _ascend_FusedMoE|class AscendMoERunner|class AscendUnquantizedFusedMoEMethod" external-repos/vllm-ascend/vllm_ascend
rg -n "class DeepseekV4MoE|self.experts = FusedMoE|def forward" external-repos/vllm-ascend/vllm_ascend/models/deepseek_v4.py
rg -n "moe_gating_top_k_hash|npu_moe_init_routing|npu_moe_distribute_dispatch|npu_grouped_matmul|dispatch_ffn_combine" external-repos/vllm-ascend/vllm_ascend/ops/fused_moe external-repos/vllm-ascend/csrc/moe external-repos/vllm-ascend/csrc/mc2
```

## Limitations

- Static code reading only; no Kimi K3 checkpoint was loaded and no Ascend NPU
  run was executed.
- The page maps Kimi K3's routed-MoE serving concept to the available
  vllm-ascend large-MoE implementation. Exact Kimi K3 model-class differences
  would require a revision containing a Kimi K3-specific vLLM model definition
  or an explicit upstream model mapping.
