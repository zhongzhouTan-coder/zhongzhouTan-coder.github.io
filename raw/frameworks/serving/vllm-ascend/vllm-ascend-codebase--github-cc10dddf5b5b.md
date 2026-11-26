---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm-ascend
repository_url: https://github.com/vllm-project/vllm-ascend
local_checkout: external-repos/vllm-ascend-cc10dddf5b5b/
commit: cc10dddf5b5bba8b81a5c80d657ca388c2fd9de2
ref: detached
inspected: 2026-11-25
checkout_state: clean
---

# vLLM Ascend HiSparse Runtime Source Record

## Reading Scope

- HiSparse compatibility and runtime data path for DeepSeek V4.1-Flash on Ascend; includes the October 17 implementation boundary

## Important Entry Files

- `vllm_ascend/attention/sfa_v1.py` — SFA selects HiSparse metadata and implementation when configured
- `vllm_ascend/models/layer/attention/layer.py` — DeepSeek V4.1 DSA selects C4, C128, or SWA backend classes directly
- `vllm_ascend/attention/dsa_v1.py` — DSA implementation selector remains PCP-or-ordinary DSA and has no HiSparse branch
- `vllm_ascend/attention/sfa_hisparse.py` — Ascend SFA-specific HiSparse runtime path
- `vllm_ascend/hisparse/backend.py` — Ascend host mapping, resolver, and DMA backend
- `tests/ut/models/test_deepseek_v41_cache.py` — V4.1 cache test explicitly disables HiSparse

## Limitations

- Static code reading only. No Ascend 950 or A2/A3 runtime was executed.
