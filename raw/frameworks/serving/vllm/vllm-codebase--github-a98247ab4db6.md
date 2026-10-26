---
kind: repository-source
provider: github
clone_url: https://github.com/vllm-project/vllm.git
repository_url: https://github.com/vllm-project/vllm
local_checkout: external-repos/vllm-qwen38-upstream/
commit: a98247ab4db686ee03c66d5feb3c761e52a2f8ab
ref: detached
inspected: 2026-10-26
checkout_state: clean
---

# vllm Codebase Source Record

## Reading Scope

- Qwen4Exp upstream hybrid decoder, QSA cache specs, MTP, and V1 scheduler output

## Important Entry Files

- `vllm/models/qwen4_exp/amd/model.py` — Target decoder and output
- `vllm/models/qwen4_exp/common/qsa_cache.py` — QSA cache specifications

## Limitations

- Static code reading only; runtime behavior was not executed.
