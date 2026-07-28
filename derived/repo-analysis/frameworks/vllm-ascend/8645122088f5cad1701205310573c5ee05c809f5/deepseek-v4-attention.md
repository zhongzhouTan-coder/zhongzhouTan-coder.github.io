---
kind: repository-analysis
repository_id: github:vllm-project/vllm-ascend@8645122088f5cad1701205310573c5ee05c809f5
commit: 8645122088f5cad1701205310573c5ee05c809f5
source_record: raw/frameworks/vllm-ascend-codebase--github-8645122088f5.md
generated: 2026-07-28
---

# vLLM-Ascend DeepSeek V4 Attention Implementation Notes

## Evidence Map

### Core Model (`vllm_ascend/models/deepseek_v4.py`)

- `AscendCompressorStateCache`, `AscendDeepseekV4IndexerCache`, `AscendDeepseekV4SWACache`: Ascend-specific KV cache subclasses that override `get_kv_cache_spec()` to use `AscendSlidingWindowMLASpec` and `AscendMLAAttentionSpec` (ascend-aware block sizes and NPU-compatible dtypes).
- `AscendDeepseekV4ForCausalLM`: The full model class, importing attention infra from upstream vLLM's `DeepseekV4Attention`, `DeepseekV4IndexerCache`, `CompressorStateCache`, and `DeepseekV4SWACache`.
- Block size dispatch: `_dsv4_block_sizes()` returns per-device-type (A5 vs non-A5) block size mappings for `{128, 64, 32}` cache block sizes → `[mla, swa, c4_state, c128_state]` and `[page_size_padded_t1, page_size_padded_t2]`.
- Uses `AscendDeepseekSparseAttention` from `vllm_ascend.ops.dsa` for the actual attention compute.
- HF config: `DeepseekV4Config` from `vllm.transformers_utils.configs.deepseek_v4`.

### Attention Layer (`vllm_ascend/models/layer/attention/layer.py`)

- `DSAAttention`: Ascend-specific MLA attention layer. Takes q, kv_c_normed, k_pe as inputs. Uses `AscendDSABackend` as the attention backend. Sets up `AscendMLAAttentionSpec` for KV cache with `model_version="deepseek_v4"` and `compress_ratio`.
- `get_dsv4_block_sizes()`: Returns per-device-type block size configurations for A5 (Ascend 910B5) vs non-A5 devices. Different page_size_padded values due to different HBM alignment requirements.

### DSA Attention (`vllm_ascend/attention/dsa_v1.py`)

- `AscendDSABackend`: The core DSA (Deepseek Sparse Attention) backend registered as `"ASCEND_DSA"` / `"FLASH_ATTN"` (depending on model runner version). Subclasses `AttentionBackend`.
- `AscendDSAAttentionImpl`: The actual attention implementation class, handling NPU-specific computation.
- Includes Hadamard transform utilities (`rotate_activation`, `hadamard_transform_ref`) for activation rotation.
- `pad_to_blocks()`: Pads ragged/packed tensors into fixed-size blocks for NPU kernel consumption.
- Multi-stream overlap: Uses `dsv4_dsa_overlap_stream()` for NPU stream-based parallelism between compute and KV scatter operations.

### DSA Context Parallel (`vllm_ascend/attention/context_parallel/dsa_cp.py`)

- Context-parallel DSA implementation for distributed inference.

### Ops (`vllm_ascend/ops/`)

- `dsa.py`: `AscendDeepseekSparseAttention` and `DSAModules` — the NPU-optimized DSA kernel implementation.
- `rope_dsv4.py`: `ComplexExpRotaryEmbedding` — Ascend-specific RoPE implementation with `get_cos_and_sin_dsa`.
- `triton/mul_add.py`: Triton-based multiply-add kernel.

### MTP and DSpark (`vllm_ascend/models/`)

- `deepseek_v4_mtp.py`: Multi-Token Prediction variant for Ascend.
- `deepseek_v4_dspark.py`: DSpark speculative decoding variant for Ascend.

## Key Differences from NVIDIA vLLM

| Aspect | NVIDIA (vLLM) | Ascend (vllm-ascend) |
|---|---|---|
| Attention backend | FlashMLA / FlashInfer / Triton | AscendDSABackend (NPU DSA op) |
| KV cache dtype | fp8_ds_mla (UE8M0), bf16, fp8_e4m3fn | fp8_e4m3fn (A5), fp16 (non-A5) |
| Block sizes | 256 (MLA), 4/8 (compressor states) | 128/64/32 (MLA+SLA), 4-32/8-32 (compressor states) |
| Attention compute | flashinfer_trtllm_batch_decode_sparse_mla_dsv4 | AscendDeepseekSparseAttention (custom NPU op) |
| Stream overlap | CUDA streams (up to 4) | NPU streams (dsv4_dsa_overlap_stream) |
| RoPE | Triton register-based | ComplexExpRotaryEmbedding (Ascend native) |
| Hadamard rotation | Python reference | Python reference |

## Reproduction Commands

```bash
# Find all DeepSeek V4 files in vllm-ascend
find external-repos/vllm-ascend -type f -name '*.py' | xargs grep -l 'deepseek_v4\|DeepseekV4' 2>/dev/null | sort

# Count DSA attention lines
wc -l external-repos/vllm-ascend/vllm_ascend/attention/dsa_v1.py

# Find Ascend attention layer implementation
wc -l external-repos/vllm-ascend/vllm_ascend/models/layer/attention/layer.py
```
