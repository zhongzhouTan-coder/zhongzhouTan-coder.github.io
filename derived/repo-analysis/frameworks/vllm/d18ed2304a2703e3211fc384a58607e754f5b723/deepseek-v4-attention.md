---
kind: repository-analysis
repository_id: github:vllm-project/vllm@d18ed2304a2703e3211fc384a58607e754f5b723
commit: d18ed2304a2703e3211fc384a58607e754f5b723
source_record: raw/frameworks/vllm-codebase--github-d18ed2304a27.md
generated: 2026-07-28
---

# vLLM DeepSeek V4 Attention Implementation Notes

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/deepseek/v4-attention-code-reading.md` | attention-core | `vllm/models/deepseek_v4/attention.py` | `DeepseekV4Attention` | 71 | — |
| `docs/frameworks/deepseek/v4-attention-code-reading.md` | nvidia-model | `vllm/models/deepseek_v4/nvidia/model.py` | `DeepseekV4ForCausalLM` | 82 | — |
| `docs/frameworks/deepseek/v4-attention-code-reading.md` | compressor | `vllm/models/deepseek_v4/compressor.py` | `DeepseekCompressor` | 39 | — |
| `docs/frameworks/deepseek/v4-attention-code-reading.md` | fused-compress-insert | `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` | `compress_norm_rope_store_triton` | 32 | — |
| `docs/frameworks/deepseek/v4-attention-code-reading.md` | fused-indexer-q | `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` | `_fused_indexer_q_rope_quant_kernel` | 15 | — |
| `docs/frameworks/deepseek/v4-attention-code-reading.md` | flashmla-backend | `vllm/models/deepseek_v4/sparse_mla.py` | `DeepseekV4FlashMLABackend` | 35 | — |
| `docs/frameworks/deepseek/v4-attention-code-reading.md` | flashinfer-dispatch | `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` | `DeepseekV4FlashInferMLASparseBackend` | 36 | — |
| `docs/frameworks/deepseek/v4-attention-code-reading.md` | indexer-backend | `vllm/v1/attention/backends/mla/indexer.py` | `DeepseekV4IndexerBackend` | 46 | — |
| `docs/frameworks/deepseek/v4-attention-code-reading.md` | kv-cache-spec | `vllm/v1/kv_cache_interface.py` | `MLAAttentionSpec` | 33 | — |
| `docs/frameworks/deepseek/v4-attention-code-reading.md` | sparse-swa | `vllm/v1/attention/backends/mla/sparse_swa.py` | `DeepseekV4SWACache` | 43 | — |
| `docs/frameworks/deepseek/v4-attention-code-reading.md` | slot-mapping | `vllm/v1/attention/backends/mla/compressor_utils.py` | `get_compressed_slot_mapping` | 9 | — |

## Runtime Flow Evidence

1. Core attention pipeline — `attention-core`, `nvidia-model`.
2. KV compression and insertion — `compressor`, `fused-compress-insert`, `fused-indexer-q`.
3. Sparse MLA backends — `flashmla-backend`, `flashinfer-dispatch`, `indexer-backend`.
4. Heterogeneous KV cache shapes — `kv-cache-spec`, `sparse-swa`, `slot-mapping`.

## Evidence Map

### Core Attention (`vllm/models/deepseek_v4/`)

- `attention.py` (~700+ lines): `DeepseekV4Attention` base class with platform-abstract methods (`forward_mqa`, `_o_proj`, `get_padded_num_q_heads`). Contains fused `fused_wqa_wkv` GEMM, multi-stream overlap execution of qr/kv, indexer, and compressor in `attn_gemm_parallel_execute()`. The `forward()` method orchestrates: input GEMMs → RMSNorm → `attention_impl()` (eager-break for CUDAGraph) → `_o_proj` (inverse-RoPE + grouped output projection). CSA layers (compress_ratio=4) instantiate both `DeepseekV4Indexer` and `DeepseekCompressor`; HCA layers (compress_ratio=128) instantiate only `DeepseekCompressor`; SWA-only layers (compress_ratio=1) have neither.
- `compressor.py` (~350 lines): `DeepseekCompressor` owns `fused_wkv_wgate` linear, `RMSNorm`, `CompressorStateCache`, and APE (Absolute Positional Encoding) state. `CompressorBackend` + `CompressorMetadataBuilder` handle KV cache insertion. `CompressorStateCache` stores float32 partial states with per-layer sliding_window = coff * compress_ratio. Dispatches to triton `compress_norm_rope_store_triton` (or two-stage variant for ROCm).
- `sparse_mla.py` (~150 lines): `DeepseekV4FlashMLABackend` — the FlashMLA sparse-MLA backend for SM9x/SM10x GPUs. Declares KV cache shape: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale) when using `fp8_ds_mla` layout, or `head_dim` otherwise. Supports sink attention.
- `common/ops/fused_compress_quant_cache.py` (~400 lines): Three Triton kernels for fused compressor + norm + RoPE + quant + KV cache insert: (1) `_fused_kv_compress_norm_rope_insert_sparse_attn` for head=512 (nope=448 FP8 + rope=64 bf16), (2) `_fused_kv_compress_norm_rope_insert_indexer_attn` for head=128 all-FP8, (3) `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn` for head=128 MXFP4 (block=32).
- `common/ops/fused_indexer_q.py` (~200 lines): `_fused_indexer_q_rope_quant_kernel` — fused indexer Q RoPE + FP8/MXFP4 quantize. MXFP4 uses `_fp32x2_to_fp4x2` inline PTX.
- `common/ops/fused_qk_rmsnorm.py`: Fused Q and K RMSNorm kernel.
- `common/ops/cache_utils.py`: Cache layout utilities.

### Platform-specific Attention (`vllm/models/deepseek_v4/nvidia/`)

- `flashinfer_sparse.py` (~200 lines): `DeepseekV4FlashInferMLASparseBackend` extends FlashMLABackend. Uses FlashInfer's `flashinfer_trtllm_batch_decode_sparse_mla_dsv4` for sparse MLA decode on SM10x/SM12x. SM10x requires bf16 or plain FP8 KV; SM12x uses fp8_ds_mla layout. `DeepseekV4FlashInferMLAAttention` and `DeepseekV4FlashInferSM120Attention` are the attention subclasses.
- `flashmla.py`: `DeepseekV4FlashMLAAttention` — FlashMLA-based attention subclass.
- `ops/o_proj.py`: `compute_fp8_einsum_recipe` + `deep_gemm_fp8_o_proj` for the grouped output projection (inverse-RoPE → wo_a → wo_b).
- `model.py` (~600+ lines): `DeepseekV4ForCausalLM` wires layers with mHC (manifold-constrained hyper-connections) via `mhc_pre_tilelang`, `mhc_fused_post_pre_tilelang`, `mhc_post_tilelang`. `DeepseekV4MegaMoE` uses MXFP4 Megablocks. `SiluAndMulWithClamp` for SwiGLU clamping.

### Platform-specific (`amd/`, `xpu/`)

- `amd/model.py`, `amd/rocm.py`: ROCm platform attention using `DeepseekV4ROCMAiterMLAAttention`.
- `xpu/model.py`, `xpu/xpu_sparse.py`: XPU (Intel GPU) platform attention.

### V1 Attention Infrastructure (`vllm/v1/attention/`)

- `backends/mla/indexer.py` (~500 lines): `DeepseekV4IndexerBackend` — the sparse indexer backend managing indexer KV cache, top-k selection, and slot mapping for compressed indexer keys. Uses `prepare_uniform_decode_kernel` for decoding.
- `backends/mla/sparse_swa.py`: `DeepseekV4SWACache` — sliding window attention cache layer.
- `backends/mla/compressor_utils.py`: `get_compressed_slot_mapping` for mapping from expanded to compressed slot indices.

### KV Cache Interface (`vllm/v1/kv_cache_interface.py`)

- `MLAAttentionSpec`, `SlidingWindowMLASpec` dataclasses define heterogeneous KV cache shapes.
- DeepSeek V4 uses three cache types: SWA cache (sliding window), compressor state cache (partial compression states), and main MLA cache (compressed entries).

## Architecture Pattern: Multi-Stream Overlap

The attention forward pass uses up to 4 CUDA streams:

- **Default stream**: `fused_wqa_wkv` (heaviest GEMM) → `wq_b` → `_fused_qnorm_rope_kv_insert`
- **Aux stream 0**: `compressor_kv_score` GEMM → compressor triton kernel
- **Aux stream 1**: `indexer_weights_proj` GEMM
- **Aux stream 2**: `indexer_compressor_kv_score` GEMM → indexer forward (Q RoPE + quantize + top-k)

On CSA layers (compress_ratio=4), this achieves 3-way overlap. On HCA layers (compress_ratio=128), it's a simpler 2-way overlap (no indexer). ROCm falls back to sequential execution.

## Reproduction Commands

```bash
# Find all DeepSeek V4 model files
find external-repos/vllm/vllm/models/deepseek_v4 -type f -name '*.py' | sort

# Count lines in attention module
wc -l external-repos/vllm/vllm/models/deepseek_v4/attention.py

# Find Triton kernels used by V4 attention
rg '@triton\.jit' external-repos/vllm/vllm/models/deepseek_v4 -g '*.py' --no-heading
```
