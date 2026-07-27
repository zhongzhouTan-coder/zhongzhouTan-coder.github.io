# NVFP4 Reference Notes

Source URLs consulted on 2026-06-11:

- NVIDIA Technical Blog, "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference", published 2025-06-24: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- NVIDIA Transformer Engine documentation, "NVFP4", Transformer Engine 2.17.0.dev0 documentation: https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html#data-format

## Key Facts

- NVFP4 is a 4-bit floating-point format introduced with NVIDIA Blackwell Tensor Cores.
- The stored element format is E2M1: 1 sign bit, 2 exponent bits, and 1 mantissa bit.
- The raw E2M1 value range is approximately -6 to +6.
- NVFP4 reconstructs values with hierarchical scaling: a local FP8 E4M3 block scale and a global FP32 tensor scale.
- For 1D NVFP4 scaling, each local block scale is shared by 16 consecutive elements.
- Transformer Engine expresses an NVFP4 value as `x = x_e2m1 * s_block * s_global`.
- Transformer Engine computes `s_global = global_amax / (fp8_max * fp4_max)`, where `fp8_max` is 448.0 for FP8 E4M3 and `fp4_max` is 6.0 for NVFP4 E2M1.
- Transformer Engine computes `s_block = (block_amax / fp4_max) / s_global`, then stores `s_block` in FP8 E4M3.
- NVIDIA contrasts NVFP4 with MXFP4: MXFP4 uses a 32-value block with a power-of-two E8M0 scale, while NVFP4 uses a 16-value block with a fractional FP8 E4M3 scale.
- NVIDIA reports NVFP4 storage as one 4-bit value plus one FP8 scale per 16 values, or about 4.5 bits per value, plus one FP32 second-level scale per tensor.
- NVIDIA reports approximate model memory footprint reductions of 3.5x versus FP16 and 1.8x versus FP8.
- NVIDIA reports that DeepSeek-R1-0528 quantized from FP8 to NVFP4 with PTQ showed 1 percentage point or less accuracy degradation on several cited evaluations, with AIME 2024 improving by 2 percentage points.
- Transformer Engine treats NVFP4 as its first 4-bit recipe and adds training-stability features beyond the data format.
- Transformer Engine uses 2D scaling for weights by default and 1D scaling for activations and gradients. The recipe option `disable_2d_quantization=True` forces 1D scaling for weights.
- Transformer Engine applies stochastic rounding when casting scaled values to NVFP4. Stochastic rounding is enabled only for gradients and can be disabled with `disable_stochastic_rounding=True`.
- Transformer Engine says stochastic rounding is hardware-accelerated with native GPU instructions introduced in Blackwell.
- Transformer Engine's Random Hadamard Transform (RHT) smooths outliers before quantization for columnwise quantization of inputs and gradients used by the wgrad GEMM. RHT supports BF16 inputs and gradients only.
- Transformer Engine performs the RHT in tiles of size `d = 16`.
- Transformer Engine requires both rowwise and columnwise quantized tensors for GEMM operands. NVFP4 GEMM supports only the TN layout.
- Transformer Engine stores columnwise data and scaling factors in transposed layout.
- In distributed training, local block scales do not require synchronization, but the global scale requires a synchronized global amax for gathered tensors.
- Transformer Engine supports NVFP4 all-gather.
- Transformer Engine's NVFP4 PyTorch and JAX examples use the `NVFP4BlockScaling` recipe. RHT and 2D weight quantization are enabled by default.
- Transformer Engine lists NVFP4 training support for SM 10.0 and SM 10.3, and inference support for SM 10.0+.
