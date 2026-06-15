# Wiki Index

## Meta

- [Knowledge Base Introduction](../docs/README.md)

## Benchmarks

- [Benchmarks](../docs/benchmarks/index.md) — Category overview for benchmark pages.
- [τ-bench: Tool-Agent-User Interaction Benchmark](../docs/benchmarks/tau-bench.md) — Original benchmark: two customer-service domains (retail and airline), pass^k metric, empirical results across 12 models, and failure analysis.
- [τ²-Bench: Mechanism and Design](../docs/benchmarks/tau2-bench-mechanism.md) — Dec-POMDP formalism, dual-control domain, task generation, and evaluation methodology.
- [τ-Voice: Full-Duplex Voice Agent Benchmark](../docs/benchmarks/tau-voice.md) — Extends τ²-bench to voice: tick-based orchestrator, controllable voice user simulator, 278 tasks across retail/airline/telecom, empirical results for Google/OpenAI/xAI, acoustic ablations, and error analysis.

## Inference Systems and Frameworks

- [Frameworks](../docs/frameworks/index.md) — Category overview for LLM serving and programming framework pages.
- [SGLang: Structured Language Model Programs](../docs/frameworks/sglang-framework.md) — Framework architecture, Python-embedded programming model, RadixAttention KV cache reuse, compressed FSM decoding, API speculative execution, and performance results.
- [vLLM: PagedAttention Serving Framework](../docs/frameworks/vllm-framework.md) — LLM serving framework design, PagedAttention KV-cache paging, block tables, copy-on-write sharing, scheduling/preemption, distributed execution, and throughput results.

## Attention and Kernel Algorithms

- [Algorithms](../docs/algorithms/index.md) — Category overview for inference algorithm and kernel pages.
- [FlashAttention: IO-Aware Exact Attention](../docs/algorithms/flashattention.md) — Original IO-aware exact attention algorithm: tiling, online softmax, recomputation, IO complexity, block-sparse extension, and training/runtime results.
- [FlashAttention-2: Better Parallelism and Work Partitioning](../docs/algorithms/flashattention-2.md) — Exact attention kernel optimization: reduced non-matmul overhead, sequence-parallel thread blocks, warp-level work partitioning, causal block skipping, and A100/H100 performance results.
- [FlashAttention-3: Hopper Asynchrony and FP8 Attention](../docs/algorithms/flashattention-3.md) — Hopper attention kernel design: warp specialization, TMA/WGMMA asynchrony, GEMM-softmax overlap, FP8 block quantization, incoherent processing, and speed/accuracy results.
- [FlashAttention-4: Blackwell Attention Kernel Co-Design](../docs/algorithms/flashattention-4.md) — Exact attention algorithm and Blackwell kernel design: asynchronous MMA/TMEM pipeline, exponential emulation, conditional softmax rescaling, 2-CTA backward pass, LPT scheduling, and performance results.

## Hardware and Numerics

- [Hardware and Numerics](../docs/hardware/index.md) — Category overview for hardware and numerics pages.
- [NVFP4: Blackwell 4-Bit Floating Point](../docs/hardware/nvfp4.md) — NVIDIA Blackwell NVFP4 format, hierarchical FP8/FP32 scaling, memory benefits, Transformer Engine training recipe, RHT, stochastic rounding, and hardware support.

## Sources
