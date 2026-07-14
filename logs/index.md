# Wiki Index

## Meta

- [Knowledge Base Introduction](../docs/README.md)

## Benchmarks

- [Benchmarks](../docs/benchmarks/index.md) — Category overview for benchmark pages.
- [Pier: Coding-Agent Evaluation Harness](../docs/benchmarks/pier/index.md) — Harbor-compatible coding-agent evaluation harness focused on installed agents in sandboxed tasks, stricter ATIF trajectory conversion, `mini-swe-agent` integration, a DeepSWE-style local-dataset execution path, and a visual architecture explainer.
- [DeepSWE: Long-Horizon Software Engineering Benchmark](../docs/benchmarks/deepswe/index.md) — Long-horizon coding benchmark with original tasks, 91 repositories across five languages, behavioral verifiers, a visual explainer of benchmark rationale and workflow, publication leaderboard snapshot, qualitative failure analysis, and limitations.
- [DeepSWE v1.1: Execution and Scoring Changes](../docs/benchmarks/deepswe-v1-1/index.md) — DeepSWE execution-and-grading update with a visual explainer of committed-patch isolated verification, CTRF structured test reports, cleaner `main`-branch git environment, updated July 1, 2026 leaderboard snapshot, and v1 versus v1.1 impact.
- [τ-bench: Tool-Agent-User Interaction Benchmark](../docs/benchmarks/tau-bench.md) — Original benchmark: two customer-service domains (retail and airline), pass^k metric, empirical results across 12 models, and failure analysis.
- [τ²-Bench: Mechanism and Design](../docs/benchmarks/tau2-bench-mechanism.md) — Dec-POMDP formalism, dual-control domain, task generation, and evaluation methodology.
- [τ-Voice: Full-Duplex Voice Agent Benchmark](../docs/benchmarks/tau-voice.md) — Extends τ²-bench to voice: tick-based orchestrator, controllable voice user simulator, 278 tasks across retail/airline/telecom, empirical results for Google/OpenAI/xAI, acoustic ablations, and error analysis.

## Inference Systems and Frameworks

- [Frameworks](../docs/frameworks/index.md) — Category overview for LLM serving and programming framework pages.
- [Harbor: Agent Evaluation Framework Design](../docs/frameworks/harbor-framework/index.md) — Why Harbor exists beyond Terminal-Bench, its task-centric design philosophy, `Job -> Trial -> Agent/Environment/Verifier` architecture, artifact and verifier isolation model, and the editable draw.io architecture asset.
- [SGLang: Structured Language Model Programs](../docs/frameworks/sglang-framework.md) — Framework architecture, Python-embedded programming model, RadixAttention KV cache reuse, compressed FSM decoding, API speculative execution, and performance results.
- [vLLM Code Learning Path and Request Flow](../docs/frameworks/vllm-code-learning-path.md) — Current vLLM codebase map, request lifecycle from OpenAI API entrypoint to worker execution, and an achievement-driven path to build a mini vLLM.
- [vLLM: PagedAttention Serving Framework](../docs/frameworks/vllm-framework.md) — LLM serving framework design, PagedAttention KV-cache paging, block tables, copy-on-write sharing, scheduling/preemption, distributed execution, and throughput results.

## Algorithms

- [Algorithms](../docs/algorithms/index.md) — Category overview for inference algorithm and kernel pages.
- [Matrix Exponentiation for Linear Transitions](../docs/algorithms/matrix-exponentiation.md) — Binary matrix exponentiation, transition-matrix construction, linear recurrences, augmented state vectors, and fixed linear dynamic programming.
- [FlashAttention: IO-Aware Exact Attention](../docs/algorithms/flashattention.md) — Original IO-aware exact attention algorithm: tiling, online softmax, recomputation, IO complexity, block-sparse extension, and training/runtime results.
- [FlashAttention-2: Better Parallelism and Work Partitioning](../docs/algorithms/flashattention-2.md) — Exact attention kernel optimization: reduced non-matmul overhead, sequence-parallel thread blocks, warp-level work partitioning, causal block skipping, and A100/H100 performance results.
- [FlashAttention-3: Hopper Asynchrony and FP8 Attention](../docs/algorithms/flashattention-3.md) — Hopper attention kernel design: warp specialization, TMA/WGMMA asynchrony, GEMM-softmax overlap, FP8 block quantization, incoherent processing, and speed/accuracy results.
- [FlashAttention-4: Blackwell Attention Kernel Co-Design](../docs/algorithms/flashattention-4.md) — Exact attention algorithm and Blackwell kernel design: asynchronous MMA/TMEM pipeline, exponential emulation, conditional softmax rescaling, 2-CTA backward pass, LPT scheduling, and performance results.

## Training

- [Training](../docs/training/index.md) — Category overview for model training, fine-tuning, transfer learning, and generalization pages.
- [Intrinsic Dimensionality and Language Model Fine-Tuning](../docs/training/intrinsic-dimensionality-fine-tuning/index.md) — Intrinsic-dimension view of pretrained language model fine-tuning: DID/SAID subspace training, low `d90` task dimensions, pretraining as downstream task compression, model-size trends, generalization correlations, and an editable Draw.io explainer.

## Hardware and Numerics

- [Hardware and Numerics](../docs/hardware/index.md) — Category overview for hardware and numerics pages.
- [FlatQuant: Fast Learnable Affine Quantization](../docs/hardware/flatquant.md) — Post-training LLM quantization method: learnable affine transformations, Kronecker factorization, per-channel scaling, learnable clipping, fused kernels, W4A4 accuracy, and inference latency results.
- [NVFP4: Blackwell 4-Bit Floating Point](../docs/hardware/nvfp4.md) — NVIDIA Blackwell NVFP4 format, hierarchical FP8/FP32 scaling, memory benefits, Transformer Engine training recipe, RHT, stochastic rounding, and hardware support.

## Sources
