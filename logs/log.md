# Wiki Log

## 2026-05-30

- Added [τ-Voice: Full-Duplex Voice Agent Benchmark](../docs/benchmarks/tau-voice.md) to `docs/benchmarks/tau-voice.md`, sourced from the τ-Voice preprint (arXiv:2603.13686v1, Mar 2026). Covers tick-based orchestrator design, voice user simulator pipeline (7 personas, audio environment, turn-taking policy), three domains (278 total tasks), three audio-native providers, Clean vs Realistic conditions, acoustic ablation results (accents most damaging at -10pp avg), voice interaction quality metrics, qualitative error analysis (79-90% agent errors), and key conclusions.
- Added [τ-bench: Tool-Agent-User Interaction Benchmark](../docs/benchmarks/tau-bench.md) to `docs/benchmarks/tau-bench.md`, sourced from `raw/benchmark/tau-bench.pdf` (arXiv:2406.12045v1). Covers benchmark architecture, two domains (τ-retail 115 tasks, τ-airline 50 tasks), three-stage construction, pass^k metric definition, full empirical results table for 12 models, method comparison, domain policy ablation, and failure analysis.

## 2026-05-29

- Added [Knowledge Base Introduction](../docs/README.md) to `docs/README.md` as a meta page sourced from `AGENTS.md`. Describes the repo structure, category organization, navigation workflow, and the docs lint command.
- Added [τ²-Bench: Mechanism and Design](../docs/benchmarks/tau2-bench-mechanism.md) to `docs/benchmarks/tau2-bench-mechanism.md`, sourced from `raw/benchmark/tau2-bench.pdf`. Covers Dec-POMDP formalism, dual-control architecture, five-stage domain construction, programmatic task generation, user simulator design, ablation modes, and empirical results.

## 2026-06-02

- Migrated GitHub-only agent guidance into `AGENTS.md` so Codex can read repository rules directly, including docs front matter, category handling, logs maintenance, and rendering constraints.
- Copied the docs ingest agent and instruction files from `.github/` into `.codex/agents/` and `.codex/instructions/` so the repository has a Codex-local agent setup.
- Copied the PDF skill assets from `.github/skills/pdf/` into `.codex/skills/pdf/` as a Codex-facing repo-local skill mirror.

## 2026-06-11

- Added [NVFP4: Blackwell 4-Bit Floating Point](../docs/hardware/nvfp4.md) to `docs/hardware/nvfp4.md`, sourced from NVIDIA's NVFP4 inference blog and Transformer Engine NVFP4 documentation captured in `raw/nvidia/nvfp4-references.md`. Covers E2M1 encoding, FP8 E4M3 block scaling, FP32 tensor scaling, memory reduction claims, Transformer Engine recipe defaults, stochastic rounding, Random Hadamard Transform, GEMM layout, distributed-training behavior, and supported hardware.

## 2026-06-15

- Added [FlashAttention: IO-Aware Exact Attention](../docs/algorithms/flashattention.md) to `docs/algorithms/flashattention.md`, sourced from `raw/infer-algorithm/2205.14135v2.pdf`. Covers the original tiled exact attention algorithm, online softmax statistics, recomputation in backward, IO complexity, block-sparse FlashAttention, empirical training speedups, long-context results, limitations, and relationship to later FlashAttention versions.
- Added [FlashAttention-2: Better Parallelism and Work Partitioning](../docs/algorithms/flashattention-2.md) to `docs/algorithms/flashattention-2.md`, sourced from `raw/infer-algorithm/2307.08691v1.pdf`. Covers reduced non-matmul overhead, logsumexp softmax bookkeeping, sequence-level thread-block parallelism, backward atomic accumulation, warp-level work partitioning, causal block skipping, A100/H100 attention-kernel results, and GPT-style training throughput.
- Added [FlashAttention-3: Hopper Asynchrony and FP8 Attention](../docs/algorithms/flashattention-3.md) to `docs/algorithms/flashattention-3.md`, sourced from `raw/infer-algorithm/2407.08608v2.pdf`. Covers Hopper-specific warp specialization, TMA/WGMMA producer-consumer scheduling, ping-pong and two-stage GEMM-softmax overlap, FP8 layout handling, block quantization, incoherent processing, backward-pass acceleration, empirical speedups, and limitations.
- Added [SGLang: Structured Language Model Programs](../docs/frameworks/sglang-framework.md) to `docs/frameworks/sglang-framework.md`, sourced from `raw/sglang/2312.07104v2.pdf`. Covers the SGLang frontend primitives, interpreter/compiler execution modes, RadixAttention KV cache reuse, cache-aware scheduling, compressed finite-state-machine decoding, API speculative execution, evaluation setup, throughput/latency results, production deployment observations, and limitations.
- Added [vLLM: PagedAttention Serving Framework](../docs/frameworks/vllm-framework.md) to `docs/frameworks/vllm-framework.md`, sourced from `raw/vllm/2309.06180v1.pdf`. Covers the PagedAttention algorithm, KV-cache block tables, virtual-memory analogy, copy-on-write sharing for parallel sampling and beam search, shared-prefix reuse, scheduling/preemption, distributed execution, implementation details, evaluation setup, throughput results, ablations, and limitations.
- Added [FlashAttention-4: Blackwell Attention Kernel Co-Design](../docs/algorithms/flashattention-4.md) to `docs/algorithms/flashattention-4.md`, sourced from `raw/infer-algorithm/2603.05451v1.pdf`. Covers FA4's Blackwell hardware motivation, forward pipeline, software-emulated exponentials, conditional online-softmax rescaling, backward TMEM pipeline, 2-CTA MMA mode, deterministic backward pass, LPT scheduling, CuTe-DSL implementation, benchmark results, and the source's B200/B100 hardware-name inconsistency.
- Refactored [Wiki Index](../logs/index.md) categories, grouping benchmark pages, inference systems/frameworks, attention/kernel algorithms, and hardware/numerics.
- Refactored `docs/` from the previous layer folders into category folders: `docs/benchmarks/`, `docs/frameworks/`, `docs/algorithms/`, and `docs/hardware/`; added category index pages, removed legacy classification front matter, and updated Jekyll navigation/configuration.
