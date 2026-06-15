# Wiki Log

## 2026-05-30

- Added [τ-Voice: Full-Duplex Voice Agent Benchmark](../docs/layer_0/tau-voice.md) to `docs/layer_0/tau-voice.md` as `layer_0` / high confidence, sourced from the τ-Voice preprint (arXiv:2603.13686v1, Mar 2026). Covers tick-based orchestrator design, voice user simulator pipeline (7 personas, audio environment, turn-taking policy), three domains (278 total tasks), three audio-native providers, Clean vs Realistic conditions, acoustic ablation results (accents most damaging at -10pp avg), voice interaction quality metrics, qualitative error analysis (79-90% agent errors), and key conclusions. Classified layer_0 because all numerical results, architecture details, and error annotations are directly stated in the paper.
- Added [τ-bench: Tool-Agent-User Interaction Benchmark](../docs/layer_0/tau-bench.md) to `docs/layer_0/tau-bench.md` as `layer_0` / high confidence, sourced from `raw/benchmark/tau-bench.pdf` (arXiv:2406.12045v1). Covers benchmark architecture, two domains (τ-retail 115 tasks, τ-airline 50 tasks), three-stage construction, pass^k metric definition, full empirical results table for 12 models, method comparison, domain policy ablation, and failure analysis. Classified layer_0 because all facts are directly stated in the paper with quantitative support.

## 2026-05-29

- Added [Knowledge Base Introduction](../docs/README.md) to `docs/README.md` as a meta page classified `layer_0` / high confidence, sourced from `AGENTS.md`. Describes the repo structure, confidence-layer model, navigation workflow, and the docs lint command.
- Added [τ²-Bench: Mechanism and Design](../docs/layer_0/tau2-bench-mechanism.md) to `docs/layer_0/tau2-bench-mechanism.md` as `layer_0` / high confidence, sourced from `raw/benchmark/tau2-bench.pdf`. Covers Dec-POMDP formalism, dual-control architecture, five-stage domain construction, programmatic task generation, user simulator design, ablation modes, and empirical results.

## 2026-06-02

- Migrated GitHub-only agent guidance into `AGENTS.md` so Codex can read repository rules directly, including docs front matter, confidence-layer handling, logs maintenance, and rendering constraints.
- Copied the docs ingest agent and instruction files from `.github/` into `.codex/agents/` and `.codex/instructions/` so the repository has a Codex-local agent setup.
- Copied the PDF skill assets from `.github/skills/pdf/` into `.codex/skills/pdf/` as a Codex-facing repo-local skill mirror.

## 2026-06-11

- Added [NVFP4: Blackwell 4-Bit Floating Point](../docs/layer_0/nvfp4.md) to `docs/layer_0/nvfp4.md` as `layer_0` / high confidence, sourced from NVIDIA's NVFP4 inference blog and Transformer Engine NVFP4 documentation captured in `raw/nvidia/nvfp4-references.md`. Covers E2M1 encoding, FP8 E4M3 block scaling, FP32 tensor scaling, memory reduction claims, Transformer Engine recipe defaults, stochastic rounding, Random Hadamard Transform, GEMM layout, distributed-training behavior, and supported hardware.

## 2026-06-15

- Added [SGLang: Structured Language Model Programs](../docs/layer_0/sglang-framework.md) to `docs/layer_0/sglang-framework.md` as `layer_0` / high confidence, sourced from `raw/sglang/2312.07104v2.pdf`. Covers the SGLang frontend primitives, interpreter/compiler execution modes, RadixAttention KV cache reuse, cache-aware scheduling, compressed finite-state-machine decoding, API speculative execution, evaluation setup, throughput/latency results, production deployment observations, and limitations.
- Added [vLLM: PagedAttention Serving Framework](../docs/layer_0/vllm-framework.md) to `docs/layer_0/vllm-framework.md` as `layer_0` / high confidence, sourced from `raw/vllm/2309.06180v1.pdf`. Covers the PagedAttention algorithm, KV-cache block tables, virtual-memory analogy, copy-on-write sharing for parallel sampling and beam search, shared-prefix reuse, scheduling/preemption, distributed execution, implementation details, evaluation setup, throughput results, ablations, and limitations.
