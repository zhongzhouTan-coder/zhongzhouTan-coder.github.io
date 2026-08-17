---
title: "Wiki Index"
summary: "Top-level index of knowledge-base pages grouped by topic."
layout: default
confidence: high
sources:
  - AGENTS.md
  - .github/instructions/logs-maintenance.instructions.md
updated: 2026-08-17
---

# Wiki Index

## Meta

- [Knowledge Base Introduction](../README.md)

## Benchmarks

- [Benchmarks](../benchmarks/index.md) — Category overview for benchmark pages.
- [Agent Evaluation Benchmarks](../benchmarks/agent-eval/index.md) — Category hub for coding-agent and tool-use-agent benchmarks and harnesses.
- [Pier: Coding-Agent Evaluation Harness](../benchmarks/agent-eval/pier/index.md) — Harbor-compatible coding-agent evaluation harness focused on installed agents in sandboxed tasks, stricter ATIF trajectory conversion, `mini-swe-agent` integration, a DeepSWE-style local-dataset execution path, and a visual architecture explainer.
- [DeepSWE: Long-Horizon Software Engineering Benchmark](../benchmarks/agent-eval/deepswe/index.md) — Long-horizon coding benchmark with original tasks, 91 repositories across five languages, behavioral verifiers, a visual explainer of benchmark rationale and workflow, publication leaderboard snapshot, qualitative failure analysis, and limitations.
- [DeepSWE v1.1: Execution and Scoring Changes](../benchmarks/agent-eval/deepswe-v1-1/index.md) — DeepSWE execution-and-grading update with a visual explainer of committed-patch isolated verification, CTRF structured test reports, cleaner `main`-branch git environment, updated July 1, 2026 leaderboard snapshot, and v1 versus v1.1 impact.
- [τ-bench: Tool-Agent-User Interaction Benchmark](../benchmarks/agent-eval/tau-bench.md) — Original benchmark: two customer-service domains (retail and airline), pass^k metric, empirical results across 12 models, and failure analysis.
- [τ²-Bench: Mechanism and Design](../benchmarks/agent-eval/tau2-bench-mechanism.md) — Dec-POMDP formalism, dual-control domain, task generation, and evaluation methodology.
- [τ-Voice: Full-Duplex Voice Agent Benchmark](../benchmarks/agent-eval/tau-voice.md) — Extends τ²-bench to voice: tick-based orchestrator, controllable voice user simulator, 278 tasks across retail/airline/telecom, empirical results for Google/OpenAI/xAI, acoustic ablations, and error analysis.
- [AutoJudger: Agent-Driven Efficient MLLM Benchmarking](../benchmarks/agent-eval/autojudger.md) — Agent-driven framework for adaptive MLLM evaluation using IRT difficulty estimation and semantic-aware retrieval, achieving 90%+ ranking accuracy at ~4% data usage.
- [HORIZON: Agentic Hardware Design as Repository-Level Code Evolution](../benchmarks/agent-eval/agentic-hardware-design/index.md) — Git-traced RTL self-evolution with executable feedback, 100% best-so-far completion across evaluated suites, convergence-cost analysis, and reward-hacking limitations.
- [Serving Performance Benchmarks](../benchmarks/serving-perf/index.md) — Category hub for inference serving stress-testing and performance benchmarking tools.
- [EvalScope Perf: LLM Inference Stress Testing](../benchmarks/serving-perf/evalscope-perf.md) — Comprehensive model inference stress-testing tool: closed-loop and open-loop modes, SLA binary-search auto-tuning, multi-turn conversation benchmarking, embedding/rerank/multi-modal dataset support, vLLM bench parity comparison, and metrics coverage analysis.
- [AISBench Benchmark vs. EvalScope Perf](../benchmarks/serving-perf/aisbench-vs-evalscope-perf.md) — Competitive analysis for AISBench across load semantics, traffic and trace workloads, metrics, extensibility, operations, product gaps, validation methodology, and prioritized roadmap.
- [AISBench First-Class Performance Roadmap](../benchmarks/serving-perf/aisbench-first-class-roadmap/index.md) — Presentation-ready decision document defining five core AISBench implementation priorities, detailed EvalScope feature lessons, architecture, acceptance criteria, and phased delivery plan.
- [AISBench First-Class 性能能力路线图（中文）](../benchmarks/serving-perf/aisbench-first-class-roadmap/zh-cn.md) — 中文汇报版本，完整说明五项 First-Class 能力、EvalScope 功能分析、实现契约、验收标准、架构和分阶段交付计划。

## Inference Systems and Frameworks

- [CUDA Graphs in PyTorch: Capture Once, Replay Many](../frameworks/cuda/cuda-graphs/index.md) — How stream capture bundles repeated GPU work, why replay needs static shapes and stable addresses, and when PyTorch graphing removes CPU launch overhead.
- [CUDA Programming Model: From Host to SM, Warp, and Memory](../frameworks/cuda/index.md) — Hardware-grounded guide to host/device execution, grid and block scheduling, SM and warp behavior, tile programming, and the GPU memory hierarchy.
- [CUDA Tile IR: The Design Philosophy of Tile Programming](../frameworks/cuda/tile-ir/index.md) — Tile-block execution, tensor-first values, structured tensor views, compiler-owned hardware mapping, and performance-portability goals.
- [Qwen3.5 MTP: Drafting and Target-Model Verification](../frameworks/vllm-ascend/qwen3.5-mtp.md) — MTP proposal and target-logit verification path: Qwen3.5 drafts from target hidden states, while vLLM's rejection sampler commits only an accepted prefix.
- [Frameworks](../frameworks/index.md) — Category overview for LLM serving and programming framework pages.
- [Sarathi: Chunked Prefills for Efficient LLM Inference](../frameworks/sarathi/index.md) — Chunked prefills and decode-maximal batching improve decode utilization and reduce pipeline bubbles.
- [vLLM](../frameworks/vllm/index.md) — Category hub for vLLM serving framework pages.
- [vLLM Architecture and Code Organization Overview](../frameworks/vllm/vllm-overview.md) — Start here: the six-layer mental model, the `vllm/` and `vllm/v1/` directory maps, component-by-component responsibilities, the request lifecycle across processes, and the main extension points.
- [vLLM MHA Code Path: From QKV to Paged KV Cache](../frameworks/vllm/vllm-mha-code-path.md) — Code-reading trace of decoder MHA/GQA/MQA across model projection, serving metadata, paged KV cache updates, backend kernel dispatch, and output projection.
- [vLLM: PagedAttention Serving Framework](../frameworks/vllm/vllm-framework.md) — LLM serving framework design, PagedAttention KV-cache paging, block tables, copy-on-write sharing, scheduling/preemption, distributed execution, and throughput results.
- [vLLM Continuous Batching: Scheduler, KV Blocks, and Runtime Flow](../frameworks/vllm/vllm-continuous-batching/index.md) — Current V1 iteration loop, token and sequence budgets, running/waiting admission, chunked prefill, paged KV-slot allocation, persistent worker batches, completion, and preemption.
- [vLLM Prefill/Decode Disaggregated Deployment Path](../frameworks/vllm/prefill-decode-disaggregated-deployment/index.md) — Deployment trace across the router, prefill pool, NIXL KV-transfer plane, and decode pool, with pull/push modes, scaling, compatibility gates, and failure handling.
- [vLLM Block Table Management: From PagedAttention to the V1 KV Cache Stack](../frameworks/vllm/vllm-block-management/index.md) — Deep dive into the V1 block pool, per-group KV cache managers, hash-based prefix caching, refcount/copy-on-write sharing, block recycling, and the worker-side block table tensors consumed by PagedAttention kernels.
- [vLLM DCP and PCP: Decode and Prefill Context Parallelism](../frameworks/vllm/vllm-context-parallelism.md) — Code-reading map of DCP KV ownership, PCP batch partitioning, exact LSE attention merging, cache block scaling, and support boundaries.
- [vLLM DCP Attention: From Local LSE to Exact Global Output](../frameworks/vllm/dcp-attention/index.md) — Focused derivation of the DCP local-attention, LSE-correction, and AG+RS/AG+AR reduction path.
- [vLLM Kimi K3 Code Reading Map](../frameworks/vllm/vllm-kimi-k3-code-reading.md) — Upstream vLLM Kimi K3 implementation map covering XTML request handling, multimodal wrapper, KimiLinear text model, hybrid KDA/MLA attention, latent MoE, DeepGEMM MegaMoE, MTP, and K3-specific kernels.
- [MiniMax GQA W4A4 Quantization Path: GPU (vLLM) and NPU (vllm-ascend)](../frameworks/vllm/minimax-gqa-w4a4-quantization-path.md) — Beginner-oriented explanation of what W4A4 quantizes, when offline/load/runtime work happens, how GPU and NPU paths differ, and which hardware fallbacks prevent true W4A4 execution.
- [vLLM Ascend](../frameworks/vllm-ascend/index.md) — Category hub for vLLM's Ascend NPU port.
- [vLLM-Ascend Architecture: How the Ascend NPU Port Integrates with vLLM](../frameworks/vllm-ascend/architecture.md) — Five integration mechanisms (plugin registration, NPUPlatform, ModelRegistry, monkey-patches, custom backends), the attention backend selection rule (FIA / MLA / SFA / DSA), end-to-end execution flow, ACL graph capture, HCCL communication, and what upstream vLLM code is reused as-is.
- [vLLM-Ascend Kimi K3 MoE Forward Insight](../frameworks/vllm-ascend/kimi-k3-moe-forward.md) — Latest-code insight for the Kimi K3-style routed-MoE forward substrate in vllm-ascend: patched FusedMoE construction, typed MoE stage contracts, Ascend routing, token dispatch, grouped MLP compute, routed-expert capture, Fused MC2, and dynamic EPLB.
- [DeepSeek-V4 Lightning Indexer C8 Quantization](../frameworks/vllm-ascend/deepseek-v4-lightning-indexer-c8.md) — How the DeepSeek-V4 Lightning Indexer runs on an 8-bit key cache and query in vllm-ascend: INT8 + FP16 scales on 910B/A2/A3, FP8 e4m3 + FP32 scales on A5, the quantized top-k custom operators, and the C4-vs-C8 naming.
- [DeepSeek-V4 Inference on Ascend: The DSA Serving Stack](../frameworks/vllm-ascend/deepseek-v4-inference.md) — End-to-end DeepSeek-V4 serving on Ascend NPUs: the model override with mHC hyper-connections, hybrid c4/c128 compressor layers, the AscendDSA prefill/decode flow, the five-type heterogeneous KV cache, the sparse-attention custom operator, and the MTP draft model.
- [Qwen3.5 / Qwen3.6 Inference Path on vLLM Ascend](../frameworks/vllm-ascend/qwen3.5-qwen3.6-inference.md) — How the shared `qwen3_5`-family models (Qwen3.5-27B / Qwen3.6-27B dense hybrid Mamba-Transformer, multimodal; Qwen3.5/Qwen3.6-35B-A3B and Qwen3.5-397B-A17B sparse MoE) run on Ascend by reusing the `qwen3_5` / `qwen3_5_moe` model types: GDN linear attention plus FIA full attention, ModelSlim W8A8 quantization, `qwen3_5_mtp` speculative decoding, and ACL-graph capture at pinned revision `9a52ca5fc36c`.
- [vLLM-Ascend Prefill and Decode Scheduling: Qwen3.5 GQA](../frameworks/vllm-ascend/prefill-decode-scheduling-qwen3.5.md) — Token-budget scheduling, chunked prefills, mixed FIA GQA layouts, and the parallel GDN recurrent path.
- [Triton: Tiled GPU Kernel Language and Compiler](../frameworks/triton/index.md) — Original Triton language (MAPL 2019): Triton-C tile-programming frontend, Triton-IR tile-level LLVM extensions, Triton-JIT compiler with hierarchical tiling, memory coalescing, shared memory allocation/synchronization, and auto-tuning; achieves cuBLAS/cuDNN parity for matmul and convolution.
- [Triton in Practice: How vLLM and vllm-ascend Use Triton](../frameworks/triton/triton-in-vllm.md) — Codebase-driven tour: vLLM's triton_utils infrastructure, custom op registration, ~163 kernel files across 12 categories, universal coding patterns with concrete examples, and vllm-ascend's CANN-backend adaptation with dual Triton+AscendC strategy.
- [TileLang Design and Code Learning Path](../frameworks/tilelang/index.md) — Pinned code-reading guide to TileLang's Python DSL, TIRX construction, backend-owned lowering pipelines, JIT specialization and caching, execution adapters, extension points, and practical learning stages.
- [TileLang-Ascend: Ascend Backend and TileLang Integration](../frameworks/tilelang-ascend/index.md) — Ascend C/PTO compiler paths, CANN/Bisheng source compilation, Cython launch, and the current relationship to upstream TileLang.
- [Triton Ascend: Ascend NPU Backend for Triton](../frameworks/triton-ascend/index.md) — Beginner-friendly architecture tour: five-layer design (backend registration, compilation pipeline, MLIR passes, runtime driver, CANN extensions), TTIR→HIVM→LLVM→Linalg→Bisheng compilation flow, SIMD/SIMT/Unstructured-in-SIMT three-mode compilation, and relationship to vllm-ascend.
- [Triton Ascend 算子机制学习路径](../frameworks/triton-ascend/operator-mechanisms.md) — 图解 AIC/AIV、UB/L1/L0、MTE 和异步指令队列，并用代码追踪、容量估算、profiling 症状和分阶段练习讲解 Vector、Cube 与 CV Fusion。
- [CANNBot Skills: Triton Ascend Development Workflow](../frameworks/triton-ascend/cannbot-skills-workflow.md) — Deep insight into CANNBot's seven Triton-domain skills and the triton-op-generator plugin pipeline: task extraction, algorithm sketch design, pure-Triton code generation, precision verification with five-category decision matrix, five-stage ULP isolation debugging, 25-point ordered latency optimization, and simulator-driven bottleneck diagnosis.
- [SGLang: Structured Language Model Programs](../frameworks/sglang/index.md) — Framework architecture, Python-embedded programming model, RadixAttention KV cache reuse, compressed FSM decoding, API speculative execution, and performance results.
- [DeepSeek](../frameworks/deepseek/index.md) — Category hub for DeepSeek model implementation readings.
- [DeepSeek V4 Attention: Code Reading Map](../frameworks/deepseek/v4-attention-code-reading.md) — Navigable implementation map of DeepSeek V4's hybrid compressed attention across vLLM (NVIDIA/AMD/XPU) and vllm-ascend (Ascend NPU), covering CSA/HCA compressors, sparse MLA backends, heterogeneous KV cache, multi-stream overlap, and platform-specific kernel dispatch.
- [DSpark: Confidence-Scheduled Speculative Decoding](../frameworks/dspark/index.md) — PDF-extraction-backed re-insight of semi-autoregressive drafting, calibrated prefix survival, causal two-step-delayed scheduling, and the DeepSeek-V4 production throughput frontier.
- [Harbor: Agent Evaluation Framework (Code Reading)](../frameworks/harbor/index.md) — Repository-backed tour of Harbor's task packaging model, `Job -> JobPlan -> TrialQueue -> Trial -> Agent/Environment/Verifier` runtime, local/git/package/registry distribution, multi-step trials, and compile/exec workflows at pinned revision `97e65926410b`.

## Algorithms

- [Algorithms](../algorithms/index.md) — Category overview for inference algorithm and kernel pages.
- [Attention Foundations](../algorithms/foundations/index.md) — Category hub for foundational sequence-modeling concepts.
- [The Transformer: Attention Is All You Need](../algorithms/foundations/transformer.md) — The foundational architecture: scaled dot-product attention, multi-head self-attention, sinusoidal positional encoding, encoder-decoder stacks, and the training recipe that launched modern LLMs.
- [Layer Normalization in Transformers](../algorithms/foundations/layer-normalization/index.md) — Why Transformers normalize each token across hidden features, not across batch members, with axis diagrams and a worked computation.
- [The Softmax Function: Properties, Motivation, and Interpretation](../algorithms/foundations/softmax.md) — Tutorial covering score-difference semantics, α parameter interpretation, three conceptual justifications (Gumbel noise, maximum entropy, exploration-exploitation), IO vs. IM model taxonomy, and complete mathematical properties.
- [Recurrent Neural Networks: From RNN to LSTM](../algorithms/foundations/recurrent-neural-networks/index.md) — Sequence processing through shared recurrent weights and hidden state, the long-term dependency problem, LSTM gating, and the conceptual bridge to linear attention's RNN mode.
- [Kronecker Product](../algorithms/kronecker-product.md) — The block-structured matrix product A⊗B (matrix direct product), foundational to tensor factorization and the Kronecker factorization trick behind FlatQuant's learnable affine transforms.
- [FlashAttention](../algorithms/flashattention/index.md) — Category hub for the FlashAttention algorithm and kernel family.
- [FlashAttention: IO-Aware Exact Attention](../algorithms/flashattention/flashattention.md) — Original IO-aware exact attention algorithm: tiling, online softmax, recomputation, IO complexity, block-sparse extension, landscape evolutionary tree, and training/runtime results.
- [FlashAttention-2: Better Parallelism and Work Partitioning](../algorithms/flashattention/flashattention-2.md) — Exact attention kernel optimization: reduced non-matmul overhead, sequence-parallel thread blocks, warp-level work partitioning, causal block skipping, Big Picture FA1→FA2 comparison diagram, landscape of GPU utilization gap closure, and A100/H100 performance results.
- [FlashAttention-3: Hopper Asynchrony and FP8 Attention](../algorithms/flashattention/flashattention-3.md) — Hopper attention kernel design: warp specialization, TMA/WGMMA asynchrony, GEMM-softmax overlap, FP8 block quantization, incoherent processing, Big Picture async pipeline diagram, landscape of generation-specific hardware exploitation, and speed/accuracy results.
- [FlashAttention-4: Blackwell Attention Kernel Co-Design](../algorithms/flashattention/flashattention-4.md) — Exact attention algorithm and Blackwell kernel design: asymmetric scaling response, exponential emulation, conditional softmax rescaling, TMEM-based pipelining, 2-CTA backward pass, LPT scheduling, landscape of shifting hardware bottlenecks, and performance results.
- [Attention Variants](../algorithms/attention-variants/index.md) — Category hub for attention designs that reduce query/key redundancy or KV-cache pressure.
- [Multi-Query Attention: One Write-Head is All You Need](../algorithms/attention-variants/multi-query-attention.md) — Shares one K/V across all attention heads, shrinking incremental decoder memory bandwidth 8× for a 12× inference speedup with negligible quality loss.
- [Grouped-Query Attention in Llama 2](../algorithms/attention-variants/grouped-query-attention/index.md) — Llama 2's 34B/70B GQA decision: 8 KV groups, 30B MHA/MQA/GQA ablation, higher large-batch throughput, and simpler 8-GPU tensor-parallel serving than MQA.
- [Collaborative Multi-Head Attention](../algorithms/attention-variants/collaborative-attention.md) — Redesigns MHA with shared key/query projections and per-head mixing vectors, enabling 4× compression of Q/K dimensions; CP tensor decomposition for post-hoc conversion of pretrained models.
- [DeepSeek-V2 Multi-Head Latent Attention](../algorithms/attention-variants/deepseek-v2-mla.md) — DeepSeek-V2's MLA design: low-rank joint K/V latent cache, decoupled RoPE, query compression, MoE serving context, 93.3% KV-cache reduction, and 5.76× maximum generation throughput versus DeepSeek 67B.
- [Transformers Are RNNs: Linear Attention](../algorithms/linear-attention/index.md) — Kernel feature maps, associative reordering, causal recurrent states, linear sequence complexity, benchmark interpretation, and fixed-capacity retrieval limits.
- [Linear Attention Without Softmax: Su Jianlin's Survey](../algorithms/linear-attention/linear-attention-without-softmax.md) — 苏剑林's blog survey identifying softmax as the root cause of attention's O(n²) complexity; catalogs three linear attention families (kernel maps, double-softmax, cosine-similarity Taylor approximation) and their autoregressive generation support.
- [Matrix Exponentiation for Linear Transitions](../algorithms/linear-attention/matrix-exponentiation.md) — Binary matrix exponentiation, transition-matrix construction, linear recurrences, augmented state vectors, and fixed linear dynamic programming.
- [DeepSeek-V3.2: Sparse Attention, Scaled RL, and Thinking in Tool-Use](../algorithms/deepseek-v3.2/index.md) — Three innovations: DSA sparse attention with lightning indexer and top-k token selection, scaled GRPO with four MoE stabilization tricks, and cold-start + synthetic agentic task pipeline unifying reasoning with tool-use.
- [Context Parallelism for Scalable Million-Token Inference](../algorithms/context-parallelism/index.md) — Exact pass-KV/pass-Q ring attention, load-balanced context sharding, cache-aware traffic selection, and 1M-token prefill scaling on 128 H100 GPUs.

## Training

- [Training](../training/index.md) — Category overview for model training, fine-tuning, transfer learning, and generalization pages.
- [Training Parallelism](../training/parallelism/index.md) — Category hub for data, tensor, pipeline, and sequence parallelism techniques.
- [Megatron-LM: GPU-Cluster Training Parallelism](../training/parallelism/megatron-lm/index.md) — Covers both Megatron-LM papers: intra-layer tensor parallelism with f/g conjugate operators and column-parallel GEMM splitting (2019), and PTD-P recipe for trillion-parameter GPT models (2021).
- [GPipe: Micro-Batch Pipeline Parallelism](../training/parallelism/gpipe/index.md) — Synchronous micro-batch pipeline parallelism with activation recomputation: splits mini-batches into micro-batches, pipelines them through partitioned layers, and applies synchronous gradient updates for near-linear speedup.
- [Sequence Parallelism: Splitting Sequences Across GPUs](../training/parallelism/sequence-parallelism/index.md) — Distributes input sequence chunks across GPUs with Ring Self-Attention (RSA), introducing the fourth parallelism dimension alongside data, pipeline, and tensor parallelism. Achieves 13.7× larger batch size and 3.0× longer sequences than tensor parallelism.
- [Foundation Models](../training/foundation-models/index.md) — Category hub for classic decoder-only foundation model papers.
- [GPT-1: Improving Language Understanding by Generative Pre-Training](../training/foundation-models/gpt-1.md) — Introduces the decoder-only Transformer, the pre-train + fine-tune paradigm, task-agnostic input transformations, long-contiguous-text motivation, and transfer ablations.
- [GPT-2: Language Models are Unsupervised Multitask Learners](../training/foundation-models/gpt-2.md) — Scales to 1.5B on WebText; demonstrates zero-shot task transfer, byte-level BPE evaluation, prompt-only task conditioning, and contamination analysis.
- [GPT-3: Language Models are Few-Shot Learners](../training/foundation-models/gpt-3.md) — Scales to 175B; demonstrates in-context few-shot learning, scaling-law behavior, prompt-format evaluation settings, and limitations of context-only adaptation.
- [LLaMA: Open and Efficient Foundation Language Models](../training/foundation-models/llama.md) — Original LLaMA family: 7B-65B decoder-only models trained on 1.0T-1.4T public-data tokens, inference-budget motivation, architecture defaults, efficient training implementation, benchmark comparisons, and safety limitations.
- [DeepSeek](../training/deepseek/index.md) — Category hub for DeepSeek model training papers.
- [DeepSeek-V4: Million-Token Context via Hybrid Compressed Attention](../training/deepseek/deepseek-v4/index.md) — 1.6T/284B MoE models with CSA+HCA hybrid attention, mHC, and Muon optimizer; 27% FLOPs and 10% KV cache of V3.2 at 1M-token contexts.
- [mHC: Manifold-Constrained Hyper-Connections](../training/mhc/index.md) — Widens the residual stream to n parallel streams and constrains the mixing matrix to a doubly stochastic manifold via Sinkhorn-Knopp, restoring the identity-mapping property; 27B MoE final loss 0.021 lower than baseline at 6.7% overhead.
- [Kimi](../training/kimi/index.md) — Category hub for the Kimi model family.
- [Kimi Linear: Expressive Efficient Attention Architecture](../training/kimi/kimi-linear/index.md) — Hybrid linear attention: KDA with channel-wise gating extends Gated DeltaNet, 3:1 KDA-to-MLA layer ratio with NoPE, 48B MoE (3B active); for the first time outperforms full attention across short/long/RL regimes, 6.3× decoding speedup at 1M context.
- [Kimi K3: Open 3T-Class Frontier Model](../training/kimi/kimi-k3/index.md) — 2.8T/104B-active native multimodal MoE with hybrid KDA/MLA attention, Stable LatentMoE, 1M context, multi-effort agentic RL, MoonEP balanced expert-parallel training, and long-rollout cache/sandbox infrastructure.
- [Efficient Attention Training](../training/efficient-attention/index.md) — Category hub for training approaches built around efficient attention mechanisms.
- [MiniMax Sparse Attention (MSA)](../training/efficient-attention/minimax-sparse-attention/index.md) — Blockwise sparse attention co-designed with GQA: lightweight Index Branch selects top-k KV blocks per GQA group, Main Branch computes exact softmax attention over only the selected blocks, trained with KL alignment loss; 28.4× FLOPs reduction and 14.2× prefill / 7.6× decode speedup at 1M context on a 109B MoE model with native multimodal training.
- [SWAT: Sliding Window Attention Training](../training/efficient-attention/swat-sliding-window-attention/index.md) — Sigmoid-based sliding window attention training: replaces softmax with sigmoid to eliminate attention sink, combines balanced bidirectional ALiBi with RoPE for training stability; SOTA on 8 commonsense reasoning benchmarks versus linear recurrent baselines at 340M/760M scale.
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](../training/efficient-attention/gated-delta-networks/index.md) — Fixed-state recurrent memory combining global adaptive decay with key-targeted correction, decay-aware chunkwise WY training, and SWA/Mamba2 hybrids.
- [Fine-Tuning and Adaptation](../training/fine-tuning/index.md) — Category hub for fine-tuning, transfer learning, and self-evolution methods.
- [Intrinsic Dimensionality and Language Model Fine-Tuning](../training/fine-tuning/intrinsic-dimensionality-fine-tuning/index.md) — Intrinsic-dimension view of pretrained language model fine-tuning: DID/SAID subspace training, low `d90` task dimensions, pretraining as downstream task compression, model-size trends, generalization correlations, and an editable Draw.io explainer.
- [Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Skills](../training/fine-tuning/socratic-swe/index.md) — Closed-loop self-evolution framework: trace-derived Agent Skill Registry, skill-guided Generator with four-stage Verifier Gate, gradient-aligned Generator reward via cosine similarity to validation gradient, GDPO-normalized Solver reward, and 50.40% on SWE-bench Verified across three iterations.

## Hardware and Numerics

- [Microscaling (MX) Formats: Block Floating Point for AI Hardware](../hardware/quantization/microscaling-mx-formats/index.md) — OCP MX v1.0 block contract, concrete FP4/FP6/FP8/INT8 encodings, conversion and dot-product semantics, implementation-defined boundaries, and FPGA/MSFP context.
- [Hardware and Numerics](../hardware/index.md) — Category overview for hardware and numerics pages.
- [Quantization](../hardware/quantization/index.md) — Category hub for post-training quantization methods and low-precision numeric formats.
- [GPTQ: Second-Order Weight Quantization at LLM Scale](../hardware/quantization/gptq/index.md) — Shared column ordering, lazy block updates, Cholesky-stabilized error compensation, 175B-scale calibration, 3–4-bit accuracy, and packed-weight decode results.
- [FlatQuant: Fast Learnable Affine Quantization](../hardware/quantization/flatquant/index.md) — Post-training LLM quantization method: learnable affine transformations, Kronecker factorization, per-channel scaling, learnable clipping, fused kernels, W4A4 accuracy, and inference latency results.
- [NVFP4: Blackwell 4-Bit Floating Point](../hardware/quantization/nvfp4.md) — NVIDIA Blackwell NVFP4 format with two-level hierarchical FP8/FP32 scaling, fractional E4M3 vs. power-of-two E8M0 comparison, 16-element micro-block quantization, Random Hadamard Transform, stochastic rounding, 2D weight scaling, GEMM layout constraints, distributed training behavior, and deployment ecosystem.
- [Spatial GEMM: Blocked Outer-Product Matrix Multiply](../hardware/spatial-gemm.md) — Blocked outer-product GEMM in the Spatial DSL: output tiling, MemFold/MemReduce pipelining, triple buffering, and multi-dimensional SRAM banking.

## Terms

- [Terms Glossary](../terms/index.md) — Alphabetical glossary of cross-paper technical terms with concise definitions and backlinks to the papers that use them.
- [GPTQ](../terms/gptq.md) — Second-order post-training weight quantization with activation-derived error compensation and scalable block updates.
- [Post-Training Quantization](../terms/post-training-quantization.md) — Quantization applied to a trained model using calibration data or weight statistics instead of full retraining.
- [Continuous Batching](../terms/continuous-batching.md) — Iteration-level LLM-serving scheduler that rebuilds active work every model step.
- [Block Table](../terms/block-table.md) — The per-request logical-to-physical mapping that tells paged attention kernels which physical KV block holds each logical block of a sequence.
- [PagedAttention](../terms/pagedattention.md) — Attention algorithm storing KV cache in fixed-size non-contiguous blocks addressed through a per-request block table.
- [Kimi Delta Attention](../terms/kimi-delta-attention.md) — Gated linear-attention mechanism that extends delta-rule recurrent memory with channel-wise decay and hardware-efficient chunkwise computation.
- [Microbatch](../terms/microbatch.md) — A small chunk of a training batch used to enable pipeline parallelism; the unit of work in a pipeline schedule.
- [Mixture of Experts](../terms/mixture-of-experts.md) — Sparse model architecture that routes each token through a few experts to increase total capacity without activating every parameter.
- [Scatter/Gather](../terms/scatter-gather.md) — Cross-node communication optimization that avoids redundant activation transfers over slow inter-node links.
- [General Matrix Multiply (GEMM)](../terms/gemm.md) — The dense multiply-accumulate kernel C = A×B whose execution rate is the standard performance reference for linear-algebra workloads on GPUs and NPUs.
- [Inner Product](../terms/inner-product.md) — The scalar dot product Σ xᵢyᵢ; GEMM is the matrix of inner products between rows of A and columns of B, and attention scores are dot products.
- [Kronecker Product](../terms/kronecker-product.md) — The block-structured matrix product A⊗B that builds a large matrix from two smaller ones by scaling copies of B by the entries of A.
- [Matrix Tiling](../terms/matrix-tiling.md) — Blocking a GEMM (or any kernel) into tiles that fit on-chip SRAM and registers so operands are loaded from global memory few times and reused many times.
- [Memory Banking](../terms/memory-banking.md) — Partitioning on-chip SRAM into banks so parallel accesses to different addresses hit different banks in the same cycle, avoiding bank conflicts.
- [Outer Product](../terms/outer-product.md) — A rank-1 matrix u vᵀ formed from two vectors; GEMM can be computed by accumulating outer products of columns of A with rows of B.
- [Systolic Array](../terms/systolic-array.md) — A regular grid of processing elements where data flows rhythmically between neighbors so each weight is reused across many multiply-accumulates without re-fetching.

## Sources
