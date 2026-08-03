---
title: "AISBench 性能能力路线图"
summary: "面向最终汇报的 AISBench 产品与实现路线图，重点定义负载语义、度量正确性、SLA 容量发现、证据存储和易用性五项 First-Class 能力。"
layout: default
confidence: medium
sources:
  - https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/performance_benchmark.html
  - https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/advanced_tutorials/stable_stage.html
  - https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/advanced_tutorials/rps_distribution.html
  - https://github.com/AISBench/benchmark
  - https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/
  - https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/parameters.html
  - https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/multi_turn.html
  - https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/sla_auto_tune.html
  - https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/custom.html
  - https://github.com/modelscope/evalscope
updated: 2026-07-17
---

# AISBench 性能能力路线图

**决策问题：** AISBench 必须将哪些能力建设为核心产品能力，才能成为比 EvalScope Perf 更强的推理性能评测工具？
**目标读者：** AISBench 维护者、架构师、评测负责人和工程管理者
**分析方法：** 基于截至 2026-07-17 的双方官方文档与代码仓库；尚未执行同环境、同负载的运行时对比实验

**语言：** [English](index.md)

**相关页面：** [AISBench 与 EvalScope 完整竞品分析](../aisbench-vs-evalscope-perf.md) · [EvalScope Perf 深入分析](../evalscope-perf.md) · [Benchmarks 总览](../../index.md)

## 执行结论

AISBench 应按以下顺序建设 **五项 First-Class 能力**：

1. **指标契约与一致性验证：** 确保每个结果在数学定义和运行语义上可信。
2. **显式负载模型契约：** 将闭环、开环、爬升、突发、压力和 Trace 回放定义为可观察的标准调度模式。
3. **SLA 容量引擎：** 自动发现满足服务目标的最大安全并发数或请求速率。
4. **标准运行与证据存储：** 用可查询的数据模型保存配置、原始事件、请求、响应、派生指标和有效性结论。
5. **单命令产品入口：** 让普通用户无需查找和修改 Python 任务配置，即可完成正确的性能评测。

这些能力之所以必须成为 **First-Class 能力**，是因为所有数据集、服务后端、报告和后续功能都依赖它们。Embedding、Rerank、多模态、分布式压测和在线可视化同样重要，但应构建在这五项契约之上，而不能反过来定义底层契约。

## 全局图

![AISBench First-Class 性能能力路线图](aisbench-first-class-roadmap.drawio.svg)

*① 保留 AISBench 当前的护城河：真实负载、稳态分析、Trace 回放和 Benchmark 任务组合。② 在扩展更多任务类型之前，先建设统一的度量平面。③ 在度量平面之上增加自动容量发现和持久化证据。④ 通过单命令入口暴露整套能力，同时保留高级配置方式。⑤ 最终让 AISBench 同时回答“真实负载下发生了什么”和“满足 SLO 的最大安全负载是多少”。*

## 产品核心判断

**AISBench 已经拥有更难构建的负载建模能力。** 官方文档已经覆盖突发性分布、线性和指数爬升、稳态阶段计算、持续压力测试、按时间戳回放 Mooncake Trace、通过 `hash_ids` 建模前缀局部性、性能指标重计算，以及兼容 OpenCompass 的 Benchmark 任务组合。

EvalScope 的主要优势是产品化封装。它把闭环/开环模式、SLA 搜索、数据集和 API 选择、请求控制、SQLite 存储和自定义扩展统一呈现为一个清晰的性能压测产品。

因此，AISBench 的正确策略是：

> **不要替换更强的负载引擎，而要通过显式契约、自动化决策和持久化证据，把它真正产品化。**

## 什么是 First-Class 能力

只有同时满足以下条件的能力，才能称为 First-Class 能力：

| 要求 | 含义 |
|---|---|
| 稳定的公共契约 | 具备版本化 CLI/配置字段和明确的语义文档 |
| 共享的核心实现 | 所有兼容后端复用相同的调度、事件或结果契约 |
| 可持久化表示 | 运行记录能够保存实际配置和真实观测结果 |
| 可机器验证 | CI 固定样例能够证明语义和公式正确 |
| 可组合 | 能用于合成数据、Benchmark、多轮对话和 Trace 负载 |
| 失败显式可见 | 无效或不可靠结果必须输出明确状态，不能只给出一个数字 |

数据集专用辅助函数、未公开的配置字段或仅存在于输出阶段的计算可能很有用，但不属于 First-Class 能力。

## 优先级 1：指标契约与一致性验证

### 产品承诺

**同一组原始请求事件必须始终产生相同的指标结果，与具体后端和负载适配器无关。**

这是全部能力的基础。如果 TTFT、TPOT、ITL、测试时长、Token 数或成功状态的定义含糊，那么 SLA 搜索、性能回归、跨运行比较和公开 Benchmark 结论都不可信。

### 必须定义的契约

| 指标领域 | AISBench 必须明确的内容 |
|---|---|
| 端到端时延 | 请求开始和响应结束分别采用哪个客户端时间点 |
| TTFT | 首字节、首个 SSE 数据块、首个非空内容或首个解码 Token 中的哪一个 |
| TPOT | 精确公式和分母，特别是是否排除首 Token |
| ITL | 按 Token 还是按 Chunk 计算，多 Token Chunk 如何处理 |
| 吞吐率 | 分子、时间窗口、成功请求过滤规则，以及全程/稳态口径 |
| Token | 客户端 Tokenizer 身份与版本、服务端 Token 计数、两者不一致时的处理 |
| 成功状态 | HTTP、协议错误、空响应、半截流、超时、重试和取消的分类 |
| 分位数 | 样本总体、插值算法、最小样本量和单位 |
| 多轮对话 | 请求、Turn 和 Trace 三个聚合边界 |

### 实现边界

新增版本化的 `MeasurementSpec` 和标准事件序列：

```text
request_scheduled
request_dispatched
connection_acquired
response_headers
first_content
stream_chunk*
request_completed | request_failed | request_cancelled
```

协议适配器负责把后端特有行为转换成标准事件。指标计算器只消费标准事件，不能直接依赖某个后端的响应对象。

### 验收标准

- CI 使用固定 SSE 流验证 E2E、TTFT、TPOT、ITL、Token 和状态。
- Chat 与 Completions 固定样例能够暴露模板和 Token 数差异，而不是隐藏差异。
- 每份报告包含 `measurement_spec_version`。
- 每行指标明确样本总体和时间窗口。
- 无法测量的指标输出 `unavailable` 和原因，不能默认为 0。

### 从 EvalScope 学什么

EvalScope 覆盖了标准指标族，并保存请求级数据以供后续分析。AISBench 应学习的是可持久化的度量契约，而不是只复制指标名称。

## 优先级 2：显式负载模型契约

### 产品承诺

**用户能够独立描述请求到达模型和并发安全上限，报告能够证明客户端是否真正生成了目标负载。**

EvalScope 显式区分：

- 闭环模式限制在途请求，并提供背压。
- 开环模式不等待先前请求完成，按设定速率持续调度新请求。

AISBench 已支持 `request_rate`、突发性、爬升、压力模式和按时间戳调度。当前缺少的是一个统一、具名、稳定的调度器契约。

### 必须支持的模式

| 模式 | 核心语义 | 回答的问题 |
|---|---|---|
| `closed` | 最多 N 个在途请求；请求完成后才释放下一次发送机会 | N 个等待型客户端下，服务表现如何？ |
| `open` | 到达过程独立调度；服务端积压不会降低客户端发送速率 | 输入 R 个请求/秒时会发生什么？ |
| `ramp` | 请求速率按声明的函数连续变化 | 随着压力上升，时延在哪里开始崩溃？ |
| `burst` | 通过到达间隔分布控制请求聚集程度 | 服务能否吸收真实的流量尖峰？ |
| `pressure` | Worker 在持续时间内按并发策略循环复用负载 | 服务能否长时间维持饱和状态？ |
| `trace` | 按源时间戳调度，可设置缩放、偏移和窗口 | 生产流量时间特征和前缀局部性表现如何？ |

### 必须记录的调度遥测

每个请求应保存：

- 目标调度时间。
- 实际发送时间。
- 调度延迟。
- 发送时的在途请求数。
- 队列或积压深度。
- Worker/进程标识。
- 重试次数。

每次运行应报告目标 RPS、实际 RPS、调度延迟分位数、峰值在途请求数，以及条件允许时的客户端 CPU、内存和网络饱和信息。

### 建议接口

```bash
ais_bench perf \
  --url http://server/v1/chat/completions \
  --model served-model \
  --load-model open \
  --rate 100 \
  --requests 5000 \
  --max-in-flight unlimited
```

高级流量使用结构化 Profile：

```yaml
load:
  model: burst
  target_rps: 100
  inter_arrival:
    distribution: gamma
    shape: 0.5
  safety:
    max_in_flight: 2000
```

### 验收标准

- 闭环模式绝不超过配置的在途请求上限。
- 开环模式不会等待请求完成后才调度下一次到达。
- 使用可控假时钟验证每种模式的调度序列。
- 调度延迟超过阈值时，报告必须把结果标记为无效。
- 现有 AISBench 流量配置能够迁移到统一 Schema，且不丢失突发、爬升、稳态或 Trace 能力。

### 从 EvalScope 学什么

EvalScope 的 `--open-loop` 让行为差异非常容易理解。AISBench 应达到同等清晰度，同时保留更强的 Gamma/均匀突发性、连续爬升和时间戳 Trace 回放能力。

## 优先级 3：SLA 容量引擎

### 产品承诺

**给定一个 SLO，AISBench 返回满足目标的最大已验证负载，并提供边界附近的完整证据。**

EvalScope 的 SLA 自动调优能够搜索并发数或请求速率，支持时延和吞吐约束，重复执行每个测试点，并通过边界搜索把原始测量转换为容量结论。

### SLA 数据模型

```yaml
capacity_search:
  variable: request_rate
  range: [1, 1000]
  constraints:
    - p99_ttft_ms: {op: "<=", value: 500}
    - p99_e2e_ms: {op: "<=", value: 5000}
    - success_rate: {op: ">=", value: 0.999}
  point_policy:
    warmup_requests: 20
    measured_requests: 1000
    repetitions: 3
  decision:
    required_passes: 3
    confidence: 0.95
```

### 搜索流程

1. 验证低负载基线。
2. 扩大负载，直到发现失败边界。
3. 在边界区间内搜索。
4. 重复执行候选点。
5. 再次验证最终通过点和相邻失败点。
6. 返回最大已验证负载、置信信息和完整搜索轨迹。

二分搜索效率高，但它假设指标基本单调。动态批处理可能产生噪声或非单调结果，因此 AISBench 必须保存全部测试点，并支持网格加局部细化策略。

### 首版范围

首版支持：

- E2E、TTFT、TPOT 的 `avg`、p50、p90、p95 和 p99。
- 请求吞吐、输出 Token 吞吐和总 Token 吞吐。
- 成功率、超时率和调度有效性。
- 一个 SLO 内的 AND 约束。
- 搜索并发数或请求速率。

通用 OR 表达式和任意目标优化应延后，先稳定基本决策契约。

### 验收标准

- 当成功率 SLO 要求严格时，任何失败请求都能使该测试点失败。
- 结果包含所有测试负载及其重复执行记录，而不仅是最终答案。
- 检测到不稳定或非单调结果时，输出警告并切换到细化搜索。
- 最终结果明确区分输入负载与实际完成吞吐。
- 支持断点恢复，不重复执行已经完成且有效的测试点。

### 从 EvalScope 学什么

应复制用户最终获得的能力，而不必完全复制搜索实现。AISBench 要回答“该 SLO 下最大安全负载是多少”，同时加入更严格的有效性门禁，并允许选择稳态和高级流量 Profile。

## 优先级 4：标准运行与证据存储

### 产品承诺

**每个报告数字都能追溯到运行配置、标准事件、原始请求证据和指标版本。**

EvalScope 将请求、响应等测试数据写入 SQLite，以支持测试后查询。AISBench 当前输出配置、日志、请求级 CSV、汇总 JSON、详细 JSON/HDF5 和 HTML 图，并支持重计算。AISBench 应通过稳定 Schema 统一这些优势。

### 最小数据模型

| 实体 | 必须保存的内容 |
|---|---|
| `run` | ID、状态、时间、工具提交/版本、环境、度量版本 |
| `configuration` | 端点、后端、模型、Tokenizer、生成参数、负载模型、数据集哈希 |
| `request` | Request/Trace/Turn ID、调度/发送/完成时间、状态、Token 数 |
| `event` | 标准事件类型、时间戳、序号和载荷元数据 |
| `metric` | 名称、数值、单位、样本总体、时间窗口、计算版本 |
| `artifact` | 配置、原始响应、日志、图表和系统遥测 |
| `comparison` | 基线、候选运行、阈值、统计结果和结论 |
| `validity` | 检查项、状态、阈值、观测值和解释 |

首版使用 SQLite 足够。针对大规模分析，应支持导出 Parquet。原始响应存储必须支持脱敏和关闭。

### 用户能力

```bash
ais_bench runs list --model qwen --since 7d
ais_bench runs show RUN_ID
ais_bench compare BASELINE CANDIDATE --policy serving-regression.yaml
ais_bench export RUN_ID --format parquet
```

### 验收标准

- 报告必须从标准存储生成，不能继续维护平行的临时数据通路。
- 重计算生成新指标版本，但不修改原始证据。
- Run ID 稳定且支持恢复。
- 可在保留测量字段的同时对大请求载荷脱敏。
- Schema 迁移具备版本和自动测试。

### 从 EvalScope 学什么

EvalScope 证明了请求级 SQLite 分析和外部可视化的价值。AISBench 应首先让本地证据模型成为唯一权威数据源，再接入在线平台。

## 优先级 5：单命令产品入口

### 产品承诺

**新用户无需编辑 Python 配置即可正确测试一个端点，同时高级用户仍可使用完整任务系统。**

AISBench 的模型 × 数据集 × Summarizer 组合是运行大规模 Benchmark 的战略资产，应继续作为高级表达方式。问题在于，最简单的端点压测也被迫依赖这套配置流程。

### 必须提供的命令

```bash
ais_bench perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model qwen \
  --dataset random \
  --tokenizer Qwen/Qwen3 \
  --input-len 2048 \
  --output-len 512 \
  --parallel 32 \
  --requests 1000
```

CLI 必须编译成高级流程使用的同一套内部任务和配置对象，不能创建第二套执行引擎。

### 渐进式使用方式

| 用户层级 | 使用方式 |
|---|---|
| 首次运行 | URL、模型、数据集、长度、负载和请求数等参数 |
| 可复现实验 | 自动生成并提交到版本库的 YAML Profile |
| 高级 Benchmark | 现有模型/数据集/Summarizer 组合 |
| 扩展开发者 | 稳定的协议、数据集、调度器、指标和报告接口 |

### 验收标准

- 首次有效测试不需要修改源码树中的配置。
- `--dry-run` 输出完整解析后的配置。
- `--save-profile` 生成可复用的声明式 Profile。
- 无效参数组合必须在发送流量前失败。
- CLI 和高级配置必须生成相同的解析结果与性能结果。

### 从 EvalScope 学什么

EvalScope 的紧凑 CLI/Python API 降低了采用成本。AISBench 应提供同等的新手体验，但不能牺牲更强的 Benchmark 组合能力。

## EvalScope 功能详细解释

### 1. 统一的协议与执行入口

EvalScope 通过 `evalscope perf` 和 Python 参数接口提供性能测试，官方文档覆盖：

- OpenAI 兼容 Chat/Completions。
- OpenAI Responses。
- OpenAI 兼容 Embedding。
- OpenAI/Cohere 兼容 Rerank。
- 本地 Transformers 推理。
- 本地 vLLM 服务。
- 自定义 API 实现。

**价值：** 一个入口同时覆盖远程服务验收、本地速度测试、生成模型和检索模型。

**AISBench 启示：** 增加简单入口和稳定协议契约，但不要为每种端点建立独立压测引擎。

### 2. 闭环和开环调度

默认闭环模式中，`--parallel` 限制在途请求数。Worker 等待请求结束后才能继续发送。`--rate` 可以控制节奏，但仍存在背压。

启用 `--open-loop` 后，客户端按配置速率发送请求，不等待之前请求结束；`--parallel` 被忽略，并且可以扫描多个请求速率。

**价值：** 两种模式回答不同问题。闭环衡量受控并发用户体验；开环暴露输入速率超过服务能力后的排队增长。

**AISBench 启示：** 把调度行为提升为具名契约，并保存目标/实际发送时间。

### 3. 请求生命周期控制

EvalScope 提供：

- 按绝对数量或比例配置、不计入指标的预热请求。
- 总超时、连接超时和读取超时。
- 自定义 HTTP Header 和 API Key。
- 跳过连接检查。
- 请求数量和持续时间两种停止条件。
- 软退出：不再启动新工作，但允许在途请求完成。
- 不同扫描点之间的休眠时间。

多轮测试的软退出以 Trace 为单位：已经领取的对话可以完成剩余 Turn。

**价值：** 正确的生命周期能避免冷启动污染、截断样本和扫描点之间的意外过载。

**AISBench 启示：** 为所有负载模式统一定义生命周期状态和退出语义。

### 4. SLA 自动调优

EvalScope 可以调节并发数或请求速率，支持平均和 p99 的 Latency、TTFT、TPOT，以及请求吞吐、Token 吞吐约束，也支持最大化 TPS 等极值目标。

官方流程先探测边界，再进行二分搜索；默认对每个点重复运行以降低噪声。

**价值：** 将一组性能测量转化为直接可执行的容量答案。

**AISBench 启示：** 在度量正确性之后，这是价值最高的缺失能力。

### 5. 数据集和负载系统

EvalScope 性能数据集覆盖：

| 负载类型 | 示例 | 目标 |
|---|---|---|
| 短/长真实文本 | OpenQA、LongAlpaca | 真实 Prefill/Decode 分布 |
| 可控合成文本 | `random` | 固定或随机 Token 长度实验 |
| 本地自定义文本 | 逐行文件、自定义解析器 | 私有工作负载回放 |
| 视觉语言 | Flickr8k、随机图文等 | 图像与文本服务压测 |
| Embedding | 文件、随机、Batch 变体 | 向量端点吞吐和批处理 |
| Rerank | Query-Document 文件和随机数据 | 检索阶段服务测试 |
| 多轮对话 | 合成、ShareGPT、自定义消息 | 上下文增长和对话负载 |
| Agent Trace | SWE-smith、Trie 派生 Trace | 长上下文、工具调用和生产型对话 |

数据集专属参数会进行 Schema 校验。真实文本能够截断到指定 Token 长度，数据可以来自 ModelScope、Hugging Face 或本地文件。

**价值：** 用户可以在不更换负载引擎的情况下，从可控微基准切换到真实应用负载。

**AISBench 启示：** AISBench 已有强大的数据集和任务系统。应先稳定共享契约，再增加 Embedding/Rerank 适配器。

### 6. 多轮和 Agent 性能

EvalScope 会把模型的真实回复追加到历史对话，后续 Turn 携带完整上下文。工具估算历史 Token 中理论上可以被 Prefix Cache 利用的比例，但实际命中仍取决于服务端实现。

它支持合成对话、ShareGPT、本地 OpenAI Messages、SWE-smith 编码轨迹和面向生产的 Agent Trace。长上下文轨迹既可以预构建以保证复现，也可以在线生成。

**价值：** 单轮压测无法揭示持续增长的 Prefill 成本、前缀复用、完整 Trace 时延和缓存行为。

**AISBench 启示：** AISBench 已有 ShareGPT/MTBench 和 Mooncake 前缀局部性，应标准化 Request/Turn/Trace ID，并输出 Trace 级指标和缓存证据。

### 7. 指标和报告

EvalScope 输出：

- 测试时长、配置并发/速率、请求总数、成功和失败。
- 请求吞吐、输出 Token 吞吐和总 Token 吞吐。
- E2E、TTFT、TPOT 和 ITL。
- 输入/输出 Token 和解码速率。
- 分位数分布。
- 多轮 Trace 和缓存相关指标。
- 条件允许时的投机解码指标。

**价值：** 报告能够区分响应性、Decode 行为、吞吐和负载形态。

**AISBench 启示：** AISBench 已有相当的核心指标和额外的稳态/Prefill 视角。优先级应是版本化公式、事件契约和可用性状态，而不是继续增加名称。

### 8. 持久化和自定义分析

EvalScope 将请求和响应数据写入 SQLite。用户可以在测试结束后查询具体请求，例如首 Chunk 时延大于阈值的成功请求。

它还支持将结果可视化到 WandB、SwanLab 和 ClearML。

**价值：** 性能测试不再只是终端输出，而是完整证据工作流。

**AISBench 启示：** 先通过标准运行存储统一现有 CSV/JSON/HDF5/配置文件，再接入外部平台。

### 9. 扩展点

EvalScope 明确提供以下自定义方式：

- API 请求与响应处理。
- 数据集解析。
- 结果分析。

**价值：** 私有服务和私有负载能够复用相同的调度和指标系统。

**AISBench 启示：** 扩展接口必须依赖标准 Request、Event 和 Result 类型，不能绕过统一度量平面。

## AISBench 必须保留的护城河

| AISBench 现有优势 | 为什么必须保留 |
|---|---|
| 突发性分布 | 比单一泊松模式提供更丰富的到达建模 |
| 线性/指数爬升 | 连续观察过载转折点 |
| 预期与实际 RPS 图 | 不只验证服务，也验证负载发生器 |
| 稳态 Summarizer | 将稳定阶段与爬升、退出阶段分离 |
| Mooncake 时间戳/Hash Trace | 建模生产时间特征和 Prefix Cache 局部性 |
| 指标重计算 | 无需重新执行昂贵推理即可生成新视角 |
| OpenCompass 任务组合 | 将性能与公认 Benchmark 工作负载连接起来 |
| 多任务看板和编排 | 支持完整评测活动，而不只是单端点测试 |

路线图应把这些功能迁移到共享的 First-Class 契约之下，而不是隐藏或替换它们。

## 暂时不要建设为 First-Class 能力的功能

| 功能 | 决策 | 原因 |
|---|---|---|
| WandB/SwanLab/ClearML 适配器 | 后续集成 | 应消费标准结果存储 |
| Embedding/Rerank | 下一层覆盖能力 | 有价值，但不能弥补度量和负载契约缺失 |
| 分布式压测 | 调度遥测之后 | 分布式会放大时间和有效性问题 |
| 任意 SLA 表达式语言 | 延后 | 简单 AND 约束已覆盖首批运维场景 |
| 更多分位数别名 | 不建议 | 指标定义比名称数量重要 |
| 第二套“简化版”引擎 | 拒绝 | 简单入口必须编译到相同核心执行路径 |

## 目标架构

```text
CLI / YAML / 现有任务配置
              |
       解析后的 RunSpec
              |
    +---------+---------+
    |                   |
负载适配器           协议适配器
    |                   |
    +-----> 负载引擎 <---+
              |
          标准事件
              |
        度量 + 有效性
              |
        运行/证据存储
              |
 报告 / 对比 / SLA 搜索
```

**标准事件流和证据存储是整个架构的“窄腰”。** 上层能力可以独立演进，下层结果始终保持可审计。

## 交付计划

### 阶段 0：正确性基础

预计 4-6 周：

- `MeasurementSpec v1`。
- 标准事件类型。
- 固定流式响应测试样例。
- 显式 unavailable/error 状态。
- Run ID 和完整配置快照。

**退出条件：** 两个不同协议适配器对同一组合成事件产生完全一致的指标。

### 阶段 1：负载与证据平面

预计 6-8 周：

- 统一 `closed`、`open`、`ramp`、`burst`、`pressure` 和 `trace` Schema。
- 调度延迟和实际负载遥测。
- SQLite 运行/证据 Schema。
- 从证据存储生成报告。
- 现有 AISBench 性能配置迁移适配器。

**退出条件：** 开环输入负载不受请求完成影响，并且报告能够检测客户端调度器饱和。

### 阶段 2：容量产品

预计 4-6 周：

- SLA 约束。
- 并发/速率搜索。
- 多次重复和边界确认。
- 断点恢复。
- 搜索轨迹可视化。

**退出条件：** 对具有已知容量转折点的 Mock 服务，能够找到预期的最大安全负载。

### 阶段 3：产品入口与回归工作流

预计 4-6 周：

- `ais_bench perf` 直接参数。
- `--dry-run` 和 `--save-profile`。
- 运行对比和性能回归策略。
- 面向汇报的 HTML 报告。

**退出条件：** 新用户无需编辑仓库 Python 文件，即可运行、保存、复现和比较端点性能测试。

### 阶段 4：覆盖面扩展

随后增加：

- Embedding 和批量 Embedding。
- Rerank。
- 更丰富的多模态 Profile。
- 分布式负载发生器。
- 外部可视化平台。

## 完成定义

完成 First-Class 能力路线图后，用户应能够：

1. 用一个命令测试端点。
2. 选择明确且有文档的负载模型。
3. 证明客户端确实生成了目标负载。
4. 信任版本化的指标定义。
5. 查询某个 SLO 下的最大安全负载。
6. 查看所有测试点和原始请求证据。
7. 不重新运行推理即可重计算报告。
8. 将候选运行与历史基线比较。
9. 对合成、Benchmark、多轮和 Trace 负载复用同一套系统。
10. 扩展协议和数据集而不绕过核心度量系统。

## 最终建议

**先建设度量平面，再扩大功能表面。** 第一阶段产品应将 AISBench 已有的负载真实性与五项 First-Class 契约结合起来：指标、负载模型、SLA 容量、证据存储和简单入口。这能够补齐 EvalScope 最强的产品优势，同时保留 AISBench 最难被复制的部分。

## 深入阅读

- **竞品细节：** [AISBench Benchmark vs. EvalScope Perf](../aisbench-vs-evalscope-perf.md)
- **EvalScope 功能细节：** [EvalScope Perf 深入分析](../evalscope-perf.md)
- **AISBench 性能指南：** [服务化性能测评](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/performance_benchmark.html)
- **AISBench 流量模型：** [RPS 分布控制](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/advanced_tutorials/rps_distribution.html)
- **EvalScope 性能指南：** [模型推理性能压测](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/)
- **EvalScope 参数：** [性能压测参数](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/parameters.html)
- **EvalScope 多轮：** [多轮对话压测](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/multi_turn.html)
- **EvalScope SLA：** [SLA 自动调优](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/sla_auto_tune.html)
