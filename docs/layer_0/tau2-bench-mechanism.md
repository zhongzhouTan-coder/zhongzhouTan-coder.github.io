---
title: "τ²-Bench: Mechanism and Design"
summary: "How τ²-Bench evaluates conversational AI agents in a dual-control environment, covering its Dec-POMDP formalism, domain construction, task generation, and evaluation methodology."
layout: default
doc_layer: layer_0
confidence: high
sources:
  - raw/benchmark/tau2-bench.pdf
updated: 2026-05-29
---

# τ²-Bench: Mechanism and Design

τ²-Bench (Barres et al., 2025) is a benchmark for evaluating conversational AI agents in a **dual-control environment** — one where both the agent and the simulated user can take tool-based actions to modify a shared world state. It extends the earlier τ-bench by giving the user meaningful agency rather than limiting them to a passive information provider.

## Core Problem It Addresses

Existing benchmarks (τ-bench, ToolSandbox, FlowBench) are **single-control**: only the AI agent can call tools; the user only speaks. Real-world customer support is different — a user must actively perform steps (toggle airplane mode, reseat a SIM card, check a status bar) guided by the agent. τ²-Bench models this asymmetric collaboration.

## Formal Model: Dec-POMDP

Dual-control interactions are formalised as a **Decentralised Partially Observable Markov Decision Process** (Dec-POMDP), defined by a tuple $(S, \{A_i\}, \{O_i\}, T, R, \mathcal{U}, M)$:

| Component | Meaning |
|---|---|
| $S$ | Global state = world databases ⊗ interaction history |
| $A_i$ | Action space for player $i$ — either a tool call or a natural-language message |
| $O_i$ | Observations for player $i$ — tool returns or the other player's message |
| $T$ | Transition function: $(S, A) \to (S', O)$ |
| $R$ | Reward $R: S \to [0,1]$ — binary task success verified against assertion functions |
| $\mathcal{U}$ | Instruction space — scenario instructions for the user; domain policy for the agent |
| $M$ | Message space — all possible natural-language exchanges |

Only one player acts per turn. The global state updates after every tool call; messages update only the history component.

### Turn-by-Turn Interaction Loop

```mermaid
sequenceDiagram
    participant A as AI Agent
    participant E as Shared World State
    participant U as User Simulator

    Note over A,U: One player acts per turn
    U->>A: "Hi, I have no signal."
    A->>E: lookup_account(customer_id)
    E-->>A: {account: active, line: enabled}
    A->>U: "Can you check if airplane mode is on?"
    U->>E: get_airplane_mode_status()
    E-->>U: {airplane_mode: true}
    U->>A: "Yes, it shows airplane mode is on."
    A->>U: "Please toggle airplane mode off."
    U->>E: toggle_airplane_mode()
    E-->>U: {airplane_mode: false, signal: connected}
    U->>A: "Done, I have signal now!"
    Note over E: Assertion functions verify final state
```

## Environment Architecture

```mermaid
graph TB
    subgraph Agent Side
        AP[Domain Policy] --> AG[AI Agent LLM]
        AG --> AT["Agent Tools\n(6 write · 7 read)\nCRM, billing, line mgmt"]
    end

    subgraph User Side
        SI[Scenario Instruction] --> US[User Simulator LLM]
        US --> UT["User Tools\n(15 write · 15 read)\nPhone: airplane mode, SIM,\ndata toggle, speed test…"]
    end

    subgraph Shared World State
        ADB[(Agent DB\nCustomers · Lines\nPlans · Devices · Bills)]
        UDB[(User DB\nPhone status · Signal\nSIM · Data · APN)]
        HIS[Interaction History]
    end

    AT <--> ADB
    UT <--> UDB
    AG <-->|message| US
    ADB -.->|partial obs| AG
    UDB -.->|partial obs| US
```

The agent sees CRM-style data; the user sees phone-device-style data. Neither has full visibility of the other's database, creating genuine partial observability.

## Domain: Telecom

The primary new domain models a telecom technical-support interaction. Key statistics:

| | retail | airline | telecom |
|---|---|---|---|
| Agent tools | 7 write, 6 read | 6 write, 6 read | 6 write, 7 read |
| User tools | — | — | 15 write, 15 read |
| Tasks (sampled) | 115 | 50 | 114 (full: 2285) |

**Three user intents** form a difficulty hierarchy:

1. `service_issue` — no signal, SIM or airplane mode problems (mean 2.3 solution actions)
2. `mobile_data_issue` — data unavailable or slow (mean 4.3 actions)
3. `mms_issue` — picture/video messaging broken (mean 6.0 actions; requires service + data prerequisites)

## Five-Stage Domain Construction

```mermaid
flowchart LR
    S1["1. Agent DB & Tools\nLLM generates PRD\n→ schema + tool sigs\n→ unit tests + refine"]
    S2["2. User DB & Tools\nMocked phone device\n→ readable state\n→ write tools + refine"]
    S3["3. Task Generation\nAtomic subtasks:\ninit · sol · assert\n→ composite tasks"]
    S4["4. Agent Policy\nTroubleshooting doc\n+ flowcharts\n→ system prompt"]
    S5["5. Manual Refinement\nJoint review of tools,\npolicy & subtasks"]

    S1 --> S2 --> S3 --> S4 --> S5
```

**Stage 1 — Agent database and tools.** An LLM generates a Product Requirements Document (PRD) specifying the CRM schema (customers, lines, plans, devices, bills) and agent tool signatures. Implementations and unit tests are generated then manually refined until all tests pass.

**Stage 2 — User database and tools.** Similarly, a mocked phone device is defined with readable state (signal strength, SIM status, airplane mode, data toggle…) and write tools (toggle, reseat, reset APN…). Same generate-test-refine loop.

**Stage 3 — Programmatic task creation.** Each atomic subtask $t$ is a triple $(\{f^{\text{init}}\}, \{f^{\text{sol}}\}, \{f^{\text{assert}}\})$:

- **Initialization functions** — set the broken initial state (e.g., `set_airplane_mode(True)`)
- **Solution functions** — the ordered sequence of tool calls that fix it (e.g., `toggle_airplane_mode()`)
- **Assertion functions** — conditions on the final world state that confirm success (e.g., `assert_service_status("connected")`)

Atomic subtasks are grouped into mutually-exclusive groups. A **composite task** picks at most one subtask per group and concatenates their function lists. Correctness is verified automatically: running init then solution must satisfy all assertions; running init alone must not. The telecom domain has 15 atomic subtask groups yielding 2285 valid composite tasks; 114 are sampled for a balanced distribution.

```mermaid
flowchart TD
    subgraph Group A ["Subtask Group A (e.g. airplane mode)"]
        A1[airplane_on]
        A2[airplane_off]
    end
    subgraph Group B ["Subtask Group B (e.g. SIM)"]
        B1[sim_removed]
        B2[sim_locked]
    end
    subgraph Group C ["Subtask Group C (e.g. data toggle)"]
        C1[data_disabled]
    end

    A1 -->|pick one| CT
    B2 -->|pick one| CT
    C1 -->|pick one| CT
    CT(["Composite Task\n= concat inits · sols · asserts"])
    CT --> V{{"Auto-verify:\ninit+sol ⊨ asserts?\ninit alone ⊭ asserts?"}}
    V -->|pass| OK[Valid task]
    V -->|fail| DRP[Discard]
```

**Stage 4 — Domain-specific agent policy.** LLMs generate a detailed troubleshooting policy document (with optional workflow flowcharts) that the agent receives in its system prompt.

**Stage 5 — Manual refinement.** Tools, policy, and subtasks are jointly reviewed and fixed.

## User Simulator Design

The user simulator is an LLM agent given:

- A **scenario instruction** (reason for call, known/unknown info, task-specific behaviour rules)
- The **user tool set** (identical to what a real user would operate)

Key constraints that improve reliability over τ-bench's retail/airline simulators:

- User tools return **human-readable outputs only** — no raw JSON that could be misinterpreted as agent-side data.
- The simulator is instructed to call a tool only when the agent explicitly requests it, and to ask for clarification rather than guess.
- If the agent asks for multiple actions at once, the simulator states it can only do one at a time.
- Environment state constrains what tool calls can return, making fabrication harder.

Result: telecom user simulator error rate 16% (6% critical) vs. retail 40% (12%) and airline 47% (13%).

## Task Evaluation Criteria

Tasks can specify any subset of five verification methods:

| Method | How it works |
|---|---|
| **DB check** | Compare final agent-side database values to expected values |
| **Status assertion** | Run predefined assertion functions on final world state $S_{\text{world}}$ |
| **Natural language assertion** | LLM-judge check on the conversation history |
| **Communication info check** | Verify specific facts were communicated by the agent |
| **Action matching** | Confirm every required solution function appears in the trajectory |

The telecom domain uses **status assertions only**.

## Evaluation Metrics

The paper uses the **pass^k** metric from τ-bench: the fraction of $k$ independent runs on which all tasks succeed. This captures consistency, not just peak performance.

## Ablation: Separating Reasoning from Communication

Three evaluation modes isolate different failure sources:

| Mode | Setup | Purpose |
|---|---|---|
| **Default** | Agent + user simulator; dual-control | Full benchmark |
| **No-User** | Agent controls all tools; gets a ticket describing the problem | Isolates pure reasoning capability |
| **Oracle Plan** | Agent given the full ground-truth tool sequence; must guide user to execute it | Isolates communication / coordination capability |

```mermaid
flowchart LR
    subgraph Default ["Default (full benchmark)"]
        DA[Agent] <-->|messages| DU[User Simulator]
        DA --> DAT[Agent Tools]
        DU --> DUT[User Tools]
        DAT & DUT --> DSW[(Shared World)]
    end

    subgraph NoUser ["No-User (reasoning only)"]
        NA[Agent] --> NAT["All Tools\n(agent + user)"]
        NAT --> NSW[(World)]
        TKT[Ticket description] --> NA
    end

    subgraph Oracle ["Oracle Plan (communication only)"]
        OA[Agent] <-->|messages| OU[User Simulator]
        OU --> OUT[User Tools]
        OUT --> OSW[(World)]
        GT[Ground-truth plan] --> OA
    end
```

Key finding: shifting from No-User to Default causes ~18–25% pass^1 drop, confirming that guiding an active user is the dominant bottleneck beyond reasoning.

## Key Empirical Results

Pass^1 on the telecom domain (Default mode):

| Model | pass^1 |
|---|---|
| claude-3.7-sonnet | 49% |
| o4-mini | 42% |
| gpt-4.1 | 34% |
| gpt-4.1-mini | ~44% |

- Performance degrades sharply past 7 required actions in Default mode, approaching 0%.
- No-User mode outperforms Default for all task lengths, but the gap narrows for very long tasks.
- Harder issue types (`mms_issue`) show near-zero pass^4 for most models, indicating low reliability.
- Agent performs best with the **Easy** user persona, worst with **None** (no persona), which is often on par with or below **Hard**.

## Persona System

Each telecom task is randomly assigned one of three user personas:

- **None** — no persona given to the simulator.
- **Easy** — office administrator; average technical skill; patient and communicative.
- **Hard** — 64-year-old retired librarian; limited technical knowledge; gets flustered easily; needs constant reassurance.

## Limitations

- Does not model the **expert-novice gap**: the agent never needs to adapt its explanations to the user's mental model.
- Domain expansion still requires significant human expertise.
- The tool-augmented user simulator approach has not yet been applied to the existing retail and airline domains.

## Related Pages

- [Knowledge Base Introduction](../README.md)
