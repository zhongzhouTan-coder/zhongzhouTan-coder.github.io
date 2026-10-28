# Paper Insight: Mechanism and Runtime Trace

## Required Body Structure

### 7. Symbol Map or Notation Guide

Required when a page uses dense math, architecture notation, tensor names, protocol fields, abbreviations, or paper-specific superscripts/subscripts. Omit only when the page has little or no notation.

Place it before Deep Dive so readers can decode the mechanism sections. Use the following pattern:

```markdown
## Symbol Map

[One short paragraph explaining naming conventions, e.g. `C = content`, `R = RoPE`, `D = down-projection`.]

| Symbol | Human name | Shape / scope | Plain meaning |
|---|---|---|---|
| `$k_t^C$` | content key | per-token/per-head | Semantic key derived from the latent cache. |
| `$c_t^{KV}$` | K/V latent | per token | Compact state cached instead of full K and V. |
```

Rules:

- Explain superscripts, subscripts, and overloaded letters before listing individual symbols.
- Prefer human names like "content key" or "routing score" over restating the symbol.
- Include shape, dimension, or scope when it helps distinguish similar symbols.
- Add a second task-specific table when it improves comprehension, such as "cached vs computed", "training vs inference", or "sender vs receiver".
- Do not dump every symbol from the paper. Include symbols that appear in the page or are necessary to understand the core mechanism.
- Keep equations in the Deep Dive; the Symbol Map is for decoding notation, not deriving the method.

### 8. Deep Dive

Each mechanism gets its own subsection following this mini-template:

```markdown
### [Mechanism Name]

**What it does:** [One sentence.]

**Why it matters:** [One sentence connecting back to the problem from "Why This Exists."]

**How it works:** [Step-by-step. Prefer tables over walls of text. Use equations only when necessary.]

**The intuition:** [One sentence in plain language. Non-negotiable for every subsection.]

**A concrete example:** [Reuse the same scenario from "Why This Exists." Show how this mechanism fixes it.]

**Remember:** [One bullet — the single most important fact about this mechanism.]
```

Rules for Deep Dive subsections:

- Each subsection must teach exactly one new concept, building on the previous one.
- If a subsection has more than one "aha" moment, split it.
- The "intuition" sentence is mandatory. If you cannot state it in plain language, you do not understand the mechanism well enough to write the section.
- Reuse the same concrete example across subsections for spaced repetition.

### 9. Putting It Together

A numbered end-to-end trace showing all mechanisms interacting in one scenario. Answers: "how do the pieces actually interact at runtime?"

```markdown
## Putting It Together

[A numbered walkthrough: ① draft → ② schedule → ③ verify → result.]
```

Use one concrete object throughout the trace: a request, token sequence, tensor,
batch, cache block, or protocol message. At every step, name the acting
component, the input state, the action, and the resulting state. A compact table
is preferable when the state transitions would be hard to compare in prose:

```markdown
| Step | Actor | Input state | Action | Output state |
|---:|---|---|---|---|
| 1 | [component] | [before] | [operation] | [after] |
```

Do not add a runtime diagram if the numbered trace already answers the same
question. Add one only when concurrency, branching, ownership, or handoff
boundaries remain difficult to understand.
