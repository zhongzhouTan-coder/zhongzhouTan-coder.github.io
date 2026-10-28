# Paper Insight: Results and Closing

## Required Body Structure

### 10. What This Buys You

Results as a narrative, not a data dump. Structure:

```markdown
## What This Buys You

### The headline claim
[One sentence stating the main empirical takeaway.]

### How we know: [evidence category]
[One compact table. Only the headline result plus one nuance.]

### The mechanism behind the numbers
[Explain WHY the numbers look the way they do. Teach the reader how to interpret them.]

### ⚠️ How to read these numbers
[If any result is easily misinterpreted, call it out explicitly.]
```

Rules:

- Show only the numbers that answer a specific question. Link to the paper for full tables.
- Explicitly teach the reader how to interpret results, especially counterintuitive ones.
- Use caution markers (⚠️) to flag results that are commonly misinterpreted.
- Prefer an original source chart when it directly answers the question. When
  synthesizing a chart from reported values, cite the exact source table or
  values, preserve axes, units, workload conditions, and uncertainty, and label
  repository-derived calculations separately from source-reported results.

### 11. Where It Breaks

Failure modes as a table with conditions. Not a prose list.

```markdown
## Where It Breaks

| Failure mode | When it happens | Impact |
|---|---|---|
| [Mode 1] | [Condition] | [Consequence] |
| [Mode 2] | [Condition] | [Consequence] |
```

Rules:

- Frame each failure mode around the reader's perspective: "when should I NOT trust this?"
- Each row must include a concrete condition, not a vague caveat.
- Include failure modes the paper acknowledges AND failure modes you infer.

### 12. One Thing to Remember

A single paragraph that encodes the entire paper into a memorable frame. Not a bullet list. Bold the key retrieval phrase.

```markdown
## One Thing to Remember

[One paragraph. Bold the phrase a reader should recall six months later.]
```

The brain encodes a single paragraph as one chunk, making it more memorable than a bullet list. This section is the "if you forget everything else" takeaway.

### 13. Go Deeper

Curated links grouped by reader intent:

```markdown
## Go Deeper

- **Read:** [Paper link]
- **Build on:** [Baselines, competitors, follow-up work]
- **Understand the context:** [Related internal pages]
- **Reproduce:** [Code link or "not available at time of writing"]
```
