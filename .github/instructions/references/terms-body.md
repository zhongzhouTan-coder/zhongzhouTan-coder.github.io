# Glossary Terms: Body

## Body Structure

Term pages are definition-first — short, self-contained, and optimized for quick lookup. The body order is:

### 1. Definition

A single bold sentence that defines the term unambiguously. This is the answer a reader gets from a 5-second scan.
Start the sentence with the canonical title exactly as written in front matter, not an article or alternate spelling, so snippets and future auto-link previews stay consistent.

```markdown
**{Term}** is {one clear definition sentence}.
```

### 2. Why It Exists

One short paragraph. The specific problem or constraint this concept addresses. Answer: "What would break without it?"

### 3. How It Works

One to three short paragraphs with at most one equation. Show the mechanism, not the derivation. Prefer concrete numbers over abstract variables when possible.

### 4. Tradeoffs (Optional)

Use this section when the term has meaningful costs, limits, failure modes, or cases where it should not be used. Keep it short and practical.

### 5. Common Confusions (Optional)

Use this section when the term is often confused with nearby concepts. Prefer direct contrasts, especially when sibling term pages exist.

```markdown
## Common Confusions

- **Microbatch vs. minibatch:** {one-line distinction}
```

### 6. Where It Appears

Curated bullet list of the best papers and docs pages for learning where this
term matters. Each entry links to the relevant page. Do not mirror every
consumer link here; complete backlinks are generated automatically.

```markdown
## Where It Appears

- [Megatron-LM: GPU-Cluster Training Parallelism](../training/megatron-lm/index.md) — {one-line role in that paper}
- [GPipe (Huang et al., 2019)](https://arxiv.org/abs/1811.06965) — First introduced the concept
```

### 7. Related Terms

Bullet list of sibling term pages. Keep it short — only directly related concepts.
