---
description: "Use when creating or updating term glossary pages under docs/terms/. Enforces definition-first structure, cross-linking to paper insight pages, and agent-driven auto-ingestion conventions."
applyTo: "docs/terms/**/*.md"
---

# Docs Terms Rules

Use this instruction for every page under `docs/terms/`. Term pages are glossary entries — they define a technical concept, not analyze a paper.

## Front Matter

Term pages use the same required Jekyll front matter as other docs pages, with term-specific fields for glossary grouping and hover previews:

```yaml
---
title: "Term Name"
summary: "One-sentence compact definition of the term."
tooltip: "Two or three plain-language sentences for hover previews. Define the term, explain why it matters, and optionally name the most common confusion."
layout: default
confidence: high
category: training
sources:
  - raw/path/to/key-paper.pdf
aliases:
  - alternative-spelling
  - another-name
appears_in:
  - docs/training/megatron-lm/index.md
  - docs/algorithms/transformer.md
updated: 2026-07-27
---
```

### Field Rules

- `title`: required. The canonical term name. Use title case.
- `summary`: required. A compact one-sentence definition for indexes, search snippets, and dense link lists.
- `tooltip`: strongly recommended. Two or three plain-language sentences for term hover previews on related paper pages. Define the term, explain why it matters, and optionally mention the most common confusion. If omitted, hover previews should fall back to `summary`.
- `category`: required. One of `training`, `algorithms`, `hardware`, `frameworks`, `benchmarks`, or `general`. This groups terms in the index.
- `aliases`: optional. Alternative names or spellings for the term. Helps with search and auto-linking.
- `appears_in`: optional but strongly recommended. This is the canonical list of docs pages that use this term, as repository-relative paths. Every listed page must contain an ordinary Markdown link to this term page, and every docs page that links the term must be listed here.
- All other front matter fields follow [`docs-front-matter.instructions.md`](docs-front-matter.instructions.md).

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

Bullet list of papers and docs pages where this term is used. Each entry links to the relevant page. This section is the primary cross-linking surface.

```markdown
## Where It Appears

- [Megatron-LM: GPU-Cluster Training Parallelism](../training/megatron-lm/index.md) — {one-line role in that paper}
- [GPipe (Huang et al., 2019)](https://arxiv.org/abs/1811.06965) — First introduced the concept
```

### 7. Related Terms

Bullet list of sibling term pages. Keep it short — only directly related concepts.

## Agent Auto-Ingestion Rules

### During Paper Ingest

When an agent creates or updates a paper insight page, it MUST also:

1. **Identify key terms** using the signal categories below. A term is "key" if it is non-trivial (not a generic word like "model" or "layer") and satisfies at least one of these signals:

   | Signal                                           | Example candidates                                                  |
   | ------------------------------------------------ | ------------------------------------------------------------------- |
   | Named communication primitive                    | all-reduce, all-gather, reduce-scatter, broadcast, P2P send/recv    |
   | Named parallelism strategy or scheduling pattern | sequence parallelism, pipeline bubble, 1F1B, interleaved schedule   |
   | Named algorithm or kernel with a canonical paper | Ring Self-Attention, FlashAttention, PagedAttention, RadixAttention |
   | Cross-paper system pattern with a distinct name  | split/all-gather, activation recomputation, KV cache, microbatch    |
   | Named hardware format or precision               | FP8, NVFP4, block quantization                                      |

   A term is "cross-paper" if it appears in the current paper AND at least one other paper already in the knowledge base, OR if the term has a well-known name that would appear in a textbook or framework documentation. **Do not skip terms just because they are "infrastructure" or "systems" concepts** — communication primitives, scheduling patterns, and precision formats are first-class term candidates.

2. **For each term:** check if `docs/terms/{term-slug}.md` exists.
   - If missing: create it following this instruction file.
   - If present: add the new paper to the `appears_in` front matter list and update the "Where It Appears" section.
3. **Add cross-links** from the paper page body to the term pages by linking the first meaningful occurrence of each term in the prose. Do not put related-term links in front matter fields such as `summary` or `description`.
4. **Update `docs/terms/index.md`** when adding a new term.

### When Creating a Term Outside of Paper Ingest

When an agent creates a new term page for any reason other than ingesting a new paper (e.g., the user asks for a term, or a gap is discovered), the agent MUST also perform these retroactive steps:

1. **Search existing docs pages** for occurrences of the term (and its aliases). Use `grep_search` with the term name as a plain-text query, scoped to `docs/**/*.md` excluding `docs/terms/` and `docs/logs/`.
2. **For each page that mentions the term:** add it to the term's `appears_in` front matter and its "Where It Appears" section.
3. **For each page that mentions the term:** add a markdown link from the first meaningful in-content occurrence to the new term page. Do not link from front matter, headings, image captions, or code blocks.
4. **Update `docs/terms/index.md`** with the new term entry.
5. **Update `docs/logs/log.md`** with a brief entry recording the term creation and which pages were back-linked.

## Terms Index Format

`docs/terms/index.md` groups terms by category using the `category` front matter value.

For small categories, use a flat bullet list:

```markdown
## Training

- [Microbatch](microbatch.md) — A small chunk of a training batch used to enable pipeline parallelism.
```

When a category grows beyond roughly 20 terms, add alphabetic subheadings inside that category to keep scanning fast:

```markdown
## Training

### M

- [Microbatch](microbatch.md) — A small chunk of a training batch used to enable pipeline parallelism.

### P

- [Pipeline Bubble](pipeline-bubble.md) — Idle pipeline time during warmup and drain phases.
```

Keep index summaries to one sentence and reuse the term page `summary` when it is already clear.

## Hover Preview Format

Related paper pages may render term links with hover previews, similar to a compact wiki tooltip. The visible links belong on the first meaningful term occurrences in the Markdown body, not in page description/front matter and not in a detached glossary line unless the page genuinely needs a manual term list. The preview text should come from the term page `tooltip` field, falling back to `summary`, then to a short extract from the rendered term page body when those fields are missing.

Use this style when linking terms from paper pages:

```markdown
The interleaved schedule keeps [microbatches](../../terms/microbatch.md) in flight.
```

Use ordinary Markdown link text when the surrounding sentence needs pluralization, lowercase text, or an acronym. The rendered site enhances links to `docs/terms/{slug}.md` with hover previews by reading `docs/terms.json`, which is generated from term page front matter. Do not duplicate tooltip text inside paper pages; keep reusable descriptions on the term page.

## Term Link Validation

Run the glossary consistency check after adding or changing terms or their
consumer pages:

```bash
./scripts/run-in-workspace.sh python scripts/checks/term_links.py
```

The checker treats each term page as the single source of truth; do not maintain
a separate hand-written term registry. It validates the glossary index,
`appears_in`, the local links under "Where It Appears", and links from consuming
docs pages in both directions. Plain-text occurrences of a canonical title or
explicit alias that have no term link are warnings by default. Use
`--strict-mentions` to make those findings fail while cleaning or reviewing a
focused scope. Front matter, headings, code, image captions, term pages, and log
pages are excluded from mention discovery. Glossary-to-glossary navigation is
also excluded from `appears_in` validation; keep those links under "Related
Terms" instead.

## Slug Convention

- Use kebab-case: `pipeline-parallelism.md`, `scatter-gather.md`, `kv-cache.md`.
- Prefer the shortest unambiguous name.
- If a term has a widely-used acronym or shorthand, use that as the canonical slug (e.g., `gqa.md` for Grouped-Query Attention) and list the full name in `aliases`.
