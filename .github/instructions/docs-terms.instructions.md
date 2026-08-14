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
mention_aliases:
  - alternative-spelling
mention_lint: canonical
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
- `mention_aliases`: optional. The subset of `aliases` precise enough for the
  term-link checker to detect when `mention_lint: aliases` is selected. Every
  entry must also appear in `aliases`. Do not include ambiguous abbreviations
  such as `CP`, `TP`, or `PP` unless their repository-wide use is reliably
  unambiguous.
- `mention_lint`: optional; defaults to `canonical`. Use `off` for common or
  simple terms whose plain-text occurrences do not normally need glossary
  links, `canonical` to review only the title, or `aliases` to review the title
  plus `mention_aliases`.
- `appears_in`: optional but strongly recommended. This is a curated list of
  the best explanatory or source-defining docs pages, not an exhaustive
  backlink registry. Keep roughly 5-10 high-value pages when the term is widely
  used. Every listed page must contain an ordinary Markdown link to this term
  page and must appear under "Where It Appears." Other consumer links are valid
  without being added here; exhaustive backlinks are generated into
  `docs/terms.json` at site-build time.
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
   - If present: add the paper to `appears_in` and "Where It Appears" only when
     it is one of the strongest examples, definitions, or applications of the
     term. Ordinary consumers need only the in-content link.
3. **Add cross-links** from the paper page body to the term pages by linking the first meaningful occurrence of each term in the prose. Do not put related-term links in front matter fields such as `summary` or `description`.
4. **Update `docs/terms/index.md`** when adding a new term.

### When Creating a Term Outside of Paper Ingest

When an agent creates a new term page for any reason other than ingesting a new paper (e.g., the user asks for a term, or a gap is discovered), the agent MUST also perform these retroactive steps:

1. **Search relevant existing docs pages** for meaningful occurrences of the
   term and its precise aliases. Use the scoped mention-review command below;
   do not force a repository-wide cleanup for a common term.
2. **Link only pages where the glossary link materially helps comprehension.**
   Do not add links in front matter, headings, captions, code, navigation-only
   lists, or every repetitive use.
3. **Curate `appears_in` and "Where It Appears"** with the best 5-10 learning
   pages. Generated backlinks preserve complete consumer coverage.
4. **Update `docs/terms/index.md`** with the new term entry.
5. **Update `docs/logs/log.md`** with a brief entry recording the term creation
   and representative pages that were linked.

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

Use ordinary Markdown link text when the surrounding sentence needs
pluralization, lowercase text, or an acronym. The rendered site enhances links
to `docs/terms/{slug}.md` with hover previews by reading `docs/terms.json`.
That build-generated endpoint also contains exhaustive consumer backlinks,
leaving `appears_in` free to remain curated. Do not duplicate tooltip text
inside paper pages; keep reusable descriptions on the term page.

## Term Link Validation

Run the glossary consistency check after adding or changing terms or their
consumer pages:

```bash
./scripts/run-in-workspace.sh python scripts/checks/term_links.py --fix
./scripts/run-in-workspace.sh python scripts/checks/term_links.py
./scripts/run-in-workspace.sh python scripts/checks/term_links.py \
  --review-mentions docs/path/to/changed-page.md
```

The default command performs structural validation only. It checks term
metadata, the glossary index, curated `appears_in` entries, "Where It Appears,"
consumer link targets, and reviewed exclusions without searching the whole wiki
for new link opportunities. `--fix` only adds missing "Where It Appears" links
for pages already selected in `appears_in`; it never promotes every consumer
into the curated set.

Use `--review-mentions` during authoring. With no paths it reviews all consumer
docs; with file or directory arguments it reviews only that scope. Add
`--strict-mentions` to make its findings errors. Existing Markdown link labels
are excluded because a navigation link already resolves the reader's need.
Plain-text mention decisions remain agent work because they require semantic
judgment.

The checker treats each term page as the source of truth for definition,
aliases, mention policy, and curated appearances; do not maintain a separate
hand-written exhaustive registry. Front matter, headings, code, image captions,
existing Markdown links, term pages, and log pages are excluded from mention
discovery. Glossary-to-glossary navigation is also excluded from `appears_in`
validation; keep those links under "Related Terms" instead.

When a detected occurrence is genuinely not a glossary reference, suppress
that occurrence with a reviewed HTML comment using the term page's filename
slug and a required reason:

```markdown
KV Cache is the name of this benchmark fixture.
<!-- termlint-ignore: kv-cache -- Fixture name, not the attention concept. -->
```

The comment may instead appear at the end of the prose line. A comment-only
directive applies only to the immediately preceding line. Suppressions are
occurrence-local: the checker continues looking for later unsuppressed mentions
on the same page. Unknown slugs, missing reasons, misplaced comments, comments
for already-linked terms, and comments whose target no longer contains a
detectable name are lint errors. Do not use suppressions to avoid linking a
meaningful first occurrence.

## Slug Convention

- Use kebab-case: `pipeline-parallelism.md`, `scatter-gather.md`, `kv-cache.md`.
- Prefer the shortest unambiguous name.
- If a term has a widely-used acronym or shorthand, use that as the canonical slug (e.g., `gqa.md` for Grouped-Query Attention) and list the full name in `aliases`.
