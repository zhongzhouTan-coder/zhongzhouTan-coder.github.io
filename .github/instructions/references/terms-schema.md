# Glossary Terms: Schema

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
- All other front matter fields follow [`docs-front-matter.instructions.md`](../docs-front-matter.instructions.md).
