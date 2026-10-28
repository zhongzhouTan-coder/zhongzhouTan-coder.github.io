# Glossary Terms: Ingestion and Index

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
