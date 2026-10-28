# Paper Insight: Choosing Representations

## Representation Selection Workflow

Choose rich content by the reader question, not by a desire to decorate the
page. Before drafting, make a small representation plan in working notes for
each proposed rich element:

```text
Reader question: [What should become easier to understand?]
Evidence: [Which source figure, table, code path, or claim supports it?]
Representation: [Prose, example, table, trace, source visual, or synthesis.]
Teaching job: [What understanding would be lost if this were removed?]
```

Use the smallest representation that preserves the important relationship:

| Reader need | Preferred representation | Avoid |
|---|---|---|
| Understand one idea | Short prose plus a concrete example | A diagram that restates the paragraph |
| Decode names, symbols, shapes, or ownership | Symbol or mapping table | Dense prose enumeration |
| Compare methods, modes, or platforms | Comparison table | Parallel bullet lists |
| Follow changing state over time | Numbered worked trace | An unanchored generic flowchart |
| Understand hierarchy, branching, or concurrency | Original figure or justified diagram | A linear trace that hides topology |
| Inspect empirical evidence | Original chart or compact result table | Decorative charting of a few values |
| Notice a caveat or evidence boundary | Labeled callout | Repeated colored boxes |
| Understand prior-work relationships | Landscape Mermaid | A flat related-work list |

Search canonical source material for suitable figures and charts before
synthesizing a visual. There is no target number of visuals per page. After
drafting, apply the deletion test: if removing an element does not reduce
understanding, comparison, retrieval, or evidence quality, remove it.

## Callouts

Use portable Markdown blockquotes as semantic landmarks. Keep essential
reasoning in the normal page flow; a reader who skips callouts must still
understand the mechanism.

```markdown
> **Intuition:** [Plain-language mental model.]

> **Important:** [Requirement or invariant the reader must retain.]

> **Warning:** [Concrete misuse, failure condition, or interpretation risk.]

> **Evidence:** [What the source directly establishes.]

> **Inference:** [Repository-level conclusion derived from the evidence.]
```

Use only the labels needed by the page. Do not use callouts to add color, repeat
nearby prose, or hide long explanations. Distinguish `Evidence` from
`Inference` whenever a synthesized claim could otherwise be mistaken for a
source claim.

## Comparison Tables

When comparing methods, formats, or systems, prefer this table style:

```markdown
| Aspect | Baseline A | Baseline B | This work |
|---|---:|---:|---:|
| [Dimension 1] | ... | ... | ... |
| [Dimension 2] | ... | ... | ... |
```

Rules:

- Always include at least one baseline column for contrast.
- Use right-aligned number columns for numeric comparisons.
- Keep rows to the dimensions that actually differ.

## Related Page Maintenance

When creating or updating a paper-insight page:

- Read `docs/logs/index.md` first, then inspect likely related pages in the target topic folder before writing.
- Add internal links in the new page's **Related pages**, **The Landscape**, or **Go Deeper** sections when prior pages provide useful context.
- Update existing related pages when the new paper changes their comparison set, contradicts or refines an older claim, provides a better baseline, or should be mentioned as follow-up work.
- Prefer small, precise edits to related pages: add a cross-link, adjust confidence, add a comparison row, or clarify a limitation.
- Do not duplicate the new page's full explanation across older pages. Related pages should point to the new page and summarize only what changes the reader's understanding.
- If no related page needs changes, leave existing pages untouched and rely on the index/log updates.
