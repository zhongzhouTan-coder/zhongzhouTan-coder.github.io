---
description: "Use when creating or updating paper-insight docs pages under docs/. Enforces a deep-dive, memory-optimized body structure designed for fast comprehension and long-term retention of technical papers."
applyTo: "docs/**/*.md"
---

# Docs Content Structure Rules

Use this instruction together with `docs-front-matter.instructions.md` for every page under `docs/`. The front matter rules handle metadata; this file handles body content structure.

## Cognitive Principles

Every page should follow these principles for fast understanding and retention:

- **Big picture first.** The brain needs a mental scaffold before attaching details. Start with the one-sentence idea, then zoom in.
- **Progressive disclosure.** Layer information: must-know → should-know → nice-to-know.
- **Concrete before abstract.** Show an example or analogy before math and formalism.
- **Contrast encodes better than description.** "X does Y, unlike Z which does W" sticks better than "X does Y."
- **Visuals are memory anchors.** A diagram with numbered annotations is recalled far better than equivalent prose.
- **Symbols need names before equations.** Readers should not have to reverse-engineer notation such as `$k_t^C$` or `$W^{DKV}$`; introduce the naming convention and role before leaning on formulas.
- **Section headings as retrieval cues.** A reader should be able to scan headings alone and reconstruct the paper's argument.
- **Each section answers exactly one question the reader is naturally asking at that point.**

## Required Body Structure

Every paper-insight page must follow this section order. Each section has a specific cognitive purpose.

### 1. Paper Header (after the `#` title)

```markdown
**Paper:** [Full paper title]
**Authors:** [Author list]
**arXiv:** [ID and date]

**Related pages:** [Internal links]
```

### 2. TL;DR

Three distinct sentences answering three distinct questions. These serve as retrieval cues for the rest of the page.

```markdown
## TL;DR

**What:** [One sentence — the contribution.]
**How:** [One sentence — the mechanism.]
**The number:** [One sentence — the headline result.]
```

### 3. The Big Picture

Place the best original figure from the paper or captured web source **before**
any explanatory prose. The selected source figure should communicate the main
inputs, outputs, data flow, or novelty as directly as possible. Preserve it
under the page's local `assets/` directory and identify its source in the
caption.

Add numbered annotation captions so the reader can reconstruct the method from the diagram alone:

```markdown
## The Big Picture

![Descriptive alt text](./assets/original-figure-name.png)

*Source: [paper or captured web page]. ① [Step one.] ② [Step two.] ③ [Step three.]*
```

Do not redraw a suitable source figure in Mermaid, Draw.io, or another format.
If the sources contain no suitable Big Picture figure, use a compact table or
prose unless a synthesized visual is explicitly requested or materially
necessary; label any such fallback as synthesized.

### 4. Why This Exists

Concrete failure example that makes the reader feel the pain. Walk through a specific scenario showing what breaks without this paper's contribution. The example should be reusable — referenced again in the Deep Dive sections as spaced repetition.

### 5. The Landscape

An evolutionary tree, not a flat list. Show which prior ideas are parents,
siblings, or dead ends. **The Landscape must use Mermaid**, even when the paper
contains a related-work figure, because this diagram represents the knowledge
base's cross-source synthesis rather than a single source's framing. Save the
editable Mermaid source locally as `.mmd` and link it from the page.

```markdown
## The Landscape

[Mermaid diagram showing the phylogenetic relationship between prior work and this paper.]
```

### 6. The Core Idea

One paragraph. The unifying insight in plain language. No equations, no jargon. A reader who remembers only this paragraph should still understand the paper's contribution.

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

## Diagrams

- Search the paper extraction and captured web Markdown for relevant figures
  before creating a visual.
- Use original paper/web figures for the Big Picture, architecture, mechanism,
  ablation, result, and runtime sections whenever suitable figures exist.
- Preserve every selected source figure under the consuming page's local
  `assets/` directory; do not hotlink remote images as the only copy.
- Use Mermaid for **The Landscape**, save its editable `.mmd` source locally,
  and link that file from the page.
- Outside The Landscape, use a synthesized Mermaid, SVG/Draw.io, Excalidraw, or
  AI-generated visual only when there is no suitable source figure and the
  visual is explicitly requested or materially necessary.
- Every visual must have a caption explaining what the reader should see and
  whether it is an original source figure or a synthesized explanation.
- Source-figure captions must identify and link the paper or web source.
- Prefer numbered annotations (①, ②, ③) that are referenced in surrounding
  prose, without altering the original image.
- Place visuals before the text that explains them.

## Style Rules

- **Section headings must be scannable.** A reader scanning only headings should reconstruct the paper's argument.
- **One concept per paragraph.** If a paragraph makes two distinct points, split it.
- **Tables over prose for comparisons.** If you find yourself writing "X does A, while Y does B, and Z does C," use a table.
- **Bold the single most important phrase in each section.** This creates visual retrieval anchors for re-readers.
- **Reuse concrete examples across sections.** The same scenario appearing in "Why This Exists," "Deep Dive," and "Putting It Together" creates spaced repetition.
- **No embedded scripts or raw HTML.** Keep it GitHub Pages friendly.
- **Relative links for internal references.**

## Self-Test Checklist

Before publishing a page, verify:

1. **Scan headings only.** Can you reconstruct the paper's argument? If not, headings are too generic.
2. **Read only TL;DR + One Thing to Remember.** Do you understand the contribution? If not, those sections are too vague.
3. **Audit every visual.** Is The Landscape a linked local Mermaid file? For
   every other visual, was a suitable original paper/web figure used and saved
   locally? If a synthesized fallback was used, is its necessity and synthesized
   status clear?
4. **Is there a concrete example in "Why This Exists"?** If not, add one.
5. **If the page uses dense notation, is there a Symbol Map before Deep Dive?** If not, add a naming-convention paragraph and a compact symbol table.
6. **Does every Deep Dive subsection have an "intuition" sentence?** If not, you don't understand it well enough yet.
7. **Are limitations framed as failure modes with conditions?** If not, rewrite as a table.
8. **Does the results section teach interpretation, not just report numbers?** If not, add "how to read these numbers" guidance.

## Confidence Mapping

This instruction file does not modify the confidence mapping from `docs-front-matter.instructions.md`. Confidence (`high`/`medium`/`low`) is a front matter concern. The body structure described here applies regardless of confidence level.

## Exceptions

- Shorter reference pages (pure data, glossary entries) may omit sections 5 (The Landscape) and 8 (Putting It Together) if they add no value.
- Pages that are not paper insights (e.g., framework overviews, learning paths) should adapt this structure rather than follow it rigidly. The cognitive principles still apply.
- If the sources have no suitable original Big Picture figure and a synthesized
  fallback is not justified, section 3 may use a compact table or be omitted.
