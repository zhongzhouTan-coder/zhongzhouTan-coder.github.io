# Paper Insight: Opening and Core Idea

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
