# Paper Insight: Style and Review

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
   status clear? Can you state the reader question and teaching job in one
   sentence, and would understanding actually decrease if the visual vanished?
4. **Is there a concrete example in "Why This Exists"?** If not, add one.
5. **If the page uses dense notation, is there a Symbol Map before Deep Dive?** If not, add a naming-convention paragraph and a compact symbol table.
6. **Does every Deep Dive subsection have an "intuition" sentence?** If not, you don't understand it well enough yet.
7. **Are limitations framed as failure modes with conditions?** If not, rewrite as a table.
8. **Does the results section teach interpretation, not just report numbers?** If not, add "how to read these numbers" guidance.
9. **Does "Putting It Together" track one concrete object and expose each state
   transition?** If not, rewrite it as a numbered trace or compact state table.
10. **Are callouts sparse and semantic?** If a callout repeats prose, hides
    essential reasoning, or exists only for color, remove or rewrite it.
11. **Are charts auditable?** Verify the cited values, axes, units, conditions,
    and the boundary between reported data and repository-derived calculations.

## Confidence Mapping

This instruction file does not modify the confidence mapping from `docs-front-matter.instructions.md`. Confidence (`high`/`medium`/`low`) is a front matter concern. The body structure described here applies regardless of confidence level.

## Exceptions

- Shorter reference pages (pure data, glossary entries) may omit sections 5 (The Landscape) and 8 (Putting It Together) if they add no value.
- Pages that are not paper insights (e.g., framework overviews, learning paths) should adapt this structure rather than follow it rigidly. The cognitive principles still apply.
- If the sources have no suitable original Big Picture figure and a synthesized
  fallback is not justified, section 3 may use a compact table or be omitted.
