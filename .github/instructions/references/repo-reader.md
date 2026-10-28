# Repository Evidence: Map and Reader Contract

## Build the Evidence Map First

Before prose drafting, copy the sections from
[`repository-evidence-template.md`](../repository-evidence-template.md) into
`important-files.md` or the directly supporting purpose-specific note.

For every important finding, record:

- consuming docs page;
- stable finding ID;
- repository-relative file and symbol;
- smallest useful start and optional end line;
- its position and direction in the end-to-end runtime flow, when applicable.

The evidence table is the handoff between code reading and prose drafting. Add
rows while investigating, not as a retrospective inventory. The code-link
checker requires every declared row to have a matching link in its consuming
page.

## Design the Page for Its Reader

Choose the page type explicitly:

- A **code-reading map** teaches navigation, runtime flow, modules, extension
  points, and failure surfaces.
- A **synthesis page** teaches a mechanism or comparison using evidence from
  one or more repositories.

Before drafting either type, record a small reader contract in the working
notes:

- intended audience and assumed prerequisites;
- the question the page answers;
- a one-sentence, code-free mental model;
- which behavior occurs offline, at load time, and at runtime;
- hardware, configuration, fallback, and verification limitations.

When several paths exist, select one supported branch for the first walkthrough.
State the configuration, hardware, workload, and entry/return boundaries that
select it; do not imply that an uninspected checkpoint or unexecuted backend
was observed. Explain materially different branches after the main path.

Draft concepts before implementation names. Expand important acronyms on first
use, explain why each stage exists, and introduce concrete files and symbols
only after the reader has a system-level map. For beginner-facing pages, keep
the initial path to these answers short:

1. What is the mechanism and why does it exist?
2. What data or state changes?
3. When does each change happen?
4. Where do the supported platforms diverge?
5. What was inferred statically versus verified at runtime?

Plan each rich element around a reader question and a teaching job. Use prose
for one idea, a table for exact mappings or platform differences, and a worked
trace for runtime state changes. When a flow diagram materially improves
hierarchy, branching, concurrency, or ownership, save its editable source
locally and link it from the page (`.mmd` for Mermaid or `.drawio` for Draw.io).
Prefer one reader question per visual; keep a symbol-heavy implementation map
separate from the first conceptual visual. Omit any diagram that only restates
nearby prose.
Put a compact input/state/output path before a long sequence diagram or module
inventory. A paper's architecture figure can explain design intent, but it is
not evidence that the repository implements the same call path.

Place a code-path figure immediately before the worked trace it prepares the
reader to follow. State the selected workload and branch first. Beside the
figure, give a short reading guide in visual order: entry, what each lane or
symbol represents, the important state handoff, and the result that leaves the
figure. Explain conditional roles and distinguish intermediate state from the
user-visible result. Do not make readers cross a long table to find this guide.
Use the same names in the figure and trace, and number the figure stages when
that helps readers match them to trace steps. Caption a source figure as design
intent or a synthesized figure as an explanation of the pinned code branch;
name any unverified configuration or runtime assumptions. Keep revision-aware
code links in the adjacent trace, including both producer and consumer of
shared state; the figure itself is orientation, not code evidence.
