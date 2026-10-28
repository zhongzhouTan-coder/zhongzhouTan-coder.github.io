# Repository Evidence: Drafting and Code Links

## Draft Repository-Backed Docs

Every new or materially updated repository-backed page must:

- cite every supporting raw revision and derived note in front matter;
- state each inspected full commit SHA near the top;
- set both `code_links: strict` and `code_evidence: strict`;
- appear in each supporting manifest entry's `docs_paths`;
- use concrete paths and symbols only when they help the reader navigate;
- state static-reading, runtime-validation, dependency, and hardware limits;
- follow confidence rules from `docs-front-matter.instructions.md`.

Keep generated files, vendored dependencies, lockfiles, and large fixtures
low-priority unless central to the question. Put detailed symbol indexes and
recommended code-reading paths after the conceptual explanation when the page
targets beginners.

## Link Directly to Inspected Code

Insert revision-aware links while drafting. Link the first meaningful
occurrence of important files and symbols, every numbered runtime-flow step,
and non-obvious implementation evidence. Repeated mentions and generic language
constructs may remain ordinary inline code.

Use a short verbatim code excerpt when the actual condition, state write, core
call, or return helps the reader stay on the page. Pair it with a precise code
link and a sentence explaining what to notice. Prefer decisive operations over
incidental setup; keep excerpts contiguous, mark omissions, and compare their
lines against the pinned checkout. When an evidence row declares an end line,
the matching anchor must include `data-code-end-line`.

Use this structure:

```html
<a class="code-link"
   href="../../../external-repos/vllm/vllm/v1/core/sched/scheduler.py#L248"
   data-code-repo="vllm-a0c092ee72c0"
   data-code-path="vllm/v1/core/sched/scheduler.py"
   data-code-line="248"
   data-code-end-line="312"><code>Scheduler.schedule()</code></a>
```

The `href` is relative from the docs page to the registered checkout. The
Jekyll layout converts its `data-code-*` metadata to a provider URL pinned to
the full revision. Always provide a start line; use an end line only for a
short, complete range. Never use an ordinary Markdown link such as
`[source](../../../external-repos/repo/file.py)`: it bypasses revision metadata,
breaks when another agent has not materialized that checkout, and cannot be
rewritten to the pinned provider URL. The code-link checker rejects this form.
Never commit a machine-specific absolute path.
