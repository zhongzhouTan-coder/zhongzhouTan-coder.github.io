# Repository Evidence: Completion

## Complete the Workflow

1. Confirm the chosen mode and exact pinned revision.
2. Inspect only the declared scope and finish the evidence map.
3. Draft the page from the reader contract, then attach code evidence.
4. Update `sources.json`, navigation, related pages, and the chronological log
   only where repository rules require them.
5. Run `./scripts/lint-docs.sh`.

The normal lint is portable across agent workspaces: it checks repository
registry metadata and code-link structure but does not require every ignored
checkout to be materialized. Do not run markdownlint with a repository-wide
glob, because upstream Markdown under `external-repos/` is outside this wiki's
lint scope. When the checkout used for the current insight is available, add
the checkout-dependent verification separately:

```bash
./scripts/run-in-workspace.sh python scripts/checks/code_links.py --local
```

Completion requires agreement among raw metadata, derived metadata, the
manifest, and the code-repository registry; complete evidence coverage for
strict pages; an explicit verification boundary; and no unreported lint
failure.
