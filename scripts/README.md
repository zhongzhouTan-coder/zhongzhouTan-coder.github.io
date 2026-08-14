# Repository Scripts

The top-level shell scripts are stable entry points for common repository tasks.
Implementation commands are grouped by domain.

| Path | Purpose |
| --- | --- |
| `bootstrap-workspace.sh` | Install the locked workspace dependencies. |
| `bootstrap-external-repos.sh` | Inspect or materialize pinned external repositories. |
| `run-in-workspace.sh` | Run a command with workspace-managed runtimes. |
| `lint-docs.sh` | Run the complete documentation validation suite. |
| `serve-local.sh` | Serve the Jekyll site locally. |
| `checks/` | Validate source names, integrity, math, code links, and term links. |
| `wiki/` | Search and analyze the Markdown knowledge graph. |
| `ingestion/` | Convert PDF and web sources to Markdown. |
| `repositories/` | Initialize and manage pinned repository evidence. |
| `common/` | Shared Python support code; not invoked directly. |
| `hooks/` | Agent and automation hooks. |

Run commands from the repository root through the workspace wrapper when they
need Python, Node.js, Ruby, or Bundler dependencies. For example:

```bash
./scripts/run-in-workspace.sh python scripts/wiki/search.py "KV cache"
./scripts/run-in-workspace.sh python scripts/repositories/worktree.py --help
./scripts/run-in-workspace.sh python scripts/checks/term_links.py --fix
npm run ingest:web -- --help
./scripts/lint-docs.sh
```

The term-link fixer only adds missing "Where It Appears" links for pages already
curated in a term's `appears_in` metadata. Consumer links never expand that
curated set; exhaustive backlinks are generated in `docs/terms.json` when the
site builds.

Plain-text mention review is opt-in through `--review-mentions [PATH ...]` and
uses each term's `mention_lint` mode (`off`, `canonical`, or `aliases`). Existing
Markdown navigation links are not treated as unlinked mentions. Reviewed false
positives can use an occurrence-local `termlint-ignore` HTML comment with a term
slug and reason; invalid, unknown, or stale directives fail structural lint.
See `.github/instructions/docs-terms.instructions.md` for the full workflow.
