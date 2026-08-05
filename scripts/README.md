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
| `checks/` | Validate source names, integrity, math, and code links. |
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
npm run ingest:web -- --help
./scripts/lint-docs.sh
```
