#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  cat <<'EOF'
Usage:
  ./scripts/bootstrap-external-repos.sh [REPOSITORY_KEY ...]
  ./scripts/bootstrap-external-repos.sh --status [--json] [REPOSITORY_KEY ...]

Without repository keys, materializes every commit-pinned checkout registered in
docs/_data/code_repositories.json. Use --status for an offline readiness report.
Run ./scripts/bootstrap-workspace.sh first.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  --status)
    shift
    exec "$repo_root/scripts/run-in-workspace.sh" \
      python scripts/repositories/worktree.py status "$@"
    ;;
  *)
    exec "$repo_root/scripts/run-in-workspace.sh" \
      python scripts/repositories/worktree.py materialize-all "$@"
    ;;
esac
