#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ ! -x "$repo_root/.workspace/python/bin/python" \
  || ! -d "$repo_root/node_modules" ]]; then
  printf '%s\n' \
    'Workspace dependencies are missing. Run ./scripts/bootstrap-workspace.sh first.' >&2
  exit 2
fi

# shellcheck source=workspace-env.sh
source "$repo_root/scripts/workspace-env.sh"

if ! bundle check >/dev/null 2>&1; then
  printf '%s\n' \
    'Workspace gems are missing or stale. Run ./scripts/bootstrap-workspace.sh.' >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  printf '%s\n' 'Usage: ./scripts/run-in-workspace.sh <command> [args...]' >&2
  exit 2
fi

cd "$repo_root"
exec "$@"
