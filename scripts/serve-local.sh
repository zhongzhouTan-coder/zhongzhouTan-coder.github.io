#!/usr/bin/env bash
# serve-local.sh — Serve the docs/ site locally with Jekyll, matching GitHub Pages behaviour.
#
# Usage:
#   ./scripts/serve-local.sh          # serve on http://localhost:4000
#   ./scripts/serve-local.sh --drafts # also render draft pages
#   PORT=5000 ./scripts/serve-local.sh
#
# Prerequisites:
#   Run once: ./scripts/bootstrap-workspace.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
PORT="${PORT:-4000}"

# shellcheck source=workspace-env.sh
source "$REPO_ROOT/scripts/workspace-env.sh"

if ! bundle check >/dev/null 2>&1; then
  echo "Workspace gems are missing; run ./scripts/bootstrap-workspace.sh first." >&2
  exit 2
fi

echo "==> Changing to docs/ directory: $DOCS_DIR"
cd "$DOCS_DIR"

echo "==> Starting Jekyll server on http://localhost:$PORT"
echo "    Press Ctrl-C to stop."
echo ""

bundle exec jekyll serve \
  --port "$PORT" \
  --host 0.0.0.0 \
  --baseurl "" \
  --watch \
  "$@" 2>&1 | grep -v "bad Request-Line\|'/favicon.ico' not found" || true
