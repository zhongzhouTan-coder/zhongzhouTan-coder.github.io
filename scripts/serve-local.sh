#!/usr/bin/env bash
# serve-local.sh — Serve the docs/ site locally with Jekyll, matching GitHub Pages behaviour.
#
# Usage:
#   ./scripts/serve-local.sh          # serve on http://localhost:4000
#   ./scripts/serve-local.sh --drafts # also render draft pages
#   PORT=5000 ./scripts/serve-local.sh
#
# Prerequisites:
#   ruby >= 3.1, bundler >= 2.x
#   Run once: cd docs && bundle install

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
PORT="${PORT:-4000}"

echo "==> Changing to docs/ directory: $DOCS_DIR"
cd "$DOCS_DIR"

echo "==> Installing / updating gems (bundle install)..."
bundle install --quiet

echo "==> Starting Jekyll server on http://localhost:$PORT"
echo "    Press Ctrl-C to stop."
echo ""

bundle exec jekyll serve \
  --port "$PORT" \
  --host 0.0.0.0 \
  --baseurl "" \
  --watch \
  "$@" 2>&1 | grep -v "bad Request-Line\|'/favicon.ico' not found" || true
