#!/usr/bin/env bash
set -u

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root" || exit 2

tmp_output=$(mktemp)
trap 'rm -f "$tmp_output"' EXIT

if ./scripts/lint-docs.sh >"$tmp_output" 2>&1; then
  cat "$tmp_output" >&2
  printf '{}\n'
  exit 0
fi

cat "$tmp_output" >&2
printf '%s\n' 'Stop hook docs lint failed. Fix the lint errors above, then run ./scripts/lint-docs.sh again.' >&2
exit 2
