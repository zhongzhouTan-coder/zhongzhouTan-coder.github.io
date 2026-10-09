#!/usr/bin/env bash
set -o pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root" || exit 2

# Prefer dependencies installed by scripts/bootstrap-workspace.sh while retaining
# an ambient-runtime fallback for CI and first-time setup diagnostics.
# shellcheck source=workspace-env.sh
source "$repo_root/scripts/workspace-env.sh"
python_command=python3
if [[ -x "$repo_root/.workspace/python/bin/python" ]]; then
  python_command="$repo_root/.workspace/python/bin/python"
fi

lint_tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/lint-docs.XXXXXX") || exit 2
trap 'rm -rf "$lint_tmp_dir"' EXIT
referenced_docs_file="$lint_tmp_dir/referenced-docs"
reported_links_file="$lint_tmp_dir/reported-links"
: > "$referenced_docs_file"
: > "$reported_links_file"
has_issue=0

markdown_files=()
while IFS= read -r file_path; do
  markdown_files+=("$file_path")
done < <(find docs -type f -name '*.md' | sort)
docs_files=("${markdown_files[@]}")

if [[ ${#markdown_files[@]} -eq 0 ]]; then
  exit 0
fi

extract_link_targets() {
  local file_path=$1
  grep -oE '\[[^][]+\]\(([^)]+)\)' "$file_path" | sed -E 's/.*\(([^)]+)\)$/\1/' || true
}

canonical_path() {
  local input=$1
  local parent=${input%/*}
  local name=${input##*/}
  [[ "$parent" == "$input" ]] && parent=.
  [[ -z "$parent" ]] && parent=/
  if (cd -P "$parent" 2>/dev/null && printf '%s/%s\n' "$PWD" "$name"); then
    return 0
  fi
  "$python_command" -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

check_front_matter_field() {
  local file_path=$1
  local field_name=$2
  awk '
    BEGIN { in_block = 0; block_count = 0 }
    /^---[[:space:]]*$/ {
      block_count++
      if (block_count == 1) {
        in_block = 1
        next
      }
      if (block_count == 2) {
        exit
      }
    }
    in_block { print }
  ' "$file_path" | grep -Eq "^${field_name}:"
}

report_issue() {
  local message=$1
  printf '%s\n' "$message"

  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    emit_github_annotation "$message"
  fi

  has_issue=1
}

emit_github_annotation() {
  local message=$1
  local file_path=
  local annotation_message=$message

  case "$message" in
    "broken link: "*)
      file_path=${message#broken link: }
      file_path=${file_path%% -> *}
      ;;
    "orphan docs page: "*)
      file_path=${message#orphan docs page: }
      file_path=${file_path%% is not linked from docs/ or logs/}
      ;;
    "missing front matter: "*)
      file_path=${message#missing front matter: }
      ;;
    "missing front matter field "*)
      file_path=${message##*: }
      ;;
    "stub page: "*)
      file_path=${message#stub page: }
      file_path=${file_path%% has fewer than 8 non-empty lines}
      ;;
    "placeholder text found: "*)
      file_path=${message#placeholder text found: }
      ;;
  esac

  if [[ -n "$file_path" ]]; then
    printf '::warning file=%s::%s\n' "$file_path" "$annotation_message"
  else
    printf '::warning::%s\n' "$annotation_message"
  fi
}

for file_path in "${markdown_files[@]}"; do
  while IFS= read -r raw_target; do
    [[ -z "$raw_target" ]] && continue

    case "$raw_target" in
      http://*|https://*|mailto:*|\#*)
        continue
        ;;
    esac

    target=${raw_target%%#*}
    [[ -z "$target" ]] && continue

    # Repository checkout links are validated structurally by code_links.py.
    # Do not make the generic link pass depend on an ignored checkout being
    # materialized in the current agent workspace.
    if [[ "$target" == *external-repos/* ]]; then
      continue
    fi

    if [[ "$target" == /* ]]; then
      resolved_target=$(canonical_path "$repo_root/$target")
    else
      resolved_target=$(canonical_path "$(dirname "$file_path")/$target")
    fi

    if [[ ! -e "$resolved_target" ]]; then
      issue_key="$file_path::$raw_target"
      if ! grep -Fqx -- "$issue_key" "$reported_links_file"; then
        report_issue "broken link: $file_path -> $raw_target"
        printf '%s\n' "$issue_key" >> "$reported_links_file"
      fi
      continue
    fi

    case "$resolved_target" in
      "$repo_root"/docs/*)
        printf '%s\n' "$resolved_target" >> "$referenced_docs_file"
        ;;
    esac
  done < <(extract_link_targets "$file_path")
done

for file_path in "${docs_files[@]}"; do
  resolved_file=$(canonical_path "$file_path")

  if ! grep -Fqx -- "$resolved_file" "$referenced_docs_file"; then
    report_issue "orphan docs page: $file_path is not linked from docs/ or logs/"
  fi

  if ! head -n 1 "$file_path" | grep -qx -- '---'; then
    report_issue "missing front matter: $file_path"
  fi

  for required_field in title summary layout confidence sources updated; do
    if ! check_front_matter_field "$file_path" "$required_field"; then
      report_issue "missing front matter field '$required_field': $file_path"
    fi
  done

  non_empty_lines=$(awk 'NF { count++ } END { print count + 0 }' "$file_path")
  if (( non_empty_lines < 8 )); then
    report_issue "stub page: $file_path has fewer than 8 non-empty lines"
  fi

  if grep -Eniq '\b(todo|tbd|fixme|coming soon)\b' "$file_path"; then
    report_issue "placeholder text found: $file_path"
  fi
done

# --- Orphan image check ---
# Find image files under docs/ (excluding Jekyll _site/ build output) and verify
# each is referenced by at least one markdown file.
image_files=()
while IFS= read -r file_path; do
  image_files+=("$file_path")
done < <(find docs -type f \( \
  -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.gif' \
  -o -iname '*.svg' -o -iname '*.webp' -o -iname '*.drawio' \
  -o -iname '*.mmd' -o -iname '*.excalidraw' \
  \) -not -path 'docs/_site/*' | sort)

if [[ ${#image_files[@]} -gt 0 ]]; then
  for img_path in "${image_files[@]}"; do
    img_basename=$(basename "$img_path")
    img_dir=$(dirname "$img_path")

    # .drawio.svg files are renders of .drawio sources; skip orphan check when
    # a same-directory .drawio source exists (the source file is canonical).
    if [[ "$img_basename" == *.drawio.svg ]]; then
      drawio_source="${img_basename%.drawio.svg}.drawio"
      if [[ -f "$img_dir/$drawio_source" ]]; then
        continue
      fi
    fi

    referenced=0
    for md_file in "${markdown_files[@]}"; do
      if grep -Fq "$img_basename" "$md_file" 2>/dev/null; then
        referenced=1
        break
      fi
    done

    if [[ $referenced -eq 0 ]]; then
      report_issue "orphan image: $img_path is not referenced by any docs/ markdown file"
    fi
  done
fi

# --- markdownlint ---
# Require markdownlint-cli2 so the aggregate check cannot report success after
# silently skipping Markdown validation.
markdownlint_cmd=""
if command -v npx &>/dev/null; then
  markdownlint_cmd="npx --no-install markdownlint-cli2"
elif command -v markdownlint-cli2 &>/dev/null; then
  markdownlint_cmd="markdownlint-cli2"
fi

if [[ -n "$markdownlint_cmd" ]] && [[ -f "$repo_root/.markdownlint-cli2.jsonc" ]]; then
  printf '\n%s\n' '--- markdownlint ---'
  if ! $markdownlint_cmd; then
    has_issue=1
  fi
elif [[ -f "$repo_root/.markdownlint-cli2.jsonc" ]]; then
  report_issue 'markdownlint-cli2 is unavailable; run ./scripts/bootstrap-workspace.sh'
fi

if (( has_issue != 0 )); then
  printf '%s\n' 'docs lint found issues; review warnings above before keeping or removing pages.'
  exit 1
fi

printf '\n%s\n' '--- source names ---'
if ! "$python_command" scripts/checks/source_names.py; then
  exit 1
fi

printf '\n%s\n' '--- kb integrity ---'
if ! "$python_command" scripts/checks/repository_integrity.py; then
  exit 1
fi

printf '\n%s\n' '--- math formulations ---'
if ! "$python_command" scripts/checks/math_rendering.py; then
  exit 1
fi

printf '\n%s\n' '--- repository code links ---'
if ! "$python_command" scripts/checks/code_links.py; then
  exit 1
fi

printf '\n%s\n' '--- term links ---'
if ! "$python_command" scripts/checks/term_links.py; then
  exit 1
fi

exit 0
