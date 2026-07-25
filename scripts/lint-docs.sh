#!/usr/bin/env bash
set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root" || exit 2

declare -A referenced_docs=()
declare -A reported_links=()
has_issue=0

mapfile -t markdown_files < <(find docs -type f -name '*.md' | sort)
mapfile -t docs_files < <(find docs -type f -name '*.md' | sort)

if [[ ${#markdown_files[@]} -eq 0 ]]; then
  exit 0
fi

extract_link_targets() {
  local file_path=$1
  grep -oE '\[[^][]+\]\(([^)]+)\)' "$file_path" | sed -E 's/.*\(([^)]+)\)$/\1/' || true
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

    if [[ "$target" == /* ]]; then
      resolved_target=$(realpath -m "$repo_root/$target")
    else
      resolved_target=$(realpath -m "$(dirname "$file_path")/$target")
    fi

    if [[ ! -e "$resolved_target" ]]; then
      issue_key="$file_path::$raw_target"
      if [[ -z ${reported_links["$issue_key"]+x} ]]; then
        report_issue "broken link: $file_path -> $raw_target"
        reported_links["$issue_key"]=1
      fi
      continue
    fi

    case "$resolved_target" in
      "$repo_root"/docs/*)
        referenced_docs["$resolved_target"]=1
        ;;
    esac
  done < <(extract_link_targets "$file_path")
done

for file_path in "${docs_files[@]}"; do
  resolved_file=$(realpath -m "$file_path")

  if [[ -z ${referenced_docs["$resolved_file"]+x} ]]; then
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

if (( has_issue != 0 )); then
  printf '%s\n' 'docs lint found issues; review warnings above before keeping or removing pages.'
  exit 1
fi

exit 0
