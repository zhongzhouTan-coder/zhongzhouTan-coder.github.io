#!/usr/bin/env bash
# Configure repository-local dependencies. This file is intended to be sourced.

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  workspace_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
else
  printf '%s\n' 'workspace-env.sh must be sourced by bash.' >&2
  return 2
fi

workspace_dir="$workspace_repo_root/.workspace"
workspace_python="$workspace_dir/python"
workspace_node="$workspace_dir/node"
workspace_ruby_tools="$workspace_dir/ruby-tools"

if [[ -d "$workspace_python/bin" ]]; then
  export VIRTUAL_ENV="$workspace_python"
  PATH="$workspace_python/bin:$PATH"
fi

# Prefer workspace-local Node.js (installed by bootstrap-workspace.sh) so that
# tools like markdownlint get a Node 20+ runtime even in sandbox environments.
if [[ -x "$workspace_node/bin/node" ]]; then
  PATH="$workspace_node/bin:$PATH"
fi

# Fall back to nvm-managed Node.js when no workspace-local Node.js exists.
_nvm_versions="${NVM_DIR:-$HOME/.nvm}/versions/node"
if [[ ! -x "$workspace_node/bin/node" && -d "$_nvm_versions" ]]; then
  _nvm_latest=$(find "$_nvm_versions" -maxdepth 2 -type f -name node -executable 2>/dev/null | sort -V | tail -1)
  if [[ -n "$_nvm_latest" ]]; then
    PATH="$(dirname "$_nvm_latest"):$PATH"
  fi
fi
unset _nvm_versions _nvm_latest

if [[ -d "$workspace_repo_root/node_modules/.bin" ]]; then
  PATH="$workspace_repo_root/node_modules/.bin:$PATH"
fi

if [[ -d "$workspace_ruby_tools/bin" ]]; then
  workspace_system_gem_path=$(ruby -e 'print Gem.path.join(":")')
  export GEM_HOME="$workspace_ruby_tools"
  export GEM_PATH="$workspace_ruby_tools:$workspace_system_gem_path"
  PATH="$workspace_ruby_tools/bin:$PATH"
fi

export PATH
export BUNDLE_GEMFILE="${BUNDLE_GEMFILE:-$workspace_repo_root/docs/Gemfile}"
if [[ -d "$workspace_dir" ]]; then
  export BUNDLE_PATH="$workspace_dir/bundle"
  export BUNDLE_APP_CONFIG="$workspace_dir/bundle-config"
  export BUNDLE_USER_HOME="$workspace_dir/bundler-home"
  export GEM_SPEC_CACHE="$workspace_dir/gem-spec-cache"
fi

unset workspace_dir workspace_python workspace_node workspace_repo_root workspace_ruby_tools
unset workspace_system_gem_path
