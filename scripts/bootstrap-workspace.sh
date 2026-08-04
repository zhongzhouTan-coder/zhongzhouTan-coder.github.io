#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace_dir="$repo_root/.workspace"
python_env="$workspace_dir/python"
ruby_tools="$workspace_dir/ruby-tools"

require_command() {
  local command_name=$1
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'Missing required runtime: %s\n' "$command_name" >&2
    exit 2
  fi
}

require_command python3
require_command ruby
require_command gem

python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' || {
  printf '%s\n' 'Python 3.10 or newer is required.' >&2
  exit 2
}
ruby -rrubygems -e 'exit Gem::Version.new(RUBY_VERSION) >= Gem::Version.new("3.2") ? 0 : 1' || {
  printf '%s\n' 'Ruby 3.2 or newer is required by the locked Bundler version.' >&2
  exit 2
}

mkdir -p "$workspace_dir"

# Install a workspace-local Node.js so that scripts (markdownlint, etc.) get a
# Node 20+ runtime even in sandbox environments where ambient Node is too old.
printf '%s\n' '==> Installing workspace Node.js'
node_env="$workspace_dir/node"
node_version='24.16.0'
if [[ ! -x "$node_env/bin/node" ]]; then
  require_command curl
  require_command tar
  node_arch='linux-x64'
  node_url="https://nodejs.org/dist/v${node_version}/node-v${node_version}-${node_arch}.tar.xz"
  printf '%s\n' "Downloading Node.js v${node_version}..."
  curl -fsSL "$node_url" | tar -xJ -C "$workspace_dir"
  mv "$workspace_dir/node-v${node_version}-${node_arch}" "$node_env"
  printf '%s\n' "Node.js v${node_version} installed at $node_env"
fi
# Use workspace Node.js for npm ci
export PATH="$node_env/bin:$PATH"
require_command node
require_command npm

printf '%s\n' '==> Creating the workspace Python environment'
# Some system Pythons (e.g. Debian/Ubuntu without the python3-venv package)
# lack ensurepip and create virtual environments without pip. Prefer a python3
# that can provision pip so the venv is usable immediately.
python3_cmd='python3'
if ! "$python3_cmd" -m ensurepip --version >/dev/null 2>&1; then
  for candidate in /usr/bin/python3 /usr/local/bin/python3 python3.13 python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -m ensurepip --version >/dev/null 2>&1; then
      python3_cmd="$candidate"
      printf '%s\n' "Using $python3_cmd (has ensurepip) to create the virtual environment"
      break
    fi
  done
fi
if [[ ! -x "$python_env/bin/python" ]]; then
  "$python3_cmd" -m venv "$python_env"
fi
# Guard against pip-less virtual environments.
"$python_env/bin/python" -m pip --version >/dev/null 2>&1 || {
  printf '%s\n' '==> Bootstrapping pip into the workspace Python environment'
  "$python_env/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || {
    printf '%s\n' 'The workspace Python environment has no pip and ensurepip is unavailable.' >&2
    printf '%s\n' 'Install the python3-venv package or use a Python that includes ensurepip.' >&2
    exit 2
  }
}
"$python_env/bin/python" -m pip install \
  --disable-pip-version-check \
  --requirement "$repo_root/requirements.txt"

printf '%s\n' '==> Installing workspace Node.js dependencies'
(cd "$repo_root" && npm ci)

printf '%s\n' '==> Installing the locked Bundler version in the workspace'
bundler_version=$(awk '
  /^BUNDLED WITH$/ { getline; gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print; exit }
' "$repo_root/docs/Gemfile.lock")
if [[ -z "$bundler_version" ]]; then
  printf '%s\n' 'Could not read BUNDLED WITH from docs/Gemfile.lock.' >&2
  exit 2
fi
gem install bundler \
  --version "$bundler_version" \
  --install-dir "$ruby_tools" \
  --bindir "$ruby_tools/bin" \
  --no-document

printf '%s\n' '==> Installing workspace Ruby dependencies'
system_gem_path=$(ruby -e 'print Gem.path.join(":")')
export GEM_HOME="$ruby_tools"
export GEM_PATH="$ruby_tools:$system_gem_path"
export PATH="$ruby_tools/bin:$PATH"
export BUNDLE_GEMFILE="$repo_root/docs/Gemfile"
export BUNDLE_PATH="$workspace_dir/bundle"
export BUNDLE_APP_CONFIG="$workspace_dir/bundle-config"
export BUNDLE_USER_HOME="$workspace_dir/bundler-home"
bundle install

printf '%s\n' '==> Workspace dependencies are ready'
printf '%s\n' 'Run repository tools with: ./scripts/run-in-workspace.sh <command> [args...]'
