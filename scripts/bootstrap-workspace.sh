#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace_dir="$repo_root/.workspace"
python_env="$workspace_dir/python"
ruby_tools="$workspace_dir/ruby-tools"
stamps_dir="$workspace_dir/stamps"

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
ruby_version=$(tr -d '[:space:]' < "$repo_root/.ruby-version")
if [[ -z "$ruby_version" ]]; then
  printf '%s\n' 'Could not read the Ruby version from .ruby-version.' >&2
  exit 2
fi
ruby -e 'exit RUBY_VERSION == ARGV.fetch(0) ? 0 : 1' "$ruby_version" || {
  printf '%s\n' "Ruby $ruby_version is required; found $(ruby --version)." >&2
  exit 2
}

mkdir -p "$workspace_dir" "$stamps_dir"
export GEM_SPEC_CACHE="$workspace_dir/gem-spec-cache"
export BUNDLE_USER_HOME="$workspace_dir/bundler-home"

# Install a workspace-local Node.js so that scripts (markdownlint, etc.) get a
# Node 20+ runtime even in sandbox environments where ambient Node is too old.
printf '%s\n' '==> Installing workspace Node.js'
node_env="$workspace_dir/node"
node_version=$(tr -d '[:space:]' < "$repo_root/.node-version")
if [[ -z "$node_version" ]]; then
  printf '%s\n' 'Could not read the Node.js version from .node-version.' >&2
  exit 2
fi
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
installed_node_version=$("$node_env/bin/node" --version)
if [[ "$installed_node_version" != "v$node_version" ]]; then
  printf '%s\n' \
    "Workspace Node.js is $installed_node_version; expected v$node_version." >&2
  printf '%s\n' \
    'Remove .workspace/node and rerun ./scripts/bootstrap-workspace.sh.' >&2
  exit 2
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
python_env_created=0
if [[ ! -x "$python_env/bin/python" ]]; then
  "$python3_cmd" -m venv "$python_env"
  python_env_created=1
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
python_fingerprint=$(
  {
    sha256sum "$repo_root/requirements.txt"
    "$python_env/bin/python" --version
  } | sha256sum | awk '{print $1}'
)
python_stamp="$stamps_dir/python-dependencies.sha256"
if [[ $python_env_created -eq 0 \
  && -f "$python_stamp" \
  && "$(<"$python_stamp")" == "$python_fingerprint" ]]; then
  printf '%s\n' '==> Workspace Python dependencies are up to date'
else
  "$python_env/bin/python" -m pip install \
    --disable-pip-version-check \
    --requirement "$repo_root/requirements.txt"
  printf '%s\n' "$python_fingerprint" > "$python_stamp"
fi

printf '%s\n' '==> Installing workspace Node.js dependencies'
node_fingerprint=$(
  sha256sum \
    "$repo_root/.node-version" \
    "$repo_root/package.json" \
    "$repo_root/package-lock.json" \
    | sha256sum | awk '{print $1}'
)
node_stamp="$stamps_dir/node-dependencies.sha256"
if [[ -d "$repo_root/node_modules" \
  && -f "$node_stamp" \
  && "$(<"$node_stamp")" == "$node_fingerprint" ]]; then
  printf '%s\n' '==> Workspace Node.js dependencies are up to date'
else
  (cd "$repo_root" && npm ci)
  printf '%s\n' "$node_fingerprint" > "$node_stamp"
fi

printf '%s\n' '==> Installing the locked Bundler version in the workspace'
bundler_version=$(awk '
  /^BUNDLED WITH$/ { getline; gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print; exit }
' "$repo_root/docs/Gemfile.lock")
if [[ -z "$bundler_version" ]]; then
  printf '%s\n' 'Could not read BUNDLED WITH from docs/Gemfile.lock.' >&2
  exit 2
fi
if [[ ! -x "$ruby_tools/bin/bundle" \
  || "$("$ruby_tools/bin/bundle" --version)" != "$bundler_version" ]]; then
  gem install bundler \
    --version "$bundler_version" \
    --install-dir "$ruby_tools" \
    --bindir "$ruby_tools/bin" \
    --no-document
else
  printf '%s\n' "==> Bundler $bundler_version is up to date"
fi

printf '%s\n' '==> Installing workspace Ruby dependencies'
system_gem_path=$(ruby -e 'print Gem.path.join(":")')
export GEM_HOME="$ruby_tools"
export GEM_PATH="$ruby_tools:$system_gem_path"
export PATH="$ruby_tools/bin:$PATH"
export BUNDLE_GEMFILE="$repo_root/docs/Gemfile"
export BUNDLE_PATH="$workspace_dir/bundle"
export BUNDLE_APP_CONFIG="$workspace_dir/bundle-config"
if bundle check >/dev/null 2>&1; then
  printf '%s\n' '==> Workspace Ruby dependencies are up to date'
else
  bundle install
fi

printf '%s\n' '==> Workspace dependencies are ready'
printf '%s\n' 'Run repository tools with: ./scripts/run-in-workspace.sh <command> [args...]'
