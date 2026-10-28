# Repository Evidence: Revision and Registration

## Core Invariants

- Choose **reuse**, **defer**, **new revision**, or **new repository** before
  reading code.
- A repository source record is immutable and pinned to one full commit SHA.
  Never rewrite old raw or derived evidence to point at another revision.
- Keep third-party worktrees beneath ignored `external-repos/` and treat them as
  read-only. Do not edit, format, switch, pull, or vendor their files.
- Use `scripts/repositories/worktree.py` for refresh and materialization; do not
  manage evidence worktrees with ad hoc destructive Git commands.
- Keep factual, file-referenced evidence under `derived/repo-analysis/`; reserve
  teaching, comparison, and interpretation for `docs/`.
- Distinguish static code inference from behavior verified by tests or runtime
  execution. Surface that boundary in every consuming page.

## Choose and Prepare the Revision

For an existing repository, start from the newest relevant registry entry and
compare only the requested scope:

```bash
./scripts/run-in-workspace.sh python scripts/repositories/worktree.py sync \
  <repo-slug>-<pinned-short-sha> \
  --path path/to/subsystem \
  --sparse path/to/subsystem
```

Repeat `--path` for every relevant file or directory. Omit it only when the
whole repository is genuinely in scope.

- `decision: reuse`: the scoped implementation is unchanged; reuse its raw
  record, derived evidence, and pinned checkout.
- `decision: defer`: relevant implementation changed, but the latest evidence
  snapshot is less than 14 days old. Keep using the pinned revision and do not
  scaffold or document the upstream candidate yet.
- `decision: new revision`: relevant implementation changed; initialize the
  revision-specific worktree created by the command.
- **new repository**: use only when no registry or source entry exists.

Do not create a new evidence revision for unrelated upstream changes, and do
not use `git pull` for freshness. The default promotion interval limits an
evidence chain to one new immutable revision every 14 days. Use
`--min-revision-interval-days N` to tune that interval, or
`--force-new-revision` only when a release, security fix, regression, or
explicit user request makes an immediate refresh necessary. A deferred result
is a successful freshness check, not permission to inspect the unpinned tip.

In a fresh workspace, check and materialize only the required registered
revisions with
`./scripts/bootstrap-external-repos.sh --status` and
`./scripts/bootstrap-external-repos.sh <repo-key> ...`.

Before writing, confirm the checkout is clean and capture its normalized
origin, provider, repository URL, full SHA, ref, and inspected date. A dirty
checkout is not reproducible from its SHA alone; accept it only when explicitly
required, record the dirty state, and use low confidence on consuming pages.

## Register Immutable Evidence

Use the scaffolder rather than hand-writing repository metadata:

```bash
./scripts/run-in-workspace.sh python scripts/repositories/init_source.py \
  external-repos/<repo> \
  --category <category> \
  --topic-path <topic/project> \
  --docs-path docs/<category>/<page>.md \
  --scope "Subsystem or question inspected" \
  --important-file "path/to/file.py::Why it matters"
```

The canonical artifacts are:

```text
raw/{category}/{topic-path}/{repo-slug}-codebase--{provider}-{short-sha}.md
derived/repo-analysis/{category}/{topic-path}/{repo-slug}/{full-sha}/
```

The scaffolder maintains the matching `sources.json` entry and
`docs/_data/code_repositories.json` registry record. Repository manifest entries
use `docs_paths` because one revision may support multiple pages. Every revision
requires `important-files.md`; add purpose-specific notes such as
`runtime-flow.md` or `module-map.md` only when they improve retrieval.

Keep derived notes factual. Record the exact command behind quantitative
codebase claims. Do not edit old revision evidence after it has been superseded;
create a new revision instead.
