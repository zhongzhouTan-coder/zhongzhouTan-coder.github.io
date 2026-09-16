# Pier inspection notes

Date: 2026-07-09

## Scope

This note summarizes the local `pier` repository inspection and the related
`mini-swe-agent` repository inspection used to produce the published docs page.

## Main source files

- `pier/README.md`
- `pier/src/pier/cli/main.py`
- `pier/src/pier/cli/jobs.py`
- `pier/src/pier/job.py`
- `pier/src/pier/trial/execution.py`
- `pier/src/pier/trial/trial.py`
- `pier/src/pier/agents/factory.py`
- `pier/src/pier/environments/factory.py`
- `pier/src/pier/environments/agent_setup.py`
- `pier/src/pier/agents/installed/base.py`
- `pier/src/pier/agents/installed/mini_swe_agent.py`
- `pier/src/pier/models/job/config.py`
- `pier/src/pier/models/trial/config.py`
- `pier/src/pier/models/trajectories/step.py`
- `pier/tests/test_mini_swe_agent_conversion.py`
- `pier/tests/test_network_allowlists.py`
- `pier/tests/test_filtered_egress_env.py`
- `mini-swe-agent/src/minisweagent/agents/default.py`
- `mini-swe-agent/src/minisweagent/run/mini.py`
- `mini-swe-agent/docs/usage/mini.md`
- `mini-swe-agent/docs/usage/swebench.md`

## Why Pier exists

`pier/README.md` describes Pier as a Harbor-compatible fork for sandboxed coding
agent evaluation. The stated reasons for the fork are:

- a smaller, more opinionated base than Harbor,
- support for installed agents inside air-gapped tasks where
  `allow_internet = false`,
- augmented ATIF v1.7 trajectories,
- a chat-style viewer,
- critique jobs that inspect finished trials in a fresh sandbox.

The README is explicit that Pier is not a benchmark corpus by itself. It is a
runner and analysis harness for Harbor-style task directories.

## Architecture

The code path is straightforward:

1. `pier/src/pier/cli/main.py` registers the `run`, `job`, `view`, `critique`,
   `check`, and `analyze` commands.
2. `pier/src/pier/cli/jobs.py` converts CLI/config input into `JobConfig`.
3. `pier/src/pier/job.py` resolves datasets/tasks, creates trial configs, and
   schedules trials concurrently.
4. `pier/src/pier/trial/execution.py` builds the concrete agent and environment
   using `AgentFactory` and `EnvironmentFactory`.
5. `pier/src/pier/environments/factory.py` passes the agent's install spec and
   network allowlist into the environment implementation.
6. `pier/src/pier/trial/trial.py` runs the lifecycle: start environment,
   healthcheck, setup agent, run agent, collect logs/artifacts, verify, save
   result, and clean up.

The key design point is that installed agents are treated as first-class
objects with:

- an install spec,
- runtime env vars,
- a network allowlist,
- a post-run trajectory conversion hook.

## Air-gapped installed-agent support

`pier/src/pier/environments/agent_setup.py` shows how Pier implements filtered
egress for installed agents:

- install steps become Dockerfile commands or equivalent environment setup,
- allowlisted domains are passed through a proxy policy,
- agent process envs get proxy variables only for agent commands, not for the
  main task container globally.

`pier/tests/test_filtered_egress_env.py` checks that proxy env vars are only
added to agent processes and not injected into the main compose service env.

## Trial lifecycle details

`pier/src/pier/trial/trial.py` shows the practical order of operations:

- start sandbox,
- run environment healthcheck,
- install/setup the agent,
- run the agent with the task instruction,
- download logs from the sandbox if the environment is not mounted,
- let the installed agent populate context post-run,
- optionally upload generated logs back into the sandbox so the verifier can
  read files such as `trajectory.json`,
- collect artifacts,
- run verifier,
- persist `result.json` and directories under `jobs/<job>/<trial>/`.

For multi-step tasks, Pier repeats the same pattern per step and relocates each
step's agent, verifier, and artifact outputs into step-specific directories.

## ATIF emphasis

Pier's ATIF model is stricter than "just save a chat log".

`pier/src/pier/models/trajectories/step.py` enforces:

- only `agent` steps can carry model info, reasoning, tool calls, or metrics,
- deterministic agent dispatches with `llm_call_count == 0` cannot carry LLM
  reasoning or metrics,
- timestamps are validated as ISO 8601 strings.

The README also claims Pier preserves one step per API turn, separates visible
assistant text from reasoning, and tracks fields such as `peak_context_tokens`,
`summarization_count`, and `llm_call_count`.

## mini-swe-agent integration

`pier/src/pier/agents/installed/mini_swe_agent.py` is the most important file
for the DeepSWE-style question.

Observed behavior:

- install step uses `uv tool install mini-swe-agent`,
- extra Python packages can be injected for provider-specific needs,
- the LiteLLM model cost map backup is refreshed during install,
- provider base URLs are collected from env vars and optional `config_yaml`,
- those URLs are converted into a network allowlist,
- OpenAI-prefixed models default to `litellm_response`,
- OpenRouter-prefixed models default to `openrouter`,
- the run path writes native mini-swe output to
  `/logs/agent/mini-swe-agent.trajectory.json`,
- Pier converts that native trajectory to ATIF and saves
  `/logs/agent/trajectory.json`.

The internal command built by the adapter is effectively:

`mini-swe-agent --yolo --model=... --task=... --output=/logs/agent/mini-swe-agent.trajectory.json ...`

The converter keeps:

- visible assistant text,
- reasoning content,
- tool calls,
- tool observations,
- token usage,
- cached-token accounting,
- timestamps,
- per-step and final metrics.

## mini-swe-agent source context

The local `mini-swe-agent` repo confirms the adapter assumptions:

- `mini-swe-agent/src/minisweagent/run/mini.py` is the main local CLI,
- `mini-swe-agent/docs/usage/mini.md` documents `mini` as the REPL-style CLI,
- `mini-swe-agent/docs/usage/swebench.md` documents SWE-bench batch and
  single-instance workflows,
- `mini-swe-agent/src/minisweagent/agents/default.py` shows the simple control
  loop: query model, execute actions, append observations, save trajectory.

This matters because DeepSWE's public methodology page says it uses
`mini-swe-agent` as a shared bash-only harness. Pier's adapter is therefore a
practical way to run the same style of agent loop inside Pier while gaining
Pier's trajectory and sandbox plumbing.

## DeepSWE-style usage: what is directly supported and what is inferred

Directly supported by Pier:

- local Harbor-format task directories,
- dataset subsets via local `path`,
- `mini-swe-agent` as an installed agent,
- docker and modal environments in the README examples,
- deterministic sampling with `n_tasks` and `sample_seed`.

Important limitation from code:

- `pier/src/pier/models/job/config.py` rejects registry and package datasets in
  this build and only accepts local dataset `path`.

Important limitation from README:

- the README shows how to use Harbor to download `swebenchpro` first, then run
  Pier on the resulting local dataset path.

Inference:

- there is no first-party `pier deepswe` command and no DeepSWE-specific
  downloader in the inspected code,
- so running DeepSWE with Pier requires a local Harbor-style export or
  conversion of the DeepSWE task set before `pier run` can consume it.

This is an inference from the task-loading code plus the README examples, not a
claim explicitly documented as "DeepSWE support" by Pier.

## Useful concrete config shape

Representative job config:

```yaml
environment:
  type: docker

agents:
  - name: mini-swe-agent
    model_name: openai/gpt-5.5
    env:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    kwargs:
      reasoning_effort: xhigh
      set_cache_control: default_end

datasets:
  - path: /absolute/path/to/local-harbor-dataset
    n_tasks: 10
    sample_seed: 0
```

Run:

```bash
pier run -c pier-mini.yaml
```

If provider routing uses custom base URLs or custom mini-swe YAML, Pier's
allowlist logic will read those URLs from `agent.env` and `kwargs.config_yaml`.

## Notable code/README mismatch

The README says the environments that work today are `docker` and `modal`.
However, `pier/src/pier/environments/factory.py` also registers a
`daytona` environment class.

The safest interpretation is:

- `docker` and `modal` are the documented primary paths,
- Daytona exists in code but is not presented in the README as equally mature.
