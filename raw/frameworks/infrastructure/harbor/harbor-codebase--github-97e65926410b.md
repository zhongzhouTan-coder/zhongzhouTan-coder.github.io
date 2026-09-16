---
kind: repository-source
provider: github
clone_url: git@github.com:harbor-framework/harbor.git
repository_url: https://github.com/harbor-framework/harbor
local_checkout: external-repos/harbor/
commit: 97e65926410ba6b9cef15cd5d9bf9a0c700cef26
ref: main
inspected: 2026-08-04
checkout_state: clean
---

# Harbor Codebase Source Record

## Reading Scope

- Core runtime architecture for agent evaluation: task packaging model, job/trial orchestration, agent/environment/verifier contracts, task distribution, and compile/exec workflows (commit 97e65926410b).

## Important Entry Files

- `src/harbor/models/task/task.py` — Task task directory model (instruction.md, task.toml, environment/, tests/, solution/, multi-step)
- `src/harbor/models/task/config.py` — TaskConfig task.toml schema (network policy, verifier, agent, environment, artifacts, steps)
- `src/harbor/models/job/config.py` — JobConfig and DatasetConfig job schema and dataset source resolution
- `src/harbor/models/trial/config.py` — TrialConfig per-trial config (task x agent x attempt)
- `src/harbor/job.py` — Job orchestration entry point (create, run, hooks)
- `src/harbor/job_plan.py` — JobPlan resolves task/trial configs, metrics, task caching, pass@k aggregation
- `src/harbor/trial/trial.py` — Trial base lifecycle (prepare, agent phase, verifier phase, finalize)
- `src/harbor/trial/single_step.py` — SingleStepTrial one instruction, one agent run, one verifier
- `src/harbor/trial/multi_step.py` — MultiStepTrial sequential named steps with per-step verification
- `src/harbor/trial/queue.py` — TrialQueue concurrency limits, per-agent semaphores, retry backoff
- `src/harbor/agents/base.py` — BaseAgent agent contract (setup, run, ATIF/resume/config capabilities)
- `src/harbor/environments/base.py` — BaseEnvironment environment contract (start, exec, network policy, mounts, transfers)
- `src/harbor/verifier/verifier.py` — Verifier runs tests in environment and parses reward output
- `src/harbor/compile/compiler.py` — Compiler compiles CompileConfig into task directories incl auto-verifier
- `src/harbor/exec/executor.py` — Executor map/reduce compile+job workflow
- `src/harbor/tasks/client.py` — TaskClient downloads git/local/package tasks into cache

## Limitations

- Static code reading only; runtime behavior was not executed.
