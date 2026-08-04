---
kind: repository-analysis
repository_id: github:harbor-framework/harbor@97e65926410ba6b9cef15cd5d9bf9a0c700cef26
commit: 97e65926410ba6b9cef15cd5d9bf9a0c700cef26
source_record: raw/frameworks/harbor-codebase--github-97e65926410b.md
generated: 2026-08-04
---

# Harbor Codebase Important Files

## Required Code Evidence

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/harbor/index.md` | task-model | `src/harbor/models/task/task.py` | `Task` | 35 | — |
| `docs/frameworks/harbor/index.md` | task-config | `src/harbor/models/task/config.py` | `TaskConfig` | 795 | — |
| `docs/frameworks/harbor/index.md` | job-config | `src/harbor/models/job/config.py` | `JobConfig` | 344 | — |
| `docs/frameworks/harbor/index.md` | dataset-sources | `src/harbor/models/job/config.py` | `DatasetConfig.get_task_configs` | 154 | 216 |
| `docs/frameworks/harbor/index.md` | trial-config | `src/harbor/models/trial/config.py` | `TrialConfig` | 440 | — |
| `docs/frameworks/harbor/index.md` | job-entry | `src/harbor/job.py` | `Job.create` | 137 | 165 |
| `docs/frameworks/harbor/index.md` | job-run | `src/harbor/job.py` | `Job.run` | 961 | 1103 |
| `docs/frameworks/harbor/index.md` | job-plan | `src/harbor/job_plan.py` | `JobPlan.from_config` | 41 | 76 |
| `docs/frameworks/harbor/index.md` | trial-build | `src/harbor/job_plan.py` | `JobPlan.build_trial_configs` | 111 | 140 |
| `docs/frameworks/harbor/index.md` | trial-lifecycle | `src/harbor/trial/trial.py` | `Trial.run` | 370 | 403 |
| `docs/frameworks/harbor/index.md` | agent-phase | `src/harbor/trial/trial.py` | `Trial._run_agent_phase` | 445 | 488 |
| `docs/frameworks/harbor/index.md` | single-step | `src/harbor/trial/single_step.py` | `SingleStepTrial._run` | 37 | 60 |
| `docs/frameworks/harbor/index.md` | multi-step | `src/harbor/trial/multi_step.py` | `MultiStepTrial._run` | 57 | 83 |
| `docs/frameworks/harbor/index.md` | queue | `src/harbor/trial/queue.py` | `TrialQueue.submit_batch` | 253 | 259 |
| `docs/frameworks/harbor/index.md` | agent-contract | `src/harbor/agents/base.py` | `BaseAgent` | 20 | — |
| `docs/frameworks/harbor/index.md` | env-contract | `src/harbor/environments/base.py` | `BaseEnvironment` | 84 | — |
| `docs/frameworks/harbor/index.md` | env-exec | `src/harbor/environments/base.py` | `BaseEnvironment.exec` | 1128 | — |
| `docs/frameworks/harbor/index.md` | network-policy | `src/harbor/environments/base.py` | `BaseEnvironment.set_network_policy` | 838 | 851 |
| `docs/frameworks/harbor/index.md` | verifier | `src/harbor/verifier/verifier.py` | `Verifier.verify` | 138 | 237 |
| `docs/frameworks/harbor/index.md` | compiler | `src/harbor/compile/compiler.py` | `Compiler.compile` | 66 | 92 |
| `docs/frameworks/harbor/index.md` | executor | `src/harbor/exec/executor.py` | `Executor.execute` | 56 | 74 |
| `docs/frameworks/harbor/index.md` | task-client | `src/harbor/tasks/client.py` | `TaskClient.download_tasks` | 474 | 552 |

## Runtime Flow Evidence

1. Config resolution — `job-config`, `dataset-sources`, `trial-config`.
2. Plan building and task caching — `job-plan`, `trial-build`, `task-client`.
3. Job entry and dispatch — `job-entry`, `job-run`, `queue`.
4. Trial lifecycle — `trial-lifecycle`, `agent-phase`, `single-step`, `multi-step`.
5. Execution primitives — `agent-contract`, `env-contract`, `env-exec`, `network-policy`.
6. Verification and evidence — `verifier`.
7. Derived workflows — `compiler`, `executor`.

## Evidence Map

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

## Reproduction Commands

Read-only commands used to scope and verify this reading:

```bash
# Enumerate runtime packages and providers
find src/harbor -maxdepth 2 -type d | sort
for d in tasks trial agents environments verifier models compile mappers exec; do
  ls src/harbor/$d 2>/dev/null
done

# Locate core class and method definitions (line numbers cited above)
grep -nE '^(class |    async def |    def )' src/harbor/job.py src/harbor/trial/trial.py
grep -nE '^class ' src/harbor/models/task/config.py src/harbor/models/job/config.py src/harbor/models/trial/config.py
```

Runtime behavior was not executed; all claims are static code reading.
