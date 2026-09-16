[← All posts](https://deepswe.datacurve.ai/blog)

DeepSWE v1.1 keeps the same long-horizon engineering tasks as [v1](https://deepswe.datacurve.ai/blog/deepswe), but updates how agents are executed and scored by grading their committed code in a clean, isolated environment, making results easier to reproduce, audit, and analyze. We also fixed dependency drift and removed flaky tests on some tasks.

With the updated setup, now including Claude Fable 5 and Kimi K2.7 Code, aggregate pass rates and model ordering remain close to v1.

v1.1v1

CostOutput tokensAgent steps

113 tasks· updated July 1, 2026Models(10/10)

DeepSWE score$0$5.00$10$15$20$250%10%20%30%40%50%60%70%80%Avg cost per taskmost efficient ↗claude-fable-5 \[high\]Defaultgpt-5.5 \[medium\]Defaultclaude-sonnet-5 \[max\]Defaultclaude-opus-4.8 \[high\]Defaultgpt-5.4 \[xhigh\]glm-5.2 \[max\]Defaultgemini-3.5-flash \[medium\]kimi-k2.7-codeclaude-sonnet-4.6 \[high\]gemini-3.1-pro \[high\]

v1.1v1

BestAll effort levels

Models(10/10)

Model

Pass@1

Avg cost

Out tok

Steps

![](https://deepswe.datacurve.ai/icons/Anthropic.svg)claude-fable-5\[max\]

70%±4%

Avg cost $21.63Out tok 119kSteps 88

70%±4%

$21.63

119k

88

![](https://deepswe.datacurve.ai/icons/OpenAI.svg)gpt-5.5\[xhigh\]

67%±6%

Avg cost $7.23Out tok 46kSteps 82

67%±6%

$7.23

46k

82

![](https://deepswe.datacurve.ai/icons/Anthropic.svg)claude-opus-4.8\[max\]

59%±2%

Avg cost $13.22Out tok 135kSteps 120

59%±2%

$13.22

135k

120

![](https://deepswe.datacurve.ai/icons/Anthropic.svg)claude-sonnet-5\[max\]

54%±4%

Avg cost $26.40Out tok 214kSteps 268

54%±4%

$26.40

214k

268

![](https://deepswe.datacurve.ai/icons/OpenAI.svg)gpt-5.4\[xhigh\]

52%±2%

Avg cost $5.65Out tok 71kSteps 70

52%±2%

$5.65

71k

70

![](https://deepswe.datacurve.ai/icons/ZAI.png)glm-5.2\[max\]

44%±2%

Avg cost $3.92Out tok 78kSteps 129

44%±2%

$3.92

78k

129

![](https://deepswe.datacurve.ai/icons/Google.svg)gemini-3.5-flash\[medium\]

37%±2%

Avg cost $7.34Out tok 276kSteps 86

37%±2%

$7.34

276k

86

![](https://deepswe.datacurve.ai/icons/Moonshot.png)kimi-k2.7-code

31%±1%

Avg cost $2.82Out tok 59kSteps 149

31%±1%

$2.82

59k

149

![](https://deepswe.datacurve.ai/icons/Anthropic.svg)claude-sonnet-4.6\[high\]

30%±4%

Avg cost $5.52Out tok 76kSteps 134

30%±4%

$5.52

76k

134

![](https://deepswe.datacurve.ai/icons/Google.svg)gemini-3.1-pro\[high\]

12%±2%

Avg cost $9.48Out tok 196kSteps 81

12%±2%

$9.48

196k

81

0%20%40%60%80%

Note: 73 of Claude Fable 5's2,260 trials did not complete due to [access being suspended by a US government directive](https://www.anthropic.com/news/fable-mythos-access) partway through our sweep. Pass rates are computed over the completed trials.

Wall-clock time is no longer reported as it is highly dependent on external variables like host machine performance and provider load, making it an inconsistent metric.

## Explore

View the benchmark on GitHub, browse every rollout behind the numbers above, or run your own agent against the benchmark.

[Browse trajectories](https://deepswe.datacurve.ai/data/v1.1/trials) [Run DeepSWE](https://deepswe.datacurve.ai/run) [GitHub](https://github.com/datacurve-ai/deep-swe)

## What changed

Agent container

Checks out `main` (with future git history deleted), commits to feature branch.

committed diff only

Verifier container

Fresh container. Applies the diff, executes tests, produce CTRF report.

1. **Isolated Verification:** The agent commits its proposed changes, and we extract the git patch to evaluate in an isolated container, separate from where the agent worked. This follows the same approach as SWE-bench, and keeps grading independent of the agent's runtime environment for reproducible results.
2. **Structured test reports:** Tests now emit a CTRF report, recording each test that defines a task by name and status. This gives us a per-test view of what passed and failed, useful for analyzing results and spotting partial progress on a task.
3. **Natural Git Environment:** Rather than operating in detached HEAD mode, we set the `main` branch to the task's starting commit and ensure no future commits are visible. This enables agents to work more naturally from the`main` branch, formulate feature branches, and explicitly commit its changes as it would in normal development.

To elaborate on the git environment: one concern with our previous environment construction method was that if an implementation similar to a task had been merged upstream, the agent could potentially cheat by finding it through `git log`. We conducted a sweep of the tasks' upstream repos to check whether any had implementations similar to our tasks as of June 5th. We found no such instances, meaning results from v1.0 remain free of this form of cheating.

Together, these changes make tasks harder to game. Because we grade only the committed patch in a separate container, some easier shortcuts no longer work: an agent can't monkey-patch the test framework, and because the CTRF report records each task-defining test by name, dropping tests or forcing an early exit shows up as missing or failed results rather than a pass.

## Impact on Results

The chart below compares each model's pass rate under v1 and v1.1. Scores stay close: the ordering at the top is unchanged, and most configurations land within a few points of their v1 result.

gpt-5.5\[xhigh\]

70%→67%-3.0%

gpt-5.5\[high\]

62%→64%+2.4%

claude-opus-4.8\[max\]

58%→59%+0.8%

claude-opus-4.8\[xhigh\]

58%→54%-3.4%

gpt-5.5\[medium\]

48%→54%+6.0%

claude-opus-4.8\[high\]

51%→52%+1.1%

gpt-5.4\[xhigh\]

56%→52%-3.8%

claude-opus-4.8\[medium\]

47%→49%+1.3%

gemini-3.5-flash\[medium\]

28%→37%+9.1%

claude-sonnet-4.6\[high\]

32%→30%-1.8%

0%20%40%60%80%

v1v1.1

The 10 configurations run in both versions. Hollow dot is v1, filled dot is v1.1; the change in points is at right.

Taskv1v1.1Difference

abs-module-cache-flags88%88%+0.0%

abs-stepped-slices88%83%-5.0%

actionlint-action-pinning-lint88%90%+2.5%

adaptix-name-mapping-aliases68%70%+2.5%

aiomonitor-task-snapshots-diff84%79%-4.2%

anko-default-function-arguments78%75%-2.5%

anko-typed-variable-bindings48%50%+2.5%

arcane-drift-detection-baselines61%60%-1.2%

arktype-json-schema-refs-dependencies28%38%+10.0%

awilix-async-container-initialization28%30%+2.5%

bandit-incremental-cache-control48%57%+10.0%

bandit-interprocedural-taint-checks57%48%-10.0%

bandit-structured-nosec-directives3%0%-2.5%

boa-hierarchical-evaluation-cancellation51%50%-1.0%

cattrs-partial-structuring-recovery55%65%+10.0%

clack-async-autocomplete-options21%28%+7.0%

claude-code-by-agents-recursive-delegation10%50%+39.8%

cliffy-config-file-parsing35%28%-7.5%

csstree-shorthand-expansion-compression33%30%-2.5%

dasel-html-document-format43%55%+12.5%

dateutil-rfc5545-timezone-interop25%40%+15.0%

drizzle-orm-window-function-builders88%85%-2.8%

dynamodb-toolbox-conditional-attribute-requirements83%78%-5.0%

dynamodb-toolbox-lazy-recursive-schemas55%80%+24.9%

effect-sse-httpapi-streaming18%23%+4.1%

eicrud-keyset-pagination-cursor38%23%-15.0%

etree-xml-diff-patch68%70%+2.5%

expr-try-catch-errors10%20%+9.8%

fastapi-deprecation-response-headers59%55%-4.2%

fastapi-implicit-head-options51%53%+1.5%

fd-deterministic-multi-key-sorting40%48%+7.5%

geo-shapeindex-serialization53%57%+5.0%

go-critic-doc-link-checker55%50%-5.1%

go-genai-streamed-function-args83%73%-10.0%

go-git-worktree-merge-conflicts65%53%-12.5%

goreleaser-retry-publish-auditing53%63%+9.7%

gql-incremental-graphql-delivery5%3%-2.5%

happy-dom-abort-pending-body-reads100%98%-2.5%

happy-dom-deterministic-intersectionobserver18%13%-5.0%

helm-array-merge-strategies18%38%+20.0%

helm-unified-manifest-stream80%80%+0.0%

httpx-deterministic-cookie-store93%88%-5.0%

httpx-multipart-response-parsing67%55%-12.3%

httpx-streaming-json-iteration31%30%-0.6%

igel-persist-feature-schema35%48%+12.8%

ink-grid-box-layout18%15%-2.5%

ipython-session-bundle-replay50%45%-5.0%

katex-multicolumn-array-spans20%20%-0.4%

kcp-go-multiplexed-kcp-streams51%59%+8.4%

kea-atomic-signal-selectors58%49%-9.7%

kgateway-consistent-hash-policy70%57%-12.5%

kombu-single-active-consumer-priority65%63%-2.5%

kombu-virtual-queue-dead-lettering13%20%+7.5%

koota-composite-trait-aspects33%38%+5.0%

koota-deferred-mutation-buffer27%23%-4.0%

koota-entity-snapshot-rollback90%98%+7.5%

koota-pair-relation-tracking18%15%-2.5%

koota-query-predicates20%20%+0.0%

kysely-window-grouping-helpers86%80%-5.7%

langchain-request-coalescing14%28%+13.2%

mashumaro-flattened-dataclass-fields23%25%+2.5%

meriyah-explicit-resource-declarations38%35%-2.5%

mnamer-daemon-watch-lifecycle73%60%-12.5%

mobly-grouped-test-barriers65%63%-2.5%

narwhals-rolling-window-suite30%95%+65.0%

numba-stencil-boundary-modes77%75%-1.9%

obsidian-linter-auto-table-of-contents13%0%-12.5%

obsidian-linter-link-format-conversion23%25%+2.5%

obsidian-linter-scoped-ignore-markers85%75%-10.0%

ofetch-per-origin-circuit-breaker95%82%-12.9%

onedump-dump-encryption-pipeline43%40%-2.5%

opa-rego-rule-profiling53%48%-5.3%

opa-template-string-reconstruction63%68%+5.0%

optique-conditional-option-dependencies25%33%+7.5%

oxvg-structural-selector-preservation20%10%-10.0%

participle-grammar-conflict-analysis43%29%-13.5%

pebble-durability-wait-apis63%63%+0.0%

pest-character-class-coalescing8%15%+7.5%

prometheus-transactional-reload-status3%13%+10.0%

prometheus-typed-label-sorting37%48%+10.8%

psd-tools-blend-range-api78%85%+7.5%

pwntools-tube-multiplexing40%48%+7.5%

python-statemachine-state-data-scoping31%25%-5.6%

query-persist-restored-query-state73%60%-12.5%

quill-shared-toolbar-focus0%18%+17.5%

returns-validated-error-accumulation90%90%+0.0%

scc-bounded-memory-spilling48%65%+17.5%

scriggo-method-declarations70%60%-10.0%

skrub-duration-encoding22%60%+37.6%

sql-formatter-bigquery-pipe-formatting88%88%+0.0%

sqlfmt-create-table-ddl-formatting43%23%-20.0%

sqlite-utils-safe-import-checkpoints63%57%-5.0%

superjson-error-stack-serialization31%29%-2.0%

task-task-graph-export75%83%+7.5%

tengo-callable-instance-isolation59%63%+3.3%

tengo-destructuring-bindings63%73%+9.2%

termenv-preserve-ansi-resets18%11%-6.7%

testem-bail-on-test-failure28%15%-12.5%

testem-per-launcher-reports75%95%+20.0%

textual-kitty-key-phases50%43%-7.5%

textual-richlog-follow-state63%60%-2.5%

tomlkit-toml-table-converters73%75%+1.5%

true-myth-iterable-collection-combinators98%98%+0.0%

ts-pattern-match-each73%70%-2.5%

updo-policy-alerting22%8%-14.8%

valibot-recursive-schema-composition67%68%+0.2%

vitest-duration-sharding73%90%+17.5%

vulture-persistent-analysis-cache78%18%-60.0%

wasmi-trap-coredumps70%83%+12.5%

wazero-multi-module-snapshots68%63%-4.3%

yaegi-go-embed-directives73%55%-17.5%

yjs-map-conflict-detection83%75%-7.5%

ytt-jsonpath-query-api88%89%+2.0%

Each task's pass rate in v1 and v1.1. Click a column to sort.

The 10 configurations run in both versions. Hollow dot is v1, filled dot is v1.1; the change in points is at right.

## Citation

Please cite this work as:

```
@misc{datacurve2026deepswev11,
  title  = {DeepSWE v1.1: a cleaner, more reproducible benchmark for frontier coding agents},
  author = {Wenqi Huang and Peter Jiang},
  year   = {2026},
  url    = {https://github.com/datacurve-ai/deep-swe},
}
```

![](https://deepswe.datacurve.ai/images/careers-cta-mountains.jpg)

## Sign up for leaderboard updates

New frontier models are added to the DeepSWE leaderboard as they're released.

Notify me