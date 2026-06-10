# Semantic routing benchmark

A reproducible harness that drives the
[vLLM Semantic Router](https://github.com/vllm-project/semantic-router)
through a curated 110-query set and emits a single
`data/routed_queries_with_answers.json` artifact. That export is
consumed downstream by a separate replay UI and an independent judging
workflow — this repo does no judging and no live inference at demo time.

The most recent export is **checked in** under `data/` — clone the
repo and you can inspect a complete result set without running anything.
Re-running the pipeline (below) regenerates it.

Full design and rationale: [**PLAN.md**](PLAN.md).
Working with this repo from a Claude session: [**CLAUDE.md**](CLAUDE.md).

## Just want to see it?

```sh
git clone git@github.com:VPC-DevAdmin/semantic-router.git
cd semantic-router
make demo        # builds nothing heavy — serves the committed dataset + opens your browser
```

`make demo` is a single command that works on a bare clone with **no
`make setup`** — it needs only `python3`. It replays the committed,
already-judged benchmark dataset (real routing, real answers, real
verdicts). Everything below is for *reproducing* that dataset yourself.

## Quickstart (reproduce the dataset)

```sh
git clone git@github.com:VPC-DevAdmin/semantic-router.git
cd semantic-router
cp .env.example .env       # fill in API keys + per-tier model env

make setup     # venv, Python deps, DB schema, installs vllm-sr if missing
make load      # data/queries.json → data/router_benchmark.db
make route     # routing pass (via local OAI mock — no per-query token cost)
make answers   # for each routed query × each model in the picked tier, get a real answer
make evaluate  # LLM-judge routed vs gold (batched, per-row resumable)
make export    # write data/routed_queries_with_answers.json + data/evaluations.json (if judged)
make demo      # serve the cost-routing replay + open the browser — SINGLE COMMAND,
               # works on a bare clone with no `make setup` (just needs python3)
```

That's the whole pipeline. Pass-1 (`route`) and pass-2 (`answers`) are
each **per-row resumable**: kill mid-run, re-run, and only `pending`
or `error` rows are re-processed. Errors in `answers` don't fail the
pass — they stay as `status='error'` and retry on the next invocation.

## Contract gateway (use the router from an agent orchestrator)

`make gateway` starts an **OpenAI-compatible front door** that adapts this
router to the role-based contract an agent orchestrator expects, without
specializing the router or touching the standalone demo:

```sh
make gateway                                  # standalone (zero real backends)
make gateway ROUTER_URL=http://localhost:8801 # classify the worker via real vllm-sr
```

On top of plain `/v1/chat/completions` it adds: **role names** in the `model`
field (pinned vs. semantically-routed, defined in `config/gateway_roles.yaml`),
a **`metadata.min_tier`** floor honored as `served = max(classified, min)`, the
**`x-llm-model-served` / `x-llm-route-decision` / `x-llm-cost-usd`** response
headers (route-decision exposes `classified` vs `served` vs `min` so an
escalation reads as "classifier said L2, floor forced L3"), and **strict
structured output** for any schema. It's **additive** — `make demo`, `make
route`, and the harness are unchanged, so the standalone router demo is intact.
The gateway is role-agnostic: roles live in config, so any client defines its
own. See [`tools/router_gateway.py`](tools/router_gateway.py).

## Other targets

```sh
make resume                              # pick up pending/error rows from latest run
make clean-results                       # wipe routing/answer data (preserves queries + gold)
make router-smoke PROMPT='What is 7+5?'  # send one query through the router, print decision
make router-stop                         # tear down the vllm-sr Docker stack
make mock-bg / mock-stop                 # start/stop the local OAI mock (port 18811)
make start_LLM / stop_LLM                # launch/teardown local-CPU tier backends
make test / make lint                    # unit tests (~2s) + ruff
make help                                # full target list with flag docs
```

## Repository layout

```
config/                            # everything the operator tunes (YAML)
    tiers/                         # tier metadata (level, label, specializations,
                                   # router_alias, timeout_s)
    router-exemplars.yaml          # contrastive-embedding training data — the
                                   # source of truth for the router's decision logic
    router-backends.yaml           # router-side per-tier endpoints
    router-config.yaml             # GENERATED from the two above by `make load` /
                                   # `make route`; passed to `vllm-sr --config`
    router.yaml                    # process-management config for the vllm-sr
                                   # subprocess (ports, log path, etc.)
    local_models.yaml              # per-vendor launch recipes for tier 1/2 local vLLM

data/
    queries.json                            # 110 curated queries with per-provider
                                            # gold answers (committed)
    routed_queries_with_answers.json        # latest routed export — COMMITTED so a fresh
                                            # clone has results visible without re-running
    evaluations.json                        # latest judge verdicts — COMMITTED (when
                                            # `make evaluate` has been run for the
                                            # active export)
    external_answers/                       # externally-generated answers loaded via
                                            # `make import-answers` (committed)
    router_benchmark.db                     # SQLite run state — generated (gitignored)
    .router-config-hash                     # cached config hash — generated (gitignored)

demo/                              # browser replay demo (served via `make demo`)
    index.html / demo.css / demo.js   # data-driven front-end (5 tiers, model pickers)
    pricing.json                      # per-1M-token rates (committed, hand-edited)
    data/demo_data.json               # generated by `make demo-data` (gitignored)

src/benchmark/                     # the harness itself (Python package: `benchmark`)
tests/                             # unit tests — 193 of them, ~2 seconds total
tools/oai_mock.py                  # stdlib OAI mock — used by `make route`
tools/build_demo_data.py           # builds demo/data/demo_data.json from the exports
.env.example                       # template for per-slot env vars (the actual .env
                                   # is gitignored — your secrets live there)
PLAN.md                            # full project scope and rationale
CLAUDE.md                          # primer for Claude sessions working on this repo
```

## Pipeline

```
queries.json ──[make load]──> SQLite ──┬─[make route]──> tier routing decisions
                                       │
                                       └─[make answers]──> per-(query × model) responses
                                                                      │
                                                                      ▼
                                                               [make export]
                                                                      │
                                                                      ▼
                                                  data/routed_queries_with_answers.json
                                                                      │
                                                                      ▼
                                                          external judge + replay UI
```

`make route` always uses the local OAI mock — the routing decision is
all pass-1 needs, and the mock ACKs cheaply with no token cost. The
real per-model calls happen in `make answers`, which bypasses the
router and talks to each tier's models directly.

## Status

Feature-complete and dogfooded end-to-end. `make load`, `make route`,
`make answers`, and `make export` produce a complete demo artifact
against the real `vllm-sr` install plus a mix of vLLM, OpenAI,
Anthropic, and Google upstream backends.

## License

Apache-2.0 — matches upstream [`vllm-project/semantic-router`](https://github.com/vllm-project/semantic-router).
See [LICENSE](LICENSE).
