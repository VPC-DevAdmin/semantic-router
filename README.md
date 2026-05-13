# semantic-router benchmark harness

A benchmark harness that quantifies the value of the
[vLLM Semantic Router](https://github.com/vllm-project/semantic-router) by
sending a curated query set through it in two passes:

1. **Routing accuracy** — does the router pick a model at or above the
   expected minimum tier for each query?
2. **Response quality** — does the response from the router-selected model
   meet the bar set by a gold reference answer?

A SQLite database is the canonical run store. Every step is resumable.
All model endpoints (5 tier backends, judge) are OpenAI-compatible — change
backends with a YAML edit, no code changes.

The full project scope, design rationale, and current status live in
[**PLAN.md**](PLAN.md). What follows is the operational quickstart.

## Quickstart

```sh
git clone git@github.com:VPC-DevAdmin/semantic-router.git
cd semantic-router

make setup        # venv + Python deps + DB schema + installs `vllm-sr` if missing
make load         # 110 curated queries (with embedded gold answers) → DB
make route        # boot router, send each query, capture routing decisions
make answer       # boot router, send each query for full LLM responses
make judge        # LLM-as-judge scoring of responses vs. gold
make report       # aggregate stats; pass JSON=path or CSV=path to export
```

Other targets:

```sh
make review REVIEWER=alice [SAMPLE=20]   # human scoring TUI
make resume                              # pick up pending/error rows from latest run
make clean-results                       # wipe runs/results/scores (preserves queries)
make router-smoke PROMPT='What is 7+5?'  # send one query, print routing decision
make router-stop                         # tear down the vllm-sr Docker stack
make help                                # full target list
```

## Repository layout

```
config/           # everything the operator tunes — YAML configs
    models.yaml       # 5 tier registry; maps router-emitted model name → tier number
    router.yaml       # process-management config for the `vllm-sr` subprocess
    vllm-sr.yaml      # the router's INTERNAL config (passed via --config flag)
    judge.yaml        # LLM-as-judge endpoint
    scoring.yaml      # rubric + 1-5 scale

data/
    queries.json      # 110 queries with `expected_answer` (the gold standard)

src/benchmark/    # the harness itself
tests/            # 69 tests covering everything except the live router
PLAN.md           # full project scope
```

## What's where in the workflow

```
queries.json ──[make load]──> SQLite ──[make route]──> Pass 1 results
                                ▲                          │
                                │                          ▼
                                └──[make answer]──> Pass 2 results
                                                           │
                                                           ▼
                                                   [make judge] / [make review]
                                                           │
                                                           ▼
                                                   scores ──[make report]──> stdout/JSON/CSV
```

## Status

**M0–M6 are shipped to main.** The harness is feature-complete; we are
currently dogfooding against a real `vllm-sr` install. See
[PLAN.md § Current State](PLAN.md#10-current-state) for the live status of
the integration work and [§ Roadmap](PLAN.md#11-roadmap) for what's next.

## Tests

```sh
make test       # 69 tests, ~2 seconds
make lint       # ruff
```

## Acknowledgements

This harness is one project; the thing it measures is
[vLLM Semantic Router](https://github.com/vllm-project/semantic-router),
an Apache-2.0 routing system from the vLLM project. The benchmark exists
to validate their claim that intelligent routing produces matching answer
quality at a fraction of the cost.
