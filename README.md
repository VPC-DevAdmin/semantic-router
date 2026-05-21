# Semantic routing demo

A production-pass harness that drives the
[vLLM Semantic Router](https://github.com/vllm-project/semantic-router)
through a curated query set and emits a single
`data/evaluated_queries_with_answers.json` artifact. The export is then
replayed (and judged) downstream — this repo doesn't do any judging or
live inference at presentation time.

The full project scope, design rationale, and current status live in
[**PLAN.md**](PLAN.md). What follows is the operational quickstart.

## Quickstart

```sh
git clone git@github.com:VPC-DevAdmin/semantic-router.git
cd semantic-router

make setup         # venv + Python deps + DB schema + installs `vllm-sr` if missing
make load          # 110 curated queries (with embedded gold answers) → DB
make route         # for each query: capture which tier the router picks
make answers       # for each query × each tier: capture that tier's response
make export        # emit data/evaluated_queries_with_answers.json for downstream replay + judging
```

Other targets:

```sh
make resume                              # pick up pending/error rows from latest run
make clean-results                       # wipe runs/results (preserves queries + gold)
make router-smoke PROMPT='What is 7+5?'  # send one query, print routing decision
make router-stop                         # tear down the vllm-sr Docker stack
make help                                # full target list
```

## Repository layout

```
config/           # everything the operator tunes — YAML configs
    tiers/            # single source of truth for direct-call tier identity
                      # (router_alias, endpoint, backend.kind, ...)
    router-exemplars.yaml  # contrastive-embedding training data — the
                           # source of truth for the router's decision logic
    router-backends.yaml   # router-side per-tier endpoints (flat schema)
    router-config.yaml     # GENERATED (gitignored) from the two files above
                           # by `make load` / `make route`; the router's
                           # INTERNAL config passed via --config flag
    router.yaml       # process-management config for the `vllm-sr` subprocess

data/
    queries.json                            # 110 queries with embedded gold (per-provider)
    router_benchmark.db                     # generated SQLite store (gitignored)
    evaluated_queries_with_answers.json     # generated export artifact (gitignored)

src/benchmark/    # the harness itself
tests/            # unit tests covering everything except the live router
PLAN.md           # full project scope and current status
CLAUDE.md         # context primer for fresh Claude sessions
```

## What's where in the workflow

```
queries.json ──[make load]──> SQLite ──┬─[make route]──> tier routing decisions
                                       │
                                       └─[make answers]──> per-tier responses
                                                                      │
                                                                      ▼
                                                               [make export]
                                                                      │
                                                                      ▼
                                              data/evaluated_queries_with_answers.json
                                                                      │
                                                                      ▼
                                                          external judge + replay UI
```

## Status

The harness is feature-complete and being dogfooded against a real
`vllm-sr` install. `make setup`, `make load`, `make answers`, and
`make export` all work; `make route` reaches the router but its chat
completions return 404 from Envoy — an active integration issue. See
[PLAN.md § 13](PLAN.md#13-current-state-and-roadmap) for the live status.

## Tests

```sh
make test       # unit tests, ~2 seconds
make lint       # ruff
```

## Acknowledgements

This harness is one project; the thing it measures is
[vLLM Semantic Router](https://github.com/vllm-project/semantic-router),
an Apache-2.0 routing system from the vLLM project. The demo exists to
illustrate their claim that intelligent routing produces matching answer
quality at a fraction of the cost.
