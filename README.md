# Semantic routing demo

A production-pass harness that drives the
[vLLM Semantic Router](https://github.com/vllm-project/semantic-router)
through a curated query set and emits a single `demo.json` artifact. The
demo is then replayed (and judged) downstream — this repo doesn't do any
judging or live inference at presentation time.

The full project scope, design rationale, and current status live in
[**PLAN.md**](PLAN.md). What follows is the operational quickstart.

## Quickstart

```sh
git clone git@github.com:VPC-DevAdmin/semantic-router.git
cd semantic-router

make setup         # venv + Python deps + DB schema + installs `vllm-sr` if missing
make load          # 110 curated queries (with embedded gold answers) → DB
make route         # for each query: capture which tier the router picks
make answers       # [TODO] for each query × each tier: capture that tier's response
make export        # [TODO] emit demo.json for downstream replay + judging
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
    models.yaml       # tier registry; maps router-emitted model name → tier number
    router.yaml       # process-management config for the `vllm-sr` subprocess
    vllm-sr.yaml      # the router's INTERNAL config (passed via --config flag)

data/
    queries.json      # 110 queries with `expected_answer` (Opus-level gold)

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
                                                                  demo.json
                                                                      │
                                                                      ▼
                                                          external judge + replay UI
```

## Status

The harness is currently being dogfooded against a real `vllm-sr` install.
`make setup`, `make load`, and the launcher half of `make route` all work;
the chat-completion half of `make route` is blocked on an Envoy route
generation issue and `make answers` / `make export` are not yet
implemented. See [PLAN.md § 13](PLAN.md#13-current-state-and-roadmap) for
the live status.

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
