# semantic-router benchmark harness

A benchmark harness that quantifies the value of the
[vLLM Semantic Router](https://github.com/vllm-project/semantic-router) by routing a
curated query set through it in two passes:

1. **Routing accuracy** — does the router pick a model at or above the expected minimum tier?
2. **Response quality** — does the answer match a gold reference from a top-tier model?

SQLite is the canonical run store. Every step is resumable. All model endpoints (tiers,
gold, judge) are OpenAI-compatible.

See [PLAN.md](PLAN.md) for the full design.

## Status

Pre-implementation. This commit is the plan and skeleton; see milestones M1–M7 in
`PLAN.md`.

## Quickstart (target)

```sh
make setup     # venv, deps, init DB
make seed      # load curated queries
make gold      # generate gold answers
make run       # boot router subprocess + pass1 + pass2 + teardown
make judge     # LLM-as-judge scoring of pass 2 responses
make report    # aggregate stats
```
