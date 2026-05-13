# Semantic Router Benchmark Harness — Project Plan

> **Status as of latest commit:** the harness is feature-complete (M0–M6) and
> currently being dogfooded against a real `vllm-sr` install. We're working
> through last-mile integration issues (Envoy route generation) before
> swapping in real LLM backends. See [§ Current State](#current-state) for
> exactly where we are.

---

## 1. Purpose

The [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) is
an intelligent traffic-routing layer for a "mixture-of-models" deployment: it
inspects an incoming query and dispatches it to the appropriately sized model
(tiny CPU model for trivial questions, frontier LLM for synthesis tasks).

The promise: **same answer quality at a fraction of the cost**, because most
queries don't need a frontier model.

This project quantifies that promise on a curated query set, in two passes:

1. **Routing accuracy (Pass 1 — `make route`)**
   Does the router pick a model at or above the expected minimum tier (and
   matching specialization) for each query? Cheap to run; per-query
   `max_tokens=1` because we only care about the routing decision in the
   response headers.

2. **Response quality (Pass 2 — `make answer`)**
   Does the response from the router-selected model meet the bar set by a
   gold reference produced by a frontier model? Scored either by an
   LLM-as-judge (`make judge`) or a human reviewer (`make review`).

A SQL database is the canonical record. Each row corresponds to one query
× one run. Runs are resumable — kill and restart, the next invocation picks
up `status IN ('pending', 'error')` rows.

## 2. What success looks like

A complete benchmark run produces a report (Rich table to stdout + optional
JSON/CSV export) showing:

- **Pass 1 topline**: % of queries where `router_selected_tier ≥ expected_min_tier`,
  broken down by tier and specialization. Surfaces the cases where the router
  under-routed (assigned a too-small model to a hard query) or over-routed
  (used the frontier for trivia).
- **Pass 2 score histogram**: distribution of 1–5 judge scores against gold,
  per scorer (one entry per LLM judge model + one entry per human reviewer).
  Means and per-spec breakdowns to find weak categories.
- **Cost/quality trade-off** *(future)*: token counts and latency per tier,
  multiplied by per-million prices in `models.yaml`, to express the value
  in dollars saved per quality point lost.

## 3. Tech stack

- **Python 3.11+** (we use 3.12 on the server, 3.14 on dev macOS)
- **SQLite** with WAL — canonical run store; comfortably handles 10k+ rows
- **SQLAlchemy 2.x** for schema + sessions
- **Pydantic v2** for YAML/JSON config validation
- **Typer** + **Rich** for the CLI and human review TUI
- **httpx** (async) for OpenAI-compatible HTTP — every backend (tier models,
  the gold reference source, the judge, the router itself) speaks OAI
- **pytest** + **ruff** for tests and lint

## 4. Repository layout (current)

```
semantic-router/
├── PLAN.md                       # this document
├── README.md                     # quickstart
├── Makefile                      # 7-target user-facing workflow
├── pyproject.toml
├── config/
│   ├── models.yaml               # 5 tier registry — maps model name → numeric tier
│   ├── router.yaml               # process-management for the `vllm-sr` subprocess
│   ├── vllm-sr.yaml              # the router's INTERNAL config (passed via --config)
│   ├── judge.yaml                # judge model endpoint
│   └── scoring.yaml              # rubric + 1-5 scale
├── data/
│   └── queries.json              # 110 queries (tier 1-5) with embedded gold answers
├── src/benchmark/
│   ├── cli.py                    # Typer entrypoint
│   ├── config.py                 # pydantic-validated config loaders
│   ├── db.py                     # SQLAlchemy schema + session_scope
│   ├── load.py                   # queries.json → DB (gold from `expected_answer`)
│   ├── tiers.py                  # async OAI-compatible client
│   ├── router_proc.py            # `vllm-sr serve` lifecycle (launcher pattern)
│   ├── router_client.py          # talks to Envoy; extracts x-vsr-* headers
│   ├── runs.py                   # run lifecycle + pending-row seeding
│   ├── pass1.py                  # routing accuracy
│   ├── pass2.py                  # response generation
│   ├── judge.py                  # LLM-as-judge scoring
│   ├── review.py                 # human scoring TUI
│   └── report.py                 # aggregate stats + JSON/CSV export
└── tests/                        # 69 tests covering everything except live router
```

## 5. The 7-target workflow

| Target | What it does | Idempotent? | LLM cost |
|---|---|---|---|
| `make setup` | venv + deps + DB init + installs `vllm-sr` if missing | yes | $0 |
| `make load` | `data/queries.json` → DB (queries + gold) | yes | $0 |
| `make route` | Pass 1: boots router, sends queries, captures routing decisions | resumable | ~$0 (max_tokens=1) |
| `make answer` | Pass 2: sends queries for full LLM responses | resumable | varies by tier |
| `make judge` | LLM-as-judge scoring vs. gold | resumable | judge tokens only |
| `make review` | Human scoring TUI (`REVIEWER=alice [SAMPLE=N]`) | resumable per-reviewer | $0 |
| `make report` | Aggregate stats + `JSON=path` / `CSV=path` export | safe | $0 |

Supporting targets: `resume`, `clean-results`, `router-smoke`, `router-stop`,
`test`, `fmt`, `lint`.

## 6. Data model

### `data/queries.json` (110 entries today, schema is open-ended)

```json
{
  "id": "q00001",
  "prompt": "What is the capital of France?",
  "expected_answer": "Paris.",
  "expected_min_tier": 1,
  "specializations": ["general"],
  "domain_tags": ["geography"],
  "notes": "trivial factual lookup"
}
```

`expected_answer` is the upstream Opus-level gold — there is no separate
`make gold` step. Tier distribution:
`{tier 1: 25, tier 2: 36, tier 3: 32, tier 4: 7, tier 5: 10}`.
Specializations: `general`, `coding`, `creative_writing`. The whitelist also
permits `math`, `reasoning`, `vision`, `tts` for future extension.

### SQLite schema (canonical run state)

```
queries          (query_id PK, prompt, prompt_hash, expected_min_tier,
                  specializations, domain_tags, gold_answer, gold_model, ...)
runs             (run_id PK, started_at, finished_at, router_config_hash,
                  models_config_hash, status, notes)
pass1_results    (run_id, query_id PK, router_selected_model,
                  router_selected_tier, meets_minimum_tier,
                  matches_specialization, raw_routing_metadata, status, ...)
pass2_results    (run_id, query_id PK, router_selected_model, response_text,
                  prompt_tokens, completion_tokens, latency_ms, status, ...)
scores           (run_id, query_id, scorer, reviewer_id PK, score,
                  rubric_version, rationale, scored_at)
```

**Resume rule:** each per-pass row transitions `pending → success | error`.
Workers select rows where `status IN ('pending', 'error')` for the active
run. Per-row session commits make killing the process mid-run safe.

## 7. Router integration model

`vllm-sr serve` is not a daemon — it's a **launcher** that brings up a
Docker stack (router + envoy + dashboard + simulator + datastores +
observability) and exits cleanly. The actual router lives in those
background containers, managed by the host `vllm-sr` CLI.

The harness handles this in [`router_proc.py`](src/benchmark/router_proc.py):

1. Run `vllm-sr serve --config config/vllm-sr.yaml --minimal` synchronously.
2. Wait for the launcher subprocess to exit. Exit 0 = launch succeeded.
3. Poll `/ready` on the router's apiserver (`:8080`) until it returns 200.
4. Hand control to the benchmark passes.
5. **Leave the stack running on exit** (cold-start is slow, repeat runs are
   fast). User controls the long-lived lifecycle via `make router-stop`.

`config/vllm-sr.yaml` is the router's internal config — 5 tier models with
backends, plus keyword-signal routing decisions. It's the file that
`--config` points at. All tier backends currently point at the bundled
simulator (`host.docker.internal:8810`) so we can validate routing without
any real LLM cost.

The routing decision lands in three response headers added by the router on
2xx-non-cached responses:

- `x-vsr-selected-model` → e.g. `tier3`
- `x-vsr-selected-category` → e.g. `math`
- `x-vsr-selected-reasoning` → `on` | `off`

`config/models.yaml` maps each `x-vsr-selected-model` value back to a numeric
tier so we can compare against `expected_min_tier`. The shipped-configs test
asserts every model name in `vllm-sr.yaml` has a matching `model_id` entry
in `models.yaml` — drift between the two is the most common config bug.

## 8. Backend strategy (current and planned)

The harness is intentionally agnostic about what's behind the router. Three
phases of backend deployment, ordered by maturity:

**Phase A — Simulator only (where we are now).**
All 5 tiers point at the bundled `vllm-sr-sim` mock. Validates the routing
pipeline and our reporting end-to-end without spending a cent on tokens.
Pass 2 responses are stub text; not meaningful for quality scoring, but
fine for testing the plumbing.

**Phase B — Real small + simulator.**
Two CPU models (tier 1 + tier 2) on the user's existing CPU server, tier 3–5
still on the simulator. First real signal on small-model performance.

**Phase C — Full ladder.**
Tier 1–2 on CPU, tier 3 on GPU server (once acquired), tier 4 = Anthropic
Sonnet, tier 5 = Anthropic Opus, via their OpenAI-compatible endpoints.
Real Pass 2 responses; real judge scoring; real cost numbers.

Swapping phases is a config-only change: update `backend_refs.endpoint` in
`config/vllm-sr.yaml` per tier, update the matching `endpoint` in
`config/models.yaml` for direct-tier calls, no code changes.

## 9. Implementation milestones (history)

All shipped to main:

| M | Title | Notes |
|---|---|---|
| M0 | Plan + skeleton | PLAN.md, README, Makefile stubs |
| M1 | DB schema, config loaders, idempotent loader | SQLAlchemy + pydantic |
| M2 | Tier client + gold (now folded into `make load`) | async OAI client |
| M3 | Router subprocess + client + x-vsr-* extraction | initially mis-modeled the router as a daemon; corrected during dogfooding |
| M4 | Pass 1 + Pass 2 + run lifecycle + resume | per-row commits, bounded concurrency |
| M5 | LLM-as-judge + human review TUI | injectable I/O for TTY-less testing |
| M6 | Aggregate report + JSON/CSV export | per-spec breakdown |
| **Simplification pass** | queries.json + 7-target Makefile | dropped YAML, `seed`, `gold`, `validate-config` targets |
| **Dogfood fixes** | RouterProcess launcher pattern + `make setup` installs router + shipped vllm-sr.yaml | in response to real-world failures |

## 10. Current state

Live dogfooding against a real `vllm-sr` install on a user's Linux server:

- ✅ `make setup` installs `vllm-sr`, sets up venv, inits DB
- ✅ `make load` reads 110 queries with embedded gold
- ✅ `make route` launches `vllm-sr serve` with our config, waits for `/ready`,
  successfully gets 200 from the apiserver
- ❌ **Active bug:** all 110 chat-completion requests to Envoy at
  `http://127.0.0.1:8899/v1/chat/completions` return HTTP 404. Envoy is
  listening but doesn't know how to route the path. Almost certainly a
  missing field in our `config/vllm-sr.yaml` listener spec that drives
  Envoy route generation. Diagnostics requested:
  - `vllm-sr chat "hello"` — does upstream's own client get through?
  - `cat .vllm-sr/envoy.yaml` — what was actually generated
  - `vllm-sr logs envoy | tail -40` — Envoy's verdict on the request

## 11. Roadmap

In likely order:

1. **Unblock `make route`.** Resolve the 404 (see § Current State).
2. **Read end-to-end Pass 1 report.** First real routing-accuracy numbers
   against simulator backends; expect 40–60% `meets_min_tier` with the
   current keyword-only signals.
3. **Tune signals.** Inspect misroutes, refine `coding_keywords` /
   `complex_reasoning_keywords` / `creative_writing_keywords`, or enable the
   router's built-in MMLU domain classifier for smarter signals.
4. **Phase B: real small CPU models behind tier 1+2.** Validate real
   inference end-to-end on the cheap end of the ladder.
5. **Run Pass 2.** Now that backends are real, `make answer` produces real
   responses; `make judge` scores them.
6. **Phase C: Anthropic API for tier 4+5.** Spend tokens on the cases the
   router actually routes there.
7. **First public report.** `make report` numbers + `make report JSON=...`
   for ad-hoc analysis. Decide whether to scale the query set further
   (10k+ was an early aspiration; the 110-query set may be enough for the
   initial demonstration).

## 12. Known open design questions

- **Routing intelligence.** Current `vllm-sr.yaml` uses only BM25 keyword
  signals — fast, deterministic, but coarse. The router supports embedding
  signals, domain classifiers, prompt-guard, semantic-cache. Adding any of
  these makes Pass 1 results more interesting but requires the corresponding
  classifier/embedding model files (which the upstream config example
  references via `global.model_catalog`).
- **Combined Pass 1 + Pass 2.** PLAN's original M4 mentioned a `--combined`
  mode that uses one router call per query for both passes. Not implemented.
  Easy to add when token costs matter.
- **Multimodal.** Vision and TTS are reserved in the spec whitelist but no
  queries today exercise them. `data/queries.json` schema supports
  `attachments`; the router config doesn't yet wire them through.
- **Concurrency tuning.** Default is 8 concurrent calls per pass. May be too
  aggressive for real backends (rate limits) or too low for the simulator.

## 13. Testing strategy

- **69 unit tests** cover the schema, loaders, run lifecycle, pass logic
  with mocked router responses, judge verdict parsing, human-review state
  machine, report aggregation, and shipped config files.
- **No live-router tests in CI.** `RouterProcess` and `RouterClient` are
  exercised with `httpx.MockTransport` and `subprocess.Popen` mocks. The
  one thing not testable in CI is whether the actual `vllm-sr` binary
  behaves as documented — that's what dogfooding is for, and is what
  surfaced the launcher-vs-daemon bug we just fixed.
- **`tests/test_shipped_configs.py`** asserts every YAML/JSON file under
  `config/` and `data/` actually parses with the real loaders, plus that
  `vllm-sr.yaml`'s model names line up with `models.yaml`'s `model_id`
  values. This catches the failure modes that have actually bitten us.
