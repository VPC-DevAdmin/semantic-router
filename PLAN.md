# Semantic Router Benchmark Harness — Plan

## Goal

Quantify the value of the [vLLM Semantic Router](https://github.com/vllm-project/semantic-router)
by measuring two things across a curated query set:

1. **Routing accuracy** — does the router select a model at or above the expected minimum
   tier (and matching specialization) for each query?
2. **Response quality** — does the response from the router-selected model meet the bar
   set by a gold reference produced by a top-tier model?

The harness must be resumable, scale to 10,000+ queries, and treat the SQLite database
as the canonical record of every run.

## Tech stack

- **Python 3.11+**
- **SQLite** (single-file, resumable, easy to inspect; comfortably handles 10k+ rows)
- **SQLAlchemy 2.x** for schema and sessions
- **PyYAML** for config
- **Typer** + **Rich** for CLI and review TUI
- **httpx** (async) for OpenAI-compatible API calls — every model tier, the gold model,
  the judge, and the router itself are reached over OAI-compatible HTTP
- **pytest** for tests
- **uv** for dependency and venv management

## Repository layout

```
semantic-router/
├── PLAN.md                       # this document
├── README.md                     # quickstart, points at PLAN.md
├── Makefile                      # all operational entry points
├── pyproject.toml                # Python package + tool config
├── .gitignore
├── config/
│   ├── models.yaml               # 5+ tiers, all OpenAI-compatible endpoints
│   ├── router.yaml               # vLLM Semantic Router config (passed to subprocess)
│   ├── gold.yaml                 # gold model endpoint + params
│   ├── judge.yaml                # LLM-as-judge endpoint + rubric
│   └── scoring.yaml              # rubric and score scale
├── data/
│   ├── queries.yaml              # curated queries (prompt + expected_min_tier + specs)
│   └── gold/                     # gold answers, one file per query (also stored in DB)
├── src/benchmark/
│   ├── __init__.py
│   ├── cli.py                    # Typer entrypoint
│   ├── config.py                 # YAML loaders + validation (pydantic)
│   ├── db.py                     # SQLAlchemy models, session, resume helpers
│   ├── tiers.py                  # ModelTier client (OAI-compatible)
│   ├── router_proc.py            # manages vLLM Semantic Router subprocess lifecycle
│   ├── router_client.py          # HTTP client to the router's OAI endpoint
│   ├── seed.py                   # load queries.yaml into DB
│   ├── gold.py                   # generate/refresh gold answers
│   ├── pass1.py                  # routing accuracy
│   ├── pass2.py                  # response generation
│   ├── review.py                 # human scoring TUI
│   ├── judge.py                  # LLM-as-judge scoring
│   └── report.py                 # aggregation, export, summary stats
└── tests/
```

## Router integration

The vLLM Semantic Router is a Go service that exposes an OpenAI-compatible endpoint, so
it is not Python-importable. The harness manages it as a subprocess:

- `router_proc.py` starts the router with `config/router.yaml`, waits for its health
  endpoint to come up, and shuts it down on exit (context-managed).
- `router_client.py` is a thin async OAI-compatible HTTP client. It surfaces both the
  selected model (from response metadata or a router-specific header) and the
  generation, so a single call can feed both passes when desired.
- The router binary path is configurable; if absent, `make setup` prints install
  instructions rather than silently failing.

This keeps clean process isolation while presenting a one-command experience
(`make run` boots the router, runs the harness, and tears down).

## Model tiers (`config/models.yaml`)

All tiers are OpenAI-compatible endpoints — local vLLM, hosted APIs, or anything that
speaks the `/v1/chat/completions` shape. This makes on-system / off-system routing a
config swap with no code change.

```yaml
tiers:
  - name: tier1-tiny
    level: 1
    endpoint: http://localhost:8001/v1
    model_id: llama-3.2-1b-instruct
    api_key_env: TIER1_API_KEY        # optional; empty allowed for local
    specializations: [general]
    timeout_s: 30

  - name: tier2-small
    level: 2
    endpoint: http://localhost:8002/v1
    model_id: qwen2.5-7b-instruct
    api_key_env: TIER2_API_KEY
    specializations: [general, code]

  - name: tier3-mid
    level: 3
    endpoint: http://localhost:8003/v1
    model_id: mixtral-8x7b-instruct
    api_key_env: TIER3_API_KEY
    specializations: [general, code, math]

  - name: tier4-large
    level: 4
    endpoint: https://api.example.com/v1
    model_id: llama-3.1-70b-instruct
    api_key_env: TIER4_API_KEY
    specializations: [general, code, math, reasoning]

  - name: tier5-frontier
    level: 5
    endpoint: https://api.anthropic.com/v1   # or OpenAI / self-hosted, all OAI-compatible
    model_id: claude-opus-4-7
    api_key_env: TIER5_API_KEY
    specializations: [general, code, math, reasoning, creative]

  # Specialization variants slot in alongside numeric tiers. They carry their own
  # `level` for routing-accuracy comparisons.
  - name: tier3-vision
    level: 3
    endpoint: http://localhost:8013/v1
    model_id: qwen2-vl-7b
    api_key_env: TIER3_VISION_API_KEY
    specializations: [vision]

  - name: tier3-tts
    level: 3
    endpoint: http://localhost:8023/v1
    model_id: kokoro-tts
    api_key_env: TIER3_TTS_API_KEY
    specializations: [tts]
```

### Specialization taxonomy

```
general, code, math, reasoning, creative, vision, tts
```

Domain knowledge (scientific, legal, medical, finance, etc.) is **not** a specialization
— it's a free-form `domain_tags` list on each query. Reason: domain expertise is
typically a question of model *size* plus retrieval, not a separate model class, so
forcing it into the specialization axis would dilute that axis's meaning. We can revisit
if domain-specialized routing variants emerge.

## Curated queries (`data/queries.yaml`)

```yaml
- id: q00001
  prompt: "What is 2+2?"
  expected_min_tier: 1
  specializations: [general]
  domain_tags: []
  notes: trivial arithmetic

- id: q00017
  prompt: "Prove that the halting problem is undecidable."
  expected_min_tier: 4
  specializations: [reasoning, math]
  domain_tags: [computer-science]

- id: q00042
  prompt: "Refactor this Python class to use dataclasses: ..."
  expected_min_tier: 3
  specializations: [code]
  domain_tags: []

- id: q00103
  prompt: "Describe the contents of this image."
  attachments: [{type: image, path: data/attachments/q00103.png}]
  expected_min_tier: 3
  specializations: [vision]
  domain_tags: []
```

Gold answers are produced separately by `make gold` against the configured gold tier
(typically `tier5-frontier`) and stored both in the DB (canonical) and as files under
`data/gold/` for diffability and code review.

## Scale considerations (10k+ queries)

- **Concurrency** — passes 1 and 2 run with bounded asyncio concurrency
  (`--concurrency` flag; default 8). Per-row commits keep resume granular.
- **Per-row idempotency** — each `(run_id, query_id, pass)` row transitions
  `pending → success | error`. A second run picks up `pending`/`error` rows only.
- **Human review at scale** — a full human pass over 10k responses is impractical;
  `make review` supports stratified sampling (`--sample 200 --by specialization`) and
  prioritizing low-judge-score rows. The judge handles full coverage; humans calibrate
  and audit.
- **Reporting cost** — aggregate queries are SQL; `make report` emits both stdout
  summary and a CSV/JSON dump for downstream analysis.

## Database schema (canonical run state)

```sql
queries (
  query_id          TEXT PRIMARY KEY,
  prompt            TEXT NOT NULL,
  prompt_hash       TEXT NOT NULL,
  attachments_json  TEXT,                       -- list of {type, path}
  expected_min_tier INTEGER NOT NULL,
  specializations   TEXT NOT NULL,              -- JSON array
  domain_tags       TEXT,                       -- JSON array
  notes             TEXT,
  gold_answer       TEXT,
  gold_model        TEXT,
  gold_generated_at TIMESTAMP
)

runs (
  run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at          TIMESTAMP NOT NULL,
  finished_at         TIMESTAMP,
  router_config_hash  TEXT NOT NULL,
  models_config_hash  TEXT NOT NULL,
  notes               TEXT,
  status              TEXT NOT NULL              -- running | done | aborted
)

pass1_results (
  run_id                  INTEGER NOT NULL,
  query_id                TEXT NOT NULL,
  router_selected_model   TEXT,
  router_selected_tier    INTEGER,
  router_selected_specs   TEXT,                 -- JSON array
  meets_minimum_tier      INTEGER,              -- 0/1
  matches_specialization  INTEGER,              -- 0/1
  latency_ms              INTEGER,
  raw_routing_metadata    TEXT,                 -- JSON blob from router
  status                  TEXT NOT NULL,        -- pending | success | error
  error_msg               TEXT,
  attempted_at            TIMESTAMP,
  PRIMARY KEY (run_id, query_id)
)

pass2_results (
  run_id                INTEGER NOT NULL,
  query_id              TEXT NOT NULL,
  router_selected_model TEXT,
  response_text         TEXT,
  prompt_tokens         INTEGER,
  completion_tokens     INTEGER,
  latency_ms            INTEGER,
  status                TEXT NOT NULL,
  error_msg             TEXT,
  attempted_at          TIMESTAMP,
  PRIMARY KEY (run_id, query_id)
)

scores (
  run_id          INTEGER NOT NULL,
  query_id        TEXT NOT NULL,
  scorer          TEXT NOT NULL,                -- human | judge
  reviewer_id     TEXT NOT NULL,                -- username or judge model id
  score           INTEGER NOT NULL,             -- 1..5
  rubric_version  TEXT NOT NULL,
  rationale       TEXT,
  scored_at       TIMESTAMP NOT NULL,
  PRIMARY KEY (run_id, query_id, scorer, reviewer_id)
)
```

**Resume rule:** workers select rows where
`status IN ('pending','error') AND run_id = :active_run` and process them with
per-row commits. Killing the process mid-run is safe; re-running picks up where it
stopped. Config-hash columns on `runs` make it explicit when the router or model
config changed mid-experiment.

## The two passes

### Pass 1 — Routing accuracy

For each query, send the prompt to the router and record only the routing decision
(selected model, tier, specializations, raw metadata). Compute:

- `meets_minimum_tier = router_selected_tier >= expected_min_tier`
- `matches_specialization = expected_specializations ⊆ router_selected_specs`

Pass 1 is cheap — useful for fast iteration when only routing logic changes.

### Pass 2 — Response quality

For each query, ask the router for a completion. Persist the full response, token
counts, and latency. Scoring is a separate phase so it can be done async by humans
or judge.

The two passes can be coalesced into a single router call to save cost when both
the routing config and tier configs are stable; this is a `--combined` flag on
`make run`.

## Scoring (Pass 2 quality)

`config/scoring.yaml` defines a 5-point rubric:

```
1 — unusable (wrong, off-topic, or refuses)
2 — partially correct but materially worse than gold
3 — acceptable; meets the bar a user would tolerate
4 — close to gold; minor gaps in completeness or polish
5 — matches or exceeds gold
```

**Human path** — `make review` opens a Rich TUI with prompt / gold / router response
side-by-side. Reviewer enters score + optional rationale. Resumable: only unreviewed
rows are shown, with stratified sampling support for large runs.

**Judge path** — `make judge` runs the configured judge model (typically the gold tier
or another frontier model) with the rubric as a system prompt. Scores written with
`scorer='judge'`. Useful for full-coverage scoring at 10k scale; humans remain ground
truth and calibrate the judge.

## Makefile targets

```
make setup           # create venv via uv, install deps, init DB schema
make seed            # upsert queries.yaml into DB (idempotent by query_id)
make gold            # generate or refresh gold answers; skips queries already gold'd
make run             # new run_id; starts router subprocess; pass1 then pass2; tears down
make pass1           # pass 1 only against active or specified run (resumable)
make pass2           # pass 2 only against active or specified run (resumable)
make review          # human scoring TUI for unreviewed pass-2 rows (supports --sample)
make judge           # LLM-as-judge scoring for unreviewed pass-2 rows
make report          # aggregate stats: routing accuracy, score histograms, per-spec breakdown
make resume RUN=<id> # explicit resume of a specific run
make clean-results   # wipe runs/results/scores; preserves queries and gold
make test
make fmt             # ruff + black
make lint            # ruff + mypy
```

All long-running targets accept `--concurrency`, `--limit`, and `--query-id` filters
through the underlying CLI for targeted reruns.

## Implementation milestones

1. **M0 — Skeleton (this commit).** PLAN.md, README, Makefile stubs, pyproject, gitignore.
2. **M1 — DB and config.** SQLAlchemy schema, YAML loaders, `make setup` and `make seed`.
   Hand-written 20-query starter set in `data/queries.yaml`.
3. **M2 — Tier client + gold.** `tiers.py`, `gold.py`, `make gold` working against one
   frontier endpoint. Seed gold for the 20-query set.
4. **M3 — Router lifecycle.** `router_proc.py` + `router_client.py`; `make run` boots
   the router, hits one query end-to-end, tears down.
5. **M4 — Passes.** `pass1.py`, `pass2.py` with concurrency, per-row resume, and the
   `--combined` mode.
6. **M5 — Scoring.** Judge first (automatable, easy to validate), then the human TUI.
7. **M6 — Reporting.** `make report` with stdout summary + CSV/JSON export.
8. **M7 — Scale-up.** Grow the curated set toward 10k+; add stratified review sampling
   and judge calibration metrics. Decide on the 10k commit based on M6 results.

## Open items deferred to implementation

- Exact router invocation surface (CLI flags, health endpoint, metadata header names) —
  resolved while building M3 by reading the router's docs and source.
- Whether to pin the router as a git submodule or rely on a system-installed binary —
  likely submodule for reproducibility, decided at M3.
- Vision and TTS evaluation — text scoring rubric works for vision (judge sees both
  images and text); TTS likely needs an audio diff or transcription-then-score step,
  scoped at M5.
