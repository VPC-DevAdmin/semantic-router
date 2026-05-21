# Semantic Routing Demo — Project Plan

> **This document is the source of truth.** When the project changes, update
> this file in the same commit. PLAN.md, [CLAUDE.md](CLAUDE.md), and
> [README.md](README.md) together let any new contributor (human or Claude
> session) become productive in minutes.

---

## 1. Goal

Demonstrate that a semantic router can send each query to the **smallest
model tier capable of answering it satisfactorily**, with substantial cost
savings versus sending everything to a frontier model (Opus).

The demo presents this as a single claim:
**the router picks the right tier, and that tier's answer is good enough.**

Behind the scenes this is two independent measurements (routing accuracy and
answer adequacy), but the audience sees one event per query.

## 2. Tiers

| Tier | Model (planned) | Notes |
|---|---|---|
| 1 | Small open model (~1B) | Trivial queries |
| 2 | Qwen 3 30B-A3B | MoE, ~3B active; most "real" queries |
| 3 | GPT-OSS 120B | Heavy reasoning and synthesis |
| 4 | Claude Sonnet | Frontier quality at API cost |
| 5 | Claude Opus | Top tier; also the gold-answer source |

In code and config, tiers are referenced by neutral identifiers
(`tier1` … `tier5`) so the model behind each tier can change without
touching any code. See `config/tiers/tierN.yaml`; `router_alias` is the
tier label the router emits, `endpoint` is the actual backend.

## 3. Dataset

`data/queries.json` is the curated query set. Every query carries:

- `id` (e.g. `q00001`) — unique
- `prompt` — the only thing the router sees
- `expected_answers: [{ answer, model, provider? }, …]` — one or more
  reference answers, ALWAYS a list (a single gold is a one-entry list).
  `model` is required and is the per-query unique key that becomes
  `gold_answers.model_id` and `data/evaluated_queries_with_answers.json`'s `model`. `provider` is an
  optional label (Anthropic / OpenAI / Google). `model_id` must be
  unique within the query. Extra fields are rejected (the loader
  validates with `extra=forbid`).
- `expected_min_tier` — the lowest tier we believe should answer this well
- `specializations` — free-form `list[str]` (downstream metadata only;
  see below). Common labels: `general`, `coding`, `math`, `reasoning`,
  `creative_writing`, `vision`, `tts` — these are what the tier YAMLs
  advertise, so use them if you want `matches_specialization` to report
  cleanly.
- `domain_tags` — free-form
- `notes`

`specializations` and `domain_tags` are **not** routing inputs — they
exist for downstream sort/review and the post-hoc `matches_specialization`
metric. The router sees only `prompt` (+ multimodal attachments). The
queries.json loader stores whatever labels you provide verbatim; the
tier YAMLs do enforce a whitelist (small, author-edited).

**Current distribution:** 110 queries — T1=25, T2=36, T3=32, T4=7, T5=10.

**Action items (data work, not code):**
- Add ~8–10 more T4 queries before the production run; T4 is the
  cost-savings story and is currently underpowered.
- (External, optional) Extract 2–5 `required_key_points` per gold answer —
  the load-bearing claims an adequate answer must cover. This is the
  highest-leverage input to eval quality but is consumed by the external
  judging workflow, not by this harness.

## 4. Adequacy rubric (documented for context — implemented externally)

Four dimensions, same for every query, judged independently:

1. **Correctness** — facts, code, math, reasoning all correct.
2. **Completeness** — covers the query-specific required key points
   extracted from the gold answer.
3. **Fitness for purpose** — right form (code/prose/recommendation) and
   right depth.
4. **No serious defects** — no refusals, padding, wrong-language responses,
   made-up citations, or instruction-following failures.

**Verdict:** Adequate (all four pass), Inadequate (any fail), or Borderline
(uncertain — flagged for human review).

The framing put to the judge:
> *"A user who would have received the gold answer received this candidate
> instead. Would they be satisfied?"*

Two judges run independently (Sonnet + one open model). Both Adequate =
pass. Both Inadequate = fail. Disagreement = human review.

Creative-writing queries (n=9) get a narrower adequacy definition (follows
requested form, engages with topic) or are shown qualitatively rather than
scored.

**This harness does NOT implement the judge.** The judging step happens
outside this repo, consuming the file produced by `make export`. The rubric
lives in PLAN.md so the broader story is legible and so the export shape
is justifiable.

## 5. Unified per-query outcome (downstream framing)

Each query ultimately gets one of four outcomes by combining the routing
decision with the externally-produced adequacy verdict:

| Outcome | Definition |
|---|---|
| **Hit** | Router picked ≥ min_tier AND answer judged adequate |
| **Wasteful hit** | Router picked > min_tier + 1 AND adequate (correct but over-spent) |
| **Quality miss** | Router picked ≥ min_tier BUT answer judged inadequate |
| **Under-route** | Router picked < min_tier |

Headline metric: **Hit %**. Cost story: `Hit / (Hit + Wasteful)`. Quality
story: `Hit / (Hit + Quality miss)`.

Outcome derivation is also downstream — this harness produces the inputs
(router decision + tier answers) and an external script computes the
outcome from those plus the external judge verdicts.

## 6. Workflow

The demo is **pre-recorded and replayed**; no live inference during the
talk. The harness exists to drive the production pass.

**Production pass (run once, in this repo):**

```
make setup     # one-time
make load      # data/queries.json → SQLite
make route     # for each query: ask router which tier it picks
make answers   # for each query × each of T1..T5: get the answer
make export    # emit data/evaluated_queries_with_answers.json (the single artifact downstream uses)
```

The resulting `data/evaluated_queries_with_answers.json` is the only output anyone outside this repo
needs. It is the source of truth for the external judge, the replay UI,
slides, and plots.

**Replay (downstream, not in this repo):**
- Walks through queries one at a time from `data/evaluated_queries_with_answers.json`.
- Shows prompt, routed tier, routed answer, adequacy verdict.
- Optional drill-down: signal trace, what other tiers produced for the
  same query.

## 7. The `data/evaluated_queries_with_answers.json` shape

`make export` writes a single JSON file containing, per query (MULTI-MODEL
shape — a tier can front several provider models and they're all called):

```jsonc
{
  "id": "q00001",
  "prompt": "What is the capital of France?",
  "specializations": ["general"],
  "expected_min_tier": 1,
  "routed_tier": 3,                  // from `make route`
  "routing_metadata": { "selected_model": "tier3", "selected_tier": 3,
                         "latency_ms": 42,   // router decide time, per query
                         "raw": { ... } },

  // Per-provider gold (from queries.json + make update-gold + make import-answers).
  "expected_answers": [
    { "provider": null,        "model": "upstream",        "answer": "Paris." },
    { "provider": "Anthropic", "model": "claude-opus-4-7", "answer": "Paris, ..." }
  ],

  // EVERY model the routed tier fronts (one per provider configured).
  "routed_answers": [
    { "tier": 3, "provider": "OpenAI", "model": "gpt-5-mini",   "answer": "Paris.", "status": "success", "latency_ms": 1234 },
    { "tier": 3, "provider": "Google", "model": "gemini-flash", "answer": "Paris.", "status": "success", "latency_ms":  980 }
  ],

  // Grouped by tier; each a list of {provider, model, answer}.
  "all_tier_answers": { "tier3": [ { "provider": "OpenAI", "model": "gpt-5-mini", "answer": "Paris." }, ... ] }
}
```

The external judge compares each `routed_answers[]` entry against the
`expected_answers[]` set, so users can see how the outcome changes on
OpenAI / Google vs. Anthropic. **Comparisons are always routed-vs-top,
never top-vs-top:** queries the router sends to the top tier are skipped
by `make answers` (no model calls); their per-provider answers ARE the
gold, produced by `make update-gold` (which calls every top-tier model)
and the `expected_answers` declared in queries.json. So a top-tier-routed
query has `expected_answers[]` populated and `routed_answers: []`.
(Pre-multi-model the shape was `responses.{gold,routed}` with a single
answer each — superseded.)

## 8. Repository layout

```
semantic-router/
├── PLAN.md                       # this document — source of truth
├── CLAUDE.md                     # context primer for fresh Claude sessions
├── README.md                     # operator quickstart
├── Makefile                      # user-facing targets + supporting
├── pyproject.toml
├── .env                          # secrets (gitignored); .env.example is the template
├── config/
│   ├── tiers/                    # SINGLE SOURCE OF TRUTH — one yaml per tier
│   │   ├── README.md             # schema + backend.kind table
│   │   ├── tier1.yaml ... tier5.yaml
│   ├── router.yaml               # process-management for `vllm-sr` subprocess
│   ├── router-exemplars.yaml     # contrastive-embedding training data — source of truth
│   ├── router-backends.yaml      # flat per-tier endpoints for the exemplar builder
│   └── router-config.yaml        # GENERATED (gitignored) — what `vllm-sr serve --config` reads
├── data/
│   └── queries.json              # 110 queries with `expected_answers[]` gold
├── src/benchmark/
│   ├── cli.py                    # Typer entrypoint
│   ├── config.py                 # pydantic-validated config loaders (scans config/tiers/)
│   ├── db.py                     # SQLAlchemy schema + session_scope
│   ├── load.py                   # queries.json → DB
│   ├── tiers.py                  # async OAI-compatible client
│   ├── router_proc.py            # `vllm-sr serve` lifecycle (launcher pattern)
│   ├── router_client.py          # talks to Envoy; extracts x-vsr-* headers
│   ├── build_router_config.py    # builds config/router-config.yaml from exemplars + backends
│   ├── start_llm.py              # `start-llm`/`stop-llm` — backend.kind dispatcher
│   ├── runs.py                   # run lifecycle + per-row resume + RUN_NEW resets
│   ├── pass1.py                  # `make route` — routing decisions
│   ├── answers.py                # `make answers` — one-call-per-query, error-tolerant
│   └── export.py                 # `make export` — produces data/evaluated_queries_with_answers.json
├── tools/
│   └── oai_mock.py               # stdlib OAI mock for pipeline validation
└── tests/                        # unit tests covering everything except live router
```

## 9. The Makefile

| Target | What it does |
|---|---|
| `make setup` | venv + deps + DB + installs `vllm-sr` if missing |
| `make load` | validate exemplars; build `config/router-config.yaml`; `data/queries.json` → DB |
| `make route` | rebuild router-config; send each query through router with `max_tokens=1`, capture `x-vsr-selected-model` |
| `make answers` | For each routed query: call the tier the router picked. **One call per query**, errors don't fail the pass — they get retried on the next invocation |
| `make export` | Read DB → write `data/evaluated_queries_with_answers.json` |
| `make start_LLM` / `make stop_LLM` | Bring up / tear down local-CPU tier backends from `config/tiers/*.yaml` |
| `make mock-bg` / `make mock-stop` | Local OAI mock for pipeline validation |

Both `route` and `answers` accept `RUN_NEW=true` (env-var passthrough) —
this deletes the relevant rows for the active run before re-seeding, so
the next invocation starts from a clean slate.

Supporting: `resume`, `clean-results`, `router-smoke`, `router-stop`,
`test`, `fmt`, `lint`.

`make answers` bypasses the router. The router has already been asked
(in `make route`) which tier it would pick; `make answers` reads that
tier from `pass1_results` and dials the tier's endpoint directly.

## 10. Data model (SQLite intermediate)

SQLite is the resumable intermediate store. `make export` reads from here
and produces `data/evaluated_queries_with_answers.json`. The DB is gitignored; `data/evaluated_queries_with_answers.json` is the
output artifact.

```
queries          (query_id PK, prompt, prompt_hash, expected_min_tier,
                  specializations, domain_tags, notes, attachments_json)
runs             (run_id PK, started_at, finished_at, status, notes)
pass1_results    (run_id, query_id PK, router_selected_tier,
                  raw_routing_metadata, status, ...)
tier_answers     (run_id, query_id, tier_level, model_id PK, model_slot,
                  provider, tier_name, response_text, prompt_tokens,
                  completion_tokens, latency_ms, status, ...)
gold_answers     (query_id, model_id PK, provider, answer,
                  generated_at)   -- per-provider expected answers
```

`tier_answers` PK is `(run_id, query_id, tier_level, model_id)`: the
router picks one tier and `make answers` calls **every model that tier
fronts**, so a routed query produces one row per model. `gold_answers`
holds the per-provider expected set — seeded from queries.json at
`make load` (one row per `expected_answers` entry), with additional rows
from `make update-gold` and `make import-answers`. **Schema changed: a
fresh DB or `make clean-results` + reseed is required (no migration
script).**

**Resume rule:** workers select rows where `status IN ('pending', 'error')`
for the active run. Per-row session commits make killing the process
mid-run safe. Errors do not fail the pass — they're left in `status='error'`
for the next invocation to retry. `RUN_NEW=true` resets either pass.

## 11. Router integration model

`vllm-sr serve` is **a launcher, not a daemon**. It brings up a Docker
stack (router + envoy + dashboard + simulator + datastores +
observability) and exits cleanly with code 0. The router service lives in
those background containers, managed by the host `vllm-sr` CLI.

The harness handles this in [`router_proc.py`](src/benchmark/router_proc.py):

1. Run `vllm-sr serve --config config/router-config.yaml --minimal` synchronously.
2. Wait for the launcher subprocess to exit. Exit 0 = launch succeeded.
3. Poll `/ready` on the router's apiserver (`:8080`) until it returns 200.
4. Hand control to the benchmark passes.
5. Leave the stack running on exit (cold-start is slow, repeat runs are
   fast). Set `stop_on_exit: true` in `config/router.yaml` to tear down,
   or run `make router-stop` manually.

`config/router-config.yaml` is the router's internal config, generated
from `config/router-exemplars.yaml` + `config/router-backends.yaml` as
part of `make load` and `make route`. It uses contrastive-embedding
exemplars for routing decisions. It's the file that `--config` points at.

The routing decision lands in three response headers added by the router
on 2xx-non-cached responses:

- `x-vsr-selected-model` → the tier id, e.g. `tier3`
- `x-vsr-selected-category` → e.g. `math`
- `x-vsr-selected-reasoning` → `on` | `off`

Each `config/tiers/tierN.yaml`'s `router_alias` is the value the router
emits in `x-vsr-selected-model`; `level` is the numeric tier. The
shipped-configs test asserts every tier the exemplar-built
`router-config.yaml` declares resolves to a `router_alias` in
`config/tiers/` — drift between the per-tier files and
`config/router-backends.yaml` is the most common config bug.

## 12. Backend strategy

Each tier is described by a single YAML in `config/tiers/`. That YAML
drives the direct-call paths and provisioning:

- `make answers` reads `endpoint.url` + `served_model_name` to call directly.
- `make route` reads `router_alias` to translate the router's
  `x-vsr-selected-model` header back into a tier level.
- `make start_LLM` reads `backend.kind` and dispatches to a per-kind
  launcher (`docker_vllm_dual_socket`, `remote`, `placeholder`).

The router itself reaches each tier via `config/router-backends.yaml`,
which is kept in sync by hand with `config/tiers/*.yaml`.

Swapping a tier's direct-call endpoint is a single-file edit in
`config/tiers/`; swapping the router-side endpoint takes a matching edit
in `config/router-backends.yaml`.

Phases of backend deployment, ordered by maturity:

**Phase A — Local OAI mock.**
T1, T3, T4, T5 endpoints point at `tools/oai_mock.py` on `host.docker.internal:8811`.
Validates the pipeline end-to-end at zero cost. Mock returns tier-tagged
canned text so downstream verification can confirm which tier served each
row. Not the bundled `vllm-sr-sim` — that speaks FleetSim, not OAI. Start
with `make mock-bg`.

**Phase B — T1 + T2 on local CPU.**
- T1: small CPU runner, **TBD** — `tier1.yaml`'s `backend.kind` is
  `placeholder`. When the user provides the runner, add a new kind +
  dispatcher in `src/benchmark/start_llm.py` and set `backend.kind`
  accordingly.
- T2: vLLM Qwen3-30B-A3B-Instruct-2507 (BF16), dual NUMA-pinned replicas
  on host ports 8000 / 8001. Launch with `make start_LLM`. Note: vllm-sr
  generates LOGICAL_DNS envoy clusters which only accept one endpoint
  each, so the `tier2` entry in `config/router-backends.yaml` points at
  r0 only. Adding an LB proxy in front of r0+r1 (haproxy / nginx on a
  single port) unlocks full dual-replica throughput; deferred.

**Phase C — T3 GPU + Anthropic.**
T3 on external GPU server (`tier3.yaml`'s `endpoint.url` for direct calls
and the matching `tier3` entry in `config/router-backends.yaml` for the
router get the GPU host:port). T4/T5 on Anthropic via the OAI-compat
endpoint. `ANTHROPIC_API_KEY` lives in `.env` (gitignored), referenced
by `endpoint.api_key_env` in tier4/tier5 YAMLs.

**Open question for Phase C — vendor model-name handling.** Anthropic's
OAI endpoint expects the real model id (e.g. `claude-opus-4-7`), not our
neutral `tier5`. The TierConfig already separates `router_alias` from
`served_model_name`, so the request body's `model` field can be the real
vendor id while the router still emits `tier5` in its header. Confirm
vllm-sr forwards the body's `model` field verbatim (or rewrites it to
`provider_model_id`) before relying on this.

## 13. Current state

Feature-complete and dogfooded end-to-end. `make load → make route →
make answers → make export` produces a full
`data/evaluated_queries_with_answers.json` artifact against the real
`vllm-sr` install plus a mix of vLLM, OpenAI, Anthropic, and Google
upstream backends.

Implementation chronology — what shipped when, and what surprised us
along the way — lives in [`docs/history.md`](docs/history.md).

## 14. Known open design questions

- **Tier-cutoff calibration.** Default cutoffs are `[0.20, 0.40, 0.60, 0.80]`
  — even splits of [0, 1] before we have real production data. Per the
  exemplars-file comment, the operator should look at the
  `request_difficulty` distribution across the eval set after the first
  production run and adjust cutoffs so the bands match the desired tier
  mix. Tunable in one place: `tier_cutoffs:` in `router-exemplars.yaml`.
- **Docker networking from router to backend on Linux.**
  `host.docker.internal:8810` works on Docker Desktop but may need
  `--add-host=host.docker.internal:host-gateway` on Linux. If not, the
  alternative is to point backends at the container name
  (`vllm-sr-sim-container:8000`) and ensure the simulator is attached to
  `vllm-sr-network`.
- **Tier identifiers vs. real model names.** Keep `tier1..tier5` in code
  and configs so we can rotate models behind a tier with no code edits.
  Surface real model names only in each tier YAML's `endpoint` /
  `served_model_name` fields and in reporting metadata.

## 15. Testing strategy

- **Unit tests** cover the schema, loaders, run lifecycle, pass logic with
  mocked router responses, and shipped config files.
- **No live-router tests in CI.** `RouterProcess` and `RouterClient` are
  exercised with `httpx.MockTransport` and `subprocess.Popen` mocks. The
  one thing not testable in CI is whether the actual `vllm-sr` binary
  behaves as documented — that's what dogfooding is for, and is what
  surfaced the launcher-vs-daemon bug we already fixed.
- **`tests/test_shipped_configs.py`** asserts every YAML/JSON file under
  `config/` and `data/` actually parses with the real loaders, plus that
  the exemplar-built `router-config.yaml`'s tier names line up with
  `router_alias` values from `config/tiers/*.yaml`.

## 16. Maintaining this document

PLAN.md is the project's source of truth for **current design**. When
state changes, update the relevant section in the same commit:

- **§13 Current state** stays a one-paragraph summary. Detailed status
  belongs in tests or in the active TODO file, not here.
- **§14 Open questions** gets new entries when we hit an unknown.
- **`docs/history.md`** captures the chronology of what we built and
  what surprised us. New entries land there; nothing moves the other
  direction.

A fresh Claude session opening this repo should be able to read
PLAN.md + CLAUDE.md and know what's true. If they need *why* a
decision was made, history.md is the next stop.

