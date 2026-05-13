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
touching any code. See `config/models.yaml`; the `model_id` is the tier
label, the `endpoint` is the actual backend.

## 3. Dataset

`data/queries.json` is the curated query set. Every query carries:

- `id` (e.g. `q00001`)
- `prompt`
- `expected_answer` — the gold standard, produced by Opus upstream
- `expected_min_tier` — the lowest tier we believe should answer this well
- `specializations` — `general | coding | creative_writing | math | reasoning`
- `domain_tags` — free-form
- `notes`

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
make export    # emit demo.json (the single artifact downstream uses)
```

The resulting `demo.json` is the only output anyone outside this repo
needs. It is the source of truth for the external judge, the replay UI,
slides, and plots.

**Replay (downstream, not in this repo):**
- Walks through queries one at a time from `demo.json`.
- Shows prompt, routed tier, routed answer, adequacy verdict.
- Optional drill-down: signal trace, what other tiers produced for the
  same query.

## 7. The `demo.json` shape

`make export` writes a single JSON file containing, per query:

```jsonc
{
  "id": "q00001",
  "prompt": "What is the capital of France?",
  "specializations": ["general"],
  "expected_min_tier": 1,
  "routed_tier": 1,                  // from `make route`
  "routing_metadata": {              // signal trace from the router
    "selected_category": "...",
    "selected_reasoning": "off",
    "raw_headers": { ... }
  },
  "responses": {
    "gold":  { "tier": 5, "answer": "Paris." },     // from data/queries.json
    "routed": { "tier": 1, "answer": "Paris." }      // tier1 from `make answers`
  },
  "all_tier_answers": {              // optional: every tier's response, for drill-down
    "tier1": "Paris.",
    "tier2": "Paris, France.",
    "tier3": "...",
    "tier4": "...",
    "tier5": "..."
  }
}
```

The `responses` block is the minimum the external judge needs: two LLM
outputs to compare for each query (gold vs. routed tier). The
`all_tier_answers` block is included so the replay UI can show "what would
tier X have said" without re-running anything.

## 8. Repository layout

```
semantic-router/
├── PLAN.md                       # this document — source of truth
├── CLAUDE.md                     # context primer for fresh Claude sessions
├── README.md                     # operator quickstart
├── Makefile                      # 5 user-facing targets + supporting
├── pyproject.toml
├── config/
│   ├── models.yaml               # tier1..tier5 → backend endpoint
│   ├── router.yaml               # process-management for `vllm-sr` subprocess
│   ├── vllm-sr.yaml              # the router's INTERNAL config
│   └── (no judge.yaml or scoring.yaml — judging is external)
├── data/
│   └── queries.json              # 110 queries with `expected_answer` gold
├── src/benchmark/
│   ├── cli.py                    # Typer entrypoint
│   ├── config.py                 # pydantic-validated config loaders
│   ├── db.py                     # SQLAlchemy schema + session_scope
│   ├── load.py                   # queries.json → DB (gold from `expected_answer`)
│   ├── tiers.py                  # async OAI-compatible client
│   ├── router_proc.py            # `vllm-sr serve` lifecycle (launcher pattern)
│   ├── router_client.py          # talks to Envoy; extracts x-vsr-* headers
│   ├── runs.py                   # run lifecycle + per-row resume
│   ├── pass1.py                  # `make route` — routing decisions
│   ├── answers.py                # `make answers` — per-tier answer collection
│   └── export.py                 # `make export` — produces demo.json
└── tests/                        # unit tests covering everything except live router
```

## 9. The Makefile

| Target | What it does |
|---|---|
| `make setup` | venv + deps + DB + installs `vllm-sr` if missing |
| `make load` | `data/queries.json` → DB (queries + gold) |
| `make route` | For each query: send through router with `max_tokens=1`, capture `x-vsr-selected-model` header → tier number |
| `make answers` | For each query × each tier (T1..T5): call the tier's backend directly, save the full response |
| `make export` | Read DB + router decisions + tier answers → write `demo.json` |

Supporting: `resume`, `clean-results`, `router-smoke`, `router-stop`,
`test`, `fmt`, `lint`.

Note that `make answers` bypasses the router. The router has already been
asked (in `make route`) which tier it would pick; for `make answers` we
need every tier's response so the demo can show what each tier produced.

## 10. Data model (SQLite intermediate)

SQLite is the resumable intermediate store. `make export` reads from here
and produces `demo.json`. The DB is gitignored; `demo.json` is the
checked-in artifact when stable.

```
queries          (query_id PK, prompt, prompt_hash, expected_min_tier,
                  specializations, domain_tags, gold_answer, gold_model, ...)
runs             (run_id PK, started_at, finished_at, status, notes)
pass1_results    (run_id, query_id PK, routed_tier, raw_routing_metadata,
                  status, ...)
tier_answers     (run_id, query_id, tier_level PK, response_text,
                  prompt_tokens, completion_tokens, latency_ms, status, ...)
```

The `tier_answers` table replaced the original `pass2_results` when
`make answers` was implemented. PK is `(run_id, query_id, tier_level)`;
`tier_name` is the `model_id` from `models.yaml` so the export step can
write a `tier_name → response_text` map without re-joining.

**Resume rule:** workers select rows where `status IN ('pending', 'error')`
for the active run. Per-row session commits make killing the process
mid-run safe.

## 11. Router integration model

`vllm-sr serve` is **a launcher, not a daemon**. It brings up a Docker
stack (router + envoy + dashboard + simulator + datastores +
observability) and exits cleanly with code 0. The router service lives in
those background containers, managed by the host `vllm-sr` CLI.

The harness handles this in [`router_proc.py`](src/benchmark/router_proc.py):

1. Run `vllm-sr serve --config config/vllm-sr.yaml --minimal` synchronously.
2. Wait for the launcher subprocess to exit. Exit 0 = launch succeeded.
3. Poll `/ready` on the router's apiserver (`:8080`) until it returns 200.
4. Hand control to the benchmark passes.
5. Leave the stack running on exit (cold-start is slow, repeat runs are
   fast). Set `stop_on_exit: true` in `config/router.yaml` to tear down,
   or run `make router-stop` manually.

`config/vllm-sr.yaml` is the router's internal config — 5 tier models with
backends, plus keyword-signal routing decisions. It's the file that
`--config` points at.

The routing decision lands in three response headers added by the router
on 2xx-non-cached responses:

- `x-vsr-selected-model` → the tier id, e.g. `tier3`
- `x-vsr-selected-category` → e.g. `math`
- `x-vsr-selected-reasoning` → `on` | `off`

`config/models.yaml` maps each `x-vsr-selected-model` value back to a
numeric tier level. The shipped-configs test asserts every model name in
`vllm-sr.yaml` has a matching `model_id` entry in `models.yaml` — drift
between the two is the most common config bug.

## 12. Backend strategy

The harness is intentionally agnostic about what's behind each tier label.
Three phases of backend deployment, ordered by maturity:

**Phase A — Simulator only (where we are now).**
All 5 tiers point at the bundled `vllm-sr-sim` mock at
`host.docker.internal:8810`. Validates routing pipeline + export end-to-end
with zero LLM cost. Pass-2 responses are stub text; useful for plumbing
verification, not adequacy claims.

**Phase B — Real small + simulator.**
Two CPU models (T1 + T2) on the user's existing CPU server, T3–T5 still on
the simulator. First real signal on small-model performance.

**Phase C — Full ladder.**
T1–T2 on CPU, T3 on GPU server (once acquired), T4 = Anthropic Sonnet,
T5 = Anthropic Opus, via their OpenAI-compatible endpoints. Real Pass-2
responses; real numbers for the demo.

Swapping phases is a config-only change: update `backend_refs.endpoint` in
`config/vllm-sr.yaml` per tier, update the matching `endpoint` in
`config/models.yaml` for direct-tier calls in `make answers`. No code
changes.

## 13. Current state and roadmap

### What works (HEAD)

- ✅ `make setup` installs vllm-sr, venv, DB schema
- ✅ `make load` reads 110 queries with embedded gold into the DB
- ✅ `make route` launches `vllm-sr serve` with our config; apiserver
  `/ready` returns 200
- ✅ `make answers` collects per-tier responses with per-row resume
  (verified against unit tests; awaits real tier backends to be exercised)
- ✅ `make export` emits `demo.json` even with partial data — entries
  with no pass1 row or no tier_answers get null fields for the missing
  pieces

### Active blocker

- ❌ `make route` chat-completion requests to Envoy at
  `http://127.0.0.1:8899/v1/chat/completions` return HTTP 404. Envoy is
  listening but doesn't know how to route the path.

**Hypotheses, in order of likelihood:**

1. Our `config/vllm-sr.yaml`'s `listeners` block is missing a field
   (`paths:`, `routes:`, or a reference to a filter chain) that drives
   Envoy route generation. The auto-generated `.vllm-sr/envoy.yaml` will
   reveal exactly what's missing.
2. The path is non-standard in this build of the router. Unlikely — their
   docs are explicit about `/v1/chat/completions`.
3. `host.docker.internal:8810` may not resolve from inside the router
   container on Linux. Would cause 5xx not 404, so probably not the issue
   *for the 404* but will bite us next.

**Diagnostic order (run on the server where vllm-sr is installed):**

1. `vllm-sr chat "hello"` — does upstream's own CLI client get through
   Envoy? If yes, our request shape is the bug. If no, our config + the
   auto-generated Envoy config are the bugs.
2. `cat .vllm-sr/envoy.yaml` — find `route_config.virtual_hosts[].routes`.
   If it's empty or lacks a `/v1/chat/completions` matcher, that's the
   smoking gun.
3. `vllm-sr logs envoy 2>&1 | tail -50` — Envoy's verdict on the request.
4. `vllm-sr logs router 2>&1 | tail -50` — has the router-side service
   registered anything?

### Roadmap (in order)

1. **Unblock `make route`.** Resolve the Envoy 404 (above).
2. **Add ~8–10 more T4 queries** to `data/queries.json`.
3. **Phase B backend rollout:** stand up T1 + T2 on the CPU server,
   update `config/models.yaml` and `config/vllm-sr.yaml` endpoints.
4. **Phase C backend rollout:** wire Anthropic for T4 + T5.
5. **First production pass** — `make route && make answers && make export`
   produces a real `demo.json`. Hand to external judging workflow.

### Done

- ~~**Implement `make answers`.**~~ `src/benchmark/answers.py`; new
  `tier_answers` table with PK `(run_id, query_id, tier_level)`.
- ~~**Implement `make export`.**~~ `src/benchmark/export.py`; emits
  `demo.json` per §7. Resilient to missing data — emits null fields
  where pass1 or tier_answers haven't run.

## 14. Known open design questions

- **Envoy route generation in `vllm-sr serve`.** What field in
  `config/vllm-sr.yaml`'s `listeners` block drives the auto-generated
  Envoy `route_config.virtual_hosts[].routes`? The minimal upstream
  example we modeled ours on may have omitted required fields. Confirm by
  inspecting `.vllm-sr/envoy.yaml` and the upstream's 1,254-line reference
  config.
- **Docker networking from router to backend on Linux.**
  `host.docker.internal:8810` works on Docker Desktop but may need
  `--add-host=host.docker.internal:host-gateway` on Linux. If not, the
  alternative is to point backends at the container name
  (`vllm-sr-sim-container:8000`) and ensure the simulator is attached to
  `vllm-sr-network`.
- **Tier identifiers vs. real model names.** Keep `tier1..tier5` in code
  and configs so we can rotate models behind a tier with no code edits.
  Surface real model names only in `models.yaml`'s `endpoint` and in
  reporting metadata.

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
  `vllm-sr.yaml`'s model names line up with `models.yaml`'s `model_id`
  values.

## 16. Maintaining this document

PLAN.md is the project's source of truth. When state changes, update the
relevant section in the same commit:

- **§13 Current state** must reflect what works and what's broken at HEAD.
  When a bug is fixed, remove it from "active blocker" and add a one-line
  note in §17 history.
- **§13 Roadmap** gets reordered or pruned as items ship.
- **§14 Open questions** gets new entries when we hit an unknown;
  resolved questions move to §17 with the answer.
- **§17 History** captures the chronology of what we built and what
  surprised us.

A fresh Claude session opening this repo should be able to read PLAN.md +
CLAUDE.md and know what's true, what's not, and what to do next.

## 17. Implementation history

Chronological record of what we built and what surprised us. New entries
go at the bottom.

- **M0** — Plan + skeleton: PLAN.md, README, Makefile stubs.
- **M1** — DB schema, config loaders, idempotent loader (SQLAlchemy +
  pydantic).
- **M2** — `tiers.py` OAI-compatible client (Pass 2 backbone).
- **M3** — Router subprocess + client + `x-vsr-*` header extraction.
  *Surprise: initially modeled `vllm-sr serve` as a daemon; it's actually
  a launcher that spawns a Docker stack and exits. Corrected during
  dogfooding.*
- **M4** — Pass 1 + Pass 2 + run lifecycle + per-row resume; bounded
  concurrency.
- **M5** — *(deleted)* Originally added LLM-as-judge + human review TUI.
  Removed when we narrowed scope to "harness produces inputs; judging is
  external."
- **M6** — *(deleted)* Originally added aggregate report. Removed; the
  demo consumes `demo.json` directly via the replay UI.
- **Simplification pass** — queries.json (not YAML), folded `seed` and
  `gold` targets into `make load`, dropped `validate-config`.
- **Dogfood fixes** — RouterProcess launcher pattern; `make setup`
  installs vllm-sr; shipped first real `config/vllm-sr.yaml` with 5 tiers
  + keyword routing.
- **Scope reframe** — pivoted from "continuously-runnable benchmark
  harness with internal scoring" to "production-pass that emits a single
  `demo.json` consumed by external judging + replay UI." Dropped M5+M6
  surface area; added `make answers` (per-tier collection) and
  `make export` as the new top-level targets.
- **Prune** — deleted `judge.py`, `review.py`, `report.py`, `pass2.py`,
  the `Score` table, and related configs/tests; folded `install-router`
  into `make setup`. Net −1,729 lines.
- **answers + export landed** — new `src/benchmark/answers.py` and
  `src/benchmark/export.py`; new `tier_answers` table with PK
  `(run_id, query_id, tier_level)`; CLI commands `answers` and `export`
  wired; `make answers` and `make export` no longer stubs. End-to-end
  `make load && make export` produces a valid `demo.json` with null
  fields for parts that haven't run yet.
