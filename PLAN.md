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
                         "raw": { ... } },

  // Per-provider gold. source ∈ upstream | update-gold | import:<file>.
  "expected_answers": [
    { "source": "upstream",    "provider": null,        "model": "upstream",        "answer": "Paris." },
    { "source": "update-gold", "provider": "Anthropic", "model": "claude-opus-4-7", "answer": "Paris, ..." }
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
OpenAI / Google vs. Anthropic. There is no top-tier gold short-circuit:
every top-tier model is called and each is that provider's expected
answer. (Pre-multi-model the shape was `responses.{gold,routed}` with a
single answer each — superseded.)

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
│   └── queries.json              # 110 queries with `expected_answer` gold
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
│   └── export.py                 # `make export` — produces demo.json
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
| `make export` | Read DB → write `demo.json` |
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
and produces `demo.json`. The DB is gitignored; `demo.json` is the
output artifact.

```
queries          (query_id PK, prompt, prompt_hash, expected_min_tier,
                  specializations, domain_tags, gold_answer, gold_model, ...)
runs             (run_id PK, started_at, finished_at, status, notes)
pass1_results    (run_id, query_id PK, router_selected_tier,
                  raw_routing_metadata, status, ...)
tier_answers     (run_id, query_id, tier_level, model_id PK, model_slot,
                  provider, tier_name, response_text, prompt_tokens,
                  completion_tokens, latency_ms, status, ...)
gold_answers     (query_id, model_id PK, provider, answer, source,
                  generated_at)   -- per-provider expected answers
```

`tier_answers` PK is `(run_id, query_id, tier_level, model_id)`: the
router picks one tier and `make answers` calls **every model that tier
fronts**, so a routed query produces one row per model. `gold_answers`
holds the per-provider expected set — `source="upstream"` seeded from
queries.json at `make load`, plus `update-gold` / `import:<file>` rows.
`Query.gold_answer` is kept as a back-compat single-value mirror of the
slot-0 / upstream gold. **Schema changed for multi-model: a fresh DB or
`make clean-results` + reseed is required (no migration script).**

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

## 13. Current state and roadmap

### What works (HEAD)

- ✅ `make setup` installs vllm-sr, venv, DB schema
- ✅ `make load` reads 110 queries with embedded gold into the DB
- ✅ `make load` / `make route` build `config/router-config.yaml` from
  `config/router-exemplars.yaml` + `config/router-backends.yaml` and
  validate that exemplars don't overlap the eval set.
- ✅ `make route` — build router-config + run pass; verified 110/110
  succeed against the mock. Routing-accuracy is a separate data-quality
  story tracked elsewhere. `RUN_NEW=true` wipes pass1_results first.
- ✅ `make answers` — **one call per query** (routed tier from
  `pass1_results`), error-tolerant: failures stay as `status='error'` and
  retry on the next run. Verified 110/110 against the mock with the new
  semantics. `RUN_NEW=true` wipes tier_answers first.
- ✅ `make export` emits a complete `demo.json`: every query has
  `routed_tier`, routed answer; `all_tier_answers` has one entry (the
  routed tier).
- ✅ Local OAI mock (`tools/oai_mock.py`) + `make mock-bg` / `mock-stop`.
- ✅ `make start_LLM` / `make stop_LLM` — YAML-driven dispatch over
  `backend.kind`. Today: `docker_vllm_dual_socket` (T2 Qwen procedure)
  and `placeholder` / `remote` (no-ops).
- ✅ `.env` for secrets, auto-loaded by the CLI via `python-dotenv`.
  Gitignored; `.env.example` is the template.

### Active blocker

None. The previously-documented Envoy 404 was misdiagnosed: Envoy was
routing correctly all along, and the bundled `vllm-sr-sim` was the
404-source — it speaks the FleetSim API (`/api/fleets`, `/api/jobs`),
not `/v1/chat/completions`. Replaced with `tools/oai_mock.py`.

### Roadmap (in order)

1. **Add ~8–10 more T4 queries** to `data/queries.json` (per §3).
2. **Phase B — T1 + T2 rollout:**
   - Wire the user-provided T1 runner: edit `config/tiers/tier1.yaml`,
     change `backend.kind: placeholder` to the real runner kind, fill in
     `endpoint.url`. Mirror the host:port on the matching `tier1` entry
     in `config/router-backends.yaml`.
   - Point `config/tiers/tier2.yaml` at the real T2 backend; mirror in
     `config/router-backends.yaml`.
   - Run `make start_LLM && make router-stop && make route && make answers`.
3. **Phase C — T3 + Anthropic:**
   - T3: external GPU endpoint goes into `config/tiers/tier3.yaml`
     (endpoint; `backend.kind: remote`). Mirror in
     `config/router-backends.yaml`.
   - T4/T5: decide on the vendor-model-name approach (see §12 open
     question), then update `config/tiers/tier4.yaml` and `tier5.yaml`.
4. **First production pass** — `make route && make answers && make export`
   produces a real `demo.json`. Hand to external judging workflow.

### Done

- ~~**Multi-model tiers.**~~ A tier fronts N provider models (slot 0 =
  bare `TIER{N}_*`, indexed `TIER{N}_{i}_*`, optional `PROVIDER`).
  `make answers` calls every model in the routed tier; `gold_answers`
  holds the per-provider expected set; `demo.json` reshaped to
  `expected_answers[]` / `routed_answers[]` (§7). Top-tier gold
  short-circuit removed. Lexical/keyword routing removed (semantic only).
- ~~**Implement `make answers`.**~~ `src/benchmark/answers.py`;
  `tier_answers` PK `(run_id, query_id, tier_level, model_id)`.
- ~~**Implement `make export`.**~~ `src/benchmark/export.py`; emits
  `demo.json` per §7. Resilient to missing data — emits null fields
  where pass1 or tier_answers haven't run.
- ~~**Unblock `make route`.**~~ Bundled simulator doesn't speak OAI;
  replaced with `tools/oai_mock.py`. Pipeline now verified end-to-end.
- ~~**Plumbing for real backends.**~~ Per-tier YAMLs under `config/tiers/`
  are the single source of truth; swapping a backend is a one-file edit
  (endpoint + router_backend_refs + backend.kind). `make start_LLM` /
  `make stop_LLM` wraps the T2 dual-replica docker-run procedure via the
  `backend.kind` dispatcher in `src/benchmark/start_llm.py`.
- ~~**Router config emits real v0.3 schema.**~~ `build_router_config.py`
  rewritten to produce `routing.signals.embeddings[]` (two per axis:
  `<axis>_hard` and `<axis>_easy`) and `routing.decisions[]` with v0.3
  AND/OR/NOT composition, plus `providers.models[].backend_refs[]` for
  both OAI-compatible and Anthropic backends. Band-based exemplars file
  is unchanged (it's the audience-facing artifact); the builder does the
  band → Boolean translation. All 27 band combinations still reach all
  5 tiers per the routing test.
- ~~**Migrated to projections (v0.3 canonical pattern).**~~ The DIY
  hard/easy embedding signals + AND/OR/NOT rule tree hit a structural
  ceiling at ~84% routing accuracy because vllm-sr's `matched_signals`
  uses single-winner semantics — only the top-scoring signal globally is
  reported as matched, so rules that depended on multiple hard signals
  firing were structurally impossible to satisfy. Rewrote the builder
  and exemplars to the canonical projections shape:
    • `routing.signals.complexity[]` — one contrastive signal per axis
      with `hard` / `easy` candidate banks (`needs_reasoning`,
      `needs_expertise`, `needs_judgment`). Each emits a confidence ∈ [0, 1].
    • `routing.projections.scores.request_difficulty` (`weighted_sum`)
      combines per-signal confidences into a single continuous score.
    • `routing.projections.mappings.tier_band` (`threshold_bands`)
      partitions that score into 5 mutually-exclusive bands.
    • `routing.decisions[]` — one per tier, each conditioning on a
      single `{type: projection, name: tierN_band}`.
  Tuning surface is two human-readable knobs in `router-exemplars.yaml`:
  per-signal `weight:` (contribution to difficulty) and `tier_cutoffs:`
  (where to split the bands). Tests in `test_shipped_configs.py` assert
  the bands cover [0, 1] with no gaps/overlap and that scores at each
  cutoff land in the expected tier.

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
  installs vllm-sr; shipped first real `config/vllm-sr.yaml` with 5 tiers.
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
- **Active-blocker misdiagnosis corrected** — the Envoy 404 wasn't an
  Envoy route-generation bug. The autogenerated `.vllm-sr/envoy.yaml`
  did have correct routes (default `prefix: /` + per-tier
  `x-selected-model` header matches); the router's ext_proc emitted
  correct `routing_decision: default-tier1` logs; envoy forwarded
  successfully. The 404 came from the bundled `vllm-sr-sim` itself,
  which speaks the FleetSim API (`/api/fleets`, `/api/jobs`) and has
  no `/v1/chat/completions` endpoint. *Surprise: never substitute a
  simulator without checking its protocol.*
- **Mock + plumbing landed** — `tools/oai_mock.py` (stdlib-only,
  ~120 lines) replaces the bundled sim for pipeline validation.
  `make mock-bg` / `mock-stop` manage it. Each tier block in
  `config/vllm-sr.yaml` and `config/models.yaml` got a `# REAL:`
  comment marking the swap line so each tier could independently flip
  to its real backend with no code changes. End-to-end verified:
  `make route` 110/110, `make answers` 550/550, `make export`
  produces complete `demo.json`. (Superseded by the per-tier YAML
  refactor below; `# REAL:` markers are gone and `models.yaml` is
  removed.)
- **start_LLM / stop_LLM landed** — new Makefile targets wrap the
  documented vLLM dual-socket T2 procedure (NUMA-pinned r0/r1 on host
  ports 8000/8001, `--block-size 32`, 120 GB KV per replica). Stable
  container names so stop is straightforward. Env-var overrides for
  cores / NUMA / KV / ports. T1 is a clearly-marked placeholder
  block awaiting the user-provided runner.
- **Per-tier YAML refactor (source-of-truth split)** — `config/models.yaml`
  removed; each tier now has its own YAML in `config/tiers/`. The YAML
  drives every backend-aware code path (answers, gen-router-config,
  start_LLM). vllm-sr.yaml is now GENERATED (gitignored) from per-tier
  YAMLs + a hand-maintained `vllm-sr.routing.yaml` template; that means
  swapping a tier's backend (local docker → remote URL) is a single-file
  edit. `make start_LLM` is now a YAML-driven Python dispatcher
  (`src/benchmark/start_llm.py`) keyed off `backend.kind`.
- **make answers semantics flip** — was: 5 calls per query (one per
  tier). Now: 1 call per query (the routed tier from `pass1_results`).
  Unreachable upstreams mark rows `status='error'` instead of failing the
  pass; retries happen automatically on the next `make answers`.
- **RUN_NEW=true flag** — env-var passthrough to `--run-new` on
  `make route` and `make answers`; deletes the relevant rows for the
  active run before re-seeding.
- **.env support** — `python-dotenv` added; CLI auto-loads `.env` from
  CWD. Each tier's `endpoint.api_key_env` names which env var to read;
  for now T1-T3 are unauthed (mock or local), T4/T5 reference
  `ANTHROPIC_API_KEY`.
- **LOGICAL_DNS discovery** — vllm-sr generates envoy clusters of type
  LOGICAL_DNS, which only accept a single endpoint per cluster. Multi-
  ref `router_backend_refs` in a tier YAML breaks envoy at startup.
  Workaround: keep one ref per tier in vllm-sr.yaml; if T2 dual-replica
  load balancing is needed, front r0+r1 with haproxy/nginx exposing one
  port and reference that.
