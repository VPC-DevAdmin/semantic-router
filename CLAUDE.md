# CLAUDE.md — context primer for Claude sessions on this repo

You are working on a **benchmark harness** for the
[vLLM Semantic Router](https://github.com/vllm-project/semantic-router).
The router routes incoming queries to the right-sized LLM ("tiny CPU model
for trivial questions, frontier model for synthesis tasks"); this harness
measures whether it actually does that well.

**Read [PLAN.md](PLAN.md) for full design and current status.** This file is
the short primer.

## Workflow

User-facing make targets (see `make help`):

```
make setup              # one-time: venv + deps + DB + installs vllm-sr if missing
make load               # validate exemplars, build router-config.yaml, load queries.json → DB
make route              # rebuild router-config.yaml; routing pass via the local OAI mock (auto-started)
make answers            # for each routed query: call EVERY model the picked tier fronts (real upstreams)
make evaluate           # LLM-judge routed vs gold answers (per-row resumable, batched 50 queries/call)
make export             # emit data/routed_queries_with_answers.json + data/evaluations.json (if rows exist)
make demo-data          # build demo/data/demo_data.json from the exports + demo/pricing.json
make demo               # build demo data + serve the browser replay at :8000
make start_LLM          # YAML-driven launch of local-CPU tier backends (T2 docker procedure today)
make stop_LLM           # tear down local-CPU tier backends
make mock-bg            # local OAI mock on :18811 for pipeline validation
make mock-stop          # stop the mock
```

## Replay demo (`demo/`)

**`make demo` is a single command that works on a bare clone with no
`make setup`** — it needs only `python3` (the preprocessor is
stdlib-only + optional tiktoken; the server is stdlib http.server). It
serves the COMMITTED `demo/data/demo_data.json` as-is and opens the
browser, rebuilding only if the source exports are newer than the
dataset (i.e. you re-ran the pipeline). `make demo-data` forces a
rebuild. We deliberately don't force `make setup` from `make demo` —
the demo never touches vllm-sr or the dev toolchain, so pulling them
in would just slow down "show me the demo."

The dataset itself is built by `tools/build_demo_data.py` →
`demo/data/demo_data.json` from the committed exports +
`demo/pricing.json`, and the front-end replays the benchmark: queries flow through a router animation at the **real
measured throughput** (~22 qps), each carrying its **real routing
latency**; the readable detail panel samples ~1 query/1.5s. Per-query
cost is **real** (routed-side token counts from the export × supplier
pricing; frontier completion tokens tiktoken-estimated). Five tier
dropdowns + a frontier dropdown re-key the displayed answer, cost, and
the two judge verdicts live. The whole front-end is **data-driven** —
tiers, models, pricing, throughput all come from `demo_data.json`, so
dropping in a fresh dataset needs no code change. Editing supplier
rates = edit `demo/pricing.json` + re-run `make demo-data`.

`route` and `answers` both accept `RUN_NEW=true` to wipe the active run's
rows and re-seed before running. Errors in `answers` don't fail the pass —
they stay as `status='error'` and get retried automatically on the next
invocation.

**`make route` ALWAYS routes through the local OAI mock.** Pass 1 only
needs the router's decision headers (`x-vsr-selected-*`); it doesn't
need a real completion. The vllm-sr router is an Envoy proxy that
always forwards upstream, so without a mock every routed query would
consume a token AND surface vendor quirks (max_tokens vs
max_completion_tokens, temperature=0 on gpt-5, Anthropic adapter
body-rewrite, etc.). The mock (`tools/oai_mock.py`, port 18811) ACKs
with a tier-tagged canned reply. `make route` depends on `mock-bg` so
the mock auto-starts.

The router-config's model card names are the **tier ids**
(`tier1`…`tier5`) — not real model identifiers. The mock accepts
anything, so the routing pass stays purely tier-level: decisions,
headers, and TierLookup all speak tier IDs. `make answers` calls the
real upstream models directly (TIER{N}_{i}_MODEL from .env) and
bypasses the router entirely, so per-vendor names never need to flow
through the router-config.

Plus `resume`, `clean-results`, `router-smoke`, `router-stop`, `test`,
`fmt`, `lint`.

**Judging runs in this repo via `make evaluate`.** Reads one or more
`EVALUATOR_N_*` env slots (Anthropic / OpenAI / Google — same OAI-
compatible shape as the tier slots), batches 50 queries per judge call
(each query carrying its full (routed × gold) pair set so the judge
sees the cross-product once), writes verdicts to the `evaluations` DB
table with per-row resume. `make export` then writes
`data/evaluations.json` alongside `routed_queries_with_answers.json`
when any evaluation rows exist. The rubric (three 1-4 dimensions +
Adequate/Marginal/Failure verdict) lives in PLAN.md §4.

The external judging workflow that produced the seed
`data/evaluations.json` is still supported — the one-off
`benchmark import-evaluations <path>` CLI loads it into the same DB
table so future exports write it from there.

## Key files (canonical, don't be confused by lookalikes)

- **`data/queries.json`** — the curated 110-query benchmark set. Each query
  carries `expected_answers: [{answer, model, provider?}, …]` — always a
  list, even for a single gold. `model` is the per-query unique key
  (required). Extra fields are rejected by the loader, so stray legacy
  fields fail fast. `make load` upserts each entry into the
  `gold_answers` table (PK `(query_id, model_id)`); `update-gold` /
  `import-answers` add per-provider rows alongside. NOT YAML. NOT
  regenerated by any `make` target.
- **`config/tiers/tierN.yaml`** — tier metadata only (name, level,
  specializations, router_alias, timeout_s, max_tokens default). No
  endpoint, no backend — those moved out.
- **`.env`** — every callable model lives here as an indexed slot
  `TIER{N}_{i}_*` (i ≥ 1; no bare/slot-0 form, the loader raises on
  stale singular `TIER{N}_*`). Each slot carries URL + MODEL +
  optional API_KEY/PROVIDER/TIMEOUT/MAX_TOKENS/THINKING. The router
  picks one tier; `make answers` calls EVERY model that tier fronts.
- **`config/local_models.yaml`** — launch-recipe library. `make
  start_LLM` is tier-agnostic: it walks every env slot whose URL is
  localhost, looks the served model name up here, and executes the
  matching per-CPU-vendor (`amd:` / `intel:`) verbatim argv. Three
  placeholders fill in per launch: `{port}`, `{served_name}`,
  `{container_name}`. Each recipe optionally carries `extra_body`
  (Qwen3 thinking + sampler knobs) that the harness merges into chat
  requests against that model. Add a model → add an entry here.
- **`config/router-exemplars.yaml`** — the router's decision logic.
  Three complexity signals (`needs_reasoning`, `needs_expertise`,
  `needs_judgment`), each with ~12 hard + ~12 easy contrastive
  exemplars and a `weight:`. The builder compiles these into the v0.3
  canonical projections shape: signal confidences → `weighted_sum` into
  a single `request_difficulty` score → `threshold_bands` partition
  into 5 tier bands → one decision per band. Two tuning knobs live in
  this file: per-signal `weight:` and `tier_cutoffs:`. Compiled into
  `config/router-config.yaml` as part of `make load` and `make route`.
- **`config/router-backends.yaml`** — per-tier endpoints the router
  itself reaches (flat schema, separate from `config/tiers/*.yaml`).
- **`config/router-config.yaml`** — GENERATED build artifact passed to
  `vllm-sr serve --config`. Gitignored.
- **`config/router.yaml`** — process-management config for how the harness
  launches `vllm-sr`. Ports, args, log path. NOT routing rules.
- **`data/router_benchmark.db`** — canonical SQLite store. Gitignored.
  Inspect with `sqlite3 data/router_benchmark.db`. Schema in
  `src/benchmark/db.py`. (Older checkouts had this at `benchmark.db`
  in the repo root; if you upgrade, `mv` the file once.)
- **`.env`** — secrets (gitignored). `python-dotenv` auto-loads on every
  CLI invocation. Referenced by `endpoint.api_key_env` in tier YAMLs.
  `.env.example` is the committed template.
- **`config.yaml`** (project root) — autogenerated by `vllm-sr serve` if
  no `--config` is passed. **Gitignored.** We always pass `--config
  config/router-config.yaml` so this file is irrelevant in our flow.

## Conventions

- **JSON for data, YAML for human-edited config.** `queries.json` is data
  from an upstream source; configs are operator-tunable so they're YAML.
- **OAI-compatible everywhere.** All tier model endpoints and the router
  frontend speak `/v1/chat/completions`. Swap backends in tier YAML; no
  code changes.
- **Per-tier YAML drives direct calls.** Adding a tier = add a file in
  `config/tiers/`; swapping a direct-call backend = edit one line.
  Routing logic lives separately in `config/router-exemplars.yaml`;
  the router's per-tier endpoints live in `config/router-backends.yaml`.
- **Per-row resume.** Every pass row in the DB transitions
  `pending → success | error`. Workers re-process rows where
  `status IN ('pending', 'error')`. Killing mid-run is safe. Errors in
  `make answers` do not fail the pass — they retry on the next invocation.
- **`RUN_NEW=true`** on `make route` / `make answers` deletes existing
  rows for the active run before re-seeding.
- **Specializations:** the tier YAMLs are whitelisted to `general`,
  `coding`, `math`, `reasoning`, `creative_writing`, `vision`, `tts`
  (5 small author-edited files; catches typos cheaply). Queries.json
  specializations are FREE-FORM `list[str]` — they're downstream
  metadata (sort / review / the `matches_specialization` metric) and do
  not drive routing, so whatever labels the source uses are accepted
  verbatim. If you want `matches_specialization` to report cleanly, use
  labels that match what the tier YAMLs advertise.

## Router model (important — this surprised the original implementation)

`vllm-sr serve` is **a launcher, not a daemon**. It brings up a Docker
stack (router + envoy + dashboard + simulator + datastores +
observability) and exits cleanly with code 0. The router service lives
in those background containers, managed by the host `vllm-sr` CLI.

- Exit 0 from `vllm-sr serve` = success, not crash.
- `vllm-sr stop` tears the stack down.
- The harness does NOT tear down on exit by default; user controls the
  long-lived lifecycle via `make router-stop`.
- Routing decision is in three response headers on 2xx-non-cached
  responses: `x-vsr-selected-model`, `x-vsr-selected-category`,
  `x-vsr-selected-reasoning`.

## Test discipline

- 52 tests, ~2 seconds. Run `make test` before commits.
- `make lint` is ruff; expected to be clean.
- `tests/test_shipped_configs.py` asserts every YAML/JSON config file
  actually parses with the real loaders. This is the regression gate
  for "spec name drift" and other config-source-of-truth bugs.

## Working alongside another Claude session

Another Claude instance may be working on this repo from a different
machine (e.g., dogfooding on the server while a dev Claude works on
the laptop).

- **Sync via git, not chat.** Pull before editing. Don't both have
  uncommitted changes at the same time.
- **Commit findings, not just code.** If you run diagnostics and learn
  something important (e.g. "envoy.yaml needs an explicit routes
  block"), update PLAN.md or write a short note in your commit message
  so the other session can read it.
- **Honest commit messages.** State what you changed and why in 2-4
  sentences. The other Claude will trust your reasoning and not
  re-derive it.

## First-launch model download (operational gotcha)

`vllm-sr serve` downloads its routing **embedding model**
(`llm-semantic-router/mmbert-embed-32k-2d-matryoshka`, ~600MB) from
HuggingFace on first launch, inside the router container, before the
apiserver reports ready. Two traps we hit and fixed:

- **HF Xet is firewalled on many networks.** The model is Xet-backed,
  and the Xet CDN (`cas-bridge.xethub.hf.co`) 403s even when the HF
  metadata API and classic LFS CDN are reachable — so the in-container
  download dies with a misleading "does not seem to be on
  huggingface.co". vllm-sr forwards only a fixed HF allowlist
  (`HF_TOKEN`/`HF_ENDPOINT`/`HF_HOME`) into the container, **not**
  `HF_HUB_DISABLE_XET`, so you can't fix it via env. The fix is
  **`make fetch-router-model`** (a `make route` prereq; the UI's Apply
  calls the same logic): it pre-seeds the model into the bind-mounted
  `config/models` with Xet disabled, so the router finds it present and
  skips the download. `config/models/` is gitignored.
- **The download (~2.5 min) overran `ready_timeout_s`.** Bumped to 900s
  in `config/router.yaml`. Once seeded, startup is fast.

Also: the router's OpenAI frontend (Envoy ingress) is **:8899**, not
8801 — that's the URL the live-demo overlay and `router_client` use.

## vllm-sr version (pinned to v0.3.0 — important)

The harness emits the **v0.3** config schema, and requires the matching
**v0.3.0** vllm-sr release (CLI + container image). **v0.2.0 silently
breaks**: a plain `pip install vllm-sr` resolves to 0.2.0, whose launcher
does NOT start the postgres/redis backends the router's config requires —
so the ExtProc is left half-wired and Envoy **resets every routed request**
(HTTP 000, no router log). The symptom looks like a routing/network failure
but is purely the version. v0.3.0's launcher logs `Storage backends
required by config: postgres, redis → Starting…` and wires the ExtProc
correctly. The pin lives in three kept-in-sync places: `VLLM_SR_VERSION`
in the Makefile (CLI install + `VLLM_SR_IMAGE`), the `--image` in
`config/router.yaml` serve_args (the `make route` path), and `VLLM_SR_IMAGE`
in `tools/interactive_server.py` (the UI Apply path). `make setup` warns if
an older CLI is already installed. (The `RouterDC … mmbert status -1` and
`embedding_ready:false` log lines are harmless even on v0.3.0 — they're the
unused tools/RouterDC path; projection-based routing classifies fine.)

Two more v0.3 behaviors that cost real debugging time:

- **Every `routing.decisions` entry needs an `algorithm` block.** Ours
  omitted it; without it the per-decision selector won't resolve the model
  and `selection_method` stays empty. `build_router_config` now emits
  `algorithm: {type: static}` on every decision (one model per tier).
- **vllm-sr forwards the model card `name` upstream — NOT
  `provider_model_id`.** So a card named `tier2` makes a real provider 404
  with "model tier2 does not exist". The mock accepts any name, hiding this.
  `build_router_config --served-model-names real` renames the cards (and
  modelRefs) to the real model id; the live demo's Apply passes it. `make
  route` / the benchmark keep tier-id names (default `tier`), so
  `x-vsr-selected-model` still speaks tier ids there. On the live path the
  header is the real model id, so `interactive_server` maps it back to a tier
  by `id OR model`, and pinning a tier sends the tier's real model name.

## Current state (as of last commit)

End-to-end mock pipeline works: `make setup` → `make load` →
`make route` → `make answers` → `make export` produces a complete
`data/routed_queries_with_answers.json`. Router is driven by the v0.3 canonical projections pattern
(`routing.signals.complexity[]` + `routing.projections.{scores,mappings}`
+ per-tier decisions), compiled from `config/router-exemplars.yaml` into
`config/router-config.yaml`. Lexical/keyword routing has been removed —
semantic only.

**Multi-model tiers (current):** each non-top tier can front several
models via indexed env slots — `TIER{N}_1_*`, `TIER{N}_2_*`, … (i starts
at 1; no bare/slot-0 form, the loader raises on stale single-model
vars). Each slot takes an optional `PROVIDER` label. `make answers`
calls EVERY model in the routed tier — one `tier_answers` row per (run, query, tier, model),
PK `(run_id, query_id, tier_level, model_id)`. **Top-tier shortcut:**
the top tier is the gold reference (every comparison is routed-vs-top,
never top-vs-top), so queries routed to the top tier are SKIPPED by
`make answers` — no model calls. The per-provider gold set lives in the
`gold_answers` table: an `upstream` row from queries.json at
`make load`, plus per-model rows from `make update-gold` (which DOES
call every top-tier model) and `make import-answers`. `data/routed_queries_with_answers.json` carries `expected_answers[]`,
`routed_answers[]`, and per-tier lists of `{provider, model, answer}`.
The DB schema changed: a fresh DB or `make clean-results` + reseed is
required (no migration script).
