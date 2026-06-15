# Semantic Routing Demo — Makefile
#
# Production-pass workflow:
#   make setup        # one-time: venv, deps, DB, vllm-sr binary
#   make load         # data/queries.json → DB (idempotent)
#   make route        # for each query: ask the router which tier it picks
#   make answers      # for each query × tier: collect that tier's response  [TODO]
#   make export       # write data/routed_queries_with_answers.json

.PHONY: help setup install-vllm-sr-pypi load route answers evaluate export resume misroutes scores \
        import-answers update-gold demo-data demo gateway interactive \
        clean-results router-smoke router-stop test fmt lint \
        mock-bg mock-stop start_LLM stop_LLM

VENV := .venv
PYTHON := $(VENV)/bin/python
BENCHMARK := $(VENV)/bin/benchmark
DB := data/router_benchmark.db

# For splitting comma-separated make args (e.g. QID=q1,q2,q3).
comma := ,
space :=
space +=

HAS_UV := $(shell command -v uv 2>/dev/null)

help:
	@echo "Production pass:"
	@echo "  setup                          venv + deps + init DB + install vllm-sr (if missing)"
	@echo "  load [MOCK=true]               validate exemplars; build router-config; load queries.json into DB"
	@echo "  route [RUN_NEW=true]           rebuild router-config; routing pass via local OAI mock"
	@echo "  answers [MOCK=true] [RUN=<id>] [RUN_NEW=true] [TIER=<1-5>] [CONC=<N>] [MAXTOK=<N>] [SMOKE=true]  routed-tier answers (SMOKE: connectivity probe only)"
	@echo "  evaluate [RUN=<id>] [BATCH=<N>] [RUN_NEW=true]   LLM-judge routed vs gold, batched (default 50 queries/call)"
	@echo "  import-answers FILE=<path> TIER=<1-5> MODEL=<id> [PROVIDER=<name>] [RUN=<id>]  load externally-generated answers for one model"
	@echo "  update-gold [QID=<id[,id]>] [TIER=<1-5>] [YES=true]  regenerate gold via top tier (no scope = ALL, confirms)"
	@echo "  export [RUN=<id>] [OUTPUT=<path>]  emit the routed-queries JSON (default: data/routed_queries_with_answers.json)"
	@echo "  demo-data [CONC=<N>]           force-rebuild demo/data/demo_data.json from the exports + demo/pricing.json"
	@echo "  demo [DEMO_PORT=<n>]           serve the cost-routing replay demo + open browser (single command, no setup needed)"
	@echo "  gateway [GATEWAY_PORT=<n>] [ROUTER_URL=<url>]  OpenAI-compatible contract gateway for agent orchestrators"
	@echo "  interactive [INTERACTIVE_PORT=<n>]  chat UI: type a query, watch it routed across tiers + get answers (keys in Settings)"
	@echo "  misroutes [RUN=<id>]           diagnostic: list queries routed BELOW their min tier"
	@echo "  scores [RUN=<id>]              diagnostic: per-signal score + threshold gap for each misroute"
	@echo "  resume [RUN=<id>]              re-run pending/error rows; mark done if clean"
	@echo ""
	@echo "  MOCK=true (load/answers)   routes the build/answers through the local OAI mock (port \$$MOCK_PORT)."
	@echo "                             (route ALWAYS uses the mock — routing pass doesn't need real upstreams.)"
	@echo "              Used for pipeline verification before real backends are stood up."
	@echo ""
	@echo "Backends:"
	@echo "  mock-bg                        start the local OAI mock (port \$$MOCK_PORT, default 18811)"
	@echo "  mock-stop                      stop the local OAI mock"
	@echo "  start_LLM                      launch local-CPU tier backends per config/tiers/*.yaml"
	@echo "  stop_LLM                       stop local-CPU tier backends"
	@echo ""
	@echo "Utility:"
	@echo "  clean-results                  wipe runs/results; preserves queries + gold"
	@echo "  router-smoke PROMPT='...'      diagnostic: one query through the router"
	@echo "  router-stop                    tear down the vllm-sr Docker stack"
	@echo "  install-vllm-sr-pypi           install vllm-sr direct from PyPI (used as"
	@echo "                                 fallback by 'make setup' when the upstream"
	@echo "                                 install.sh endpoint is unreachable)"
	@echo "  test / fmt / lint"

# ---- one-time setup ----

$(VENV)/bin/python:
ifdef HAS_UV
	uv venv $(VENV)
else
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
endif

# `make setup` does four things:
#   1. create the venv (uv if available, else stdlib venv)
#   2. install our Python package in editable mode
#   3. install the vllm-sr binary if it's not already on PATH
#   4. initialize the SQLite schema
#
# Override the router-install source with VLLM_SR_INSTALL_URL=..., or skip
# the install step entirely with SKIP_ROUTER_INSTALL=1 (use this if you
# manage vllm-sr some other way — system package, container image, etc.).
VLLM_SR_INSTALL_URL ?= https://vllm-semantic-router.com/install.sh

# Fallback install location (mirrors what upstream install.sh uses) for the
# PyPI-direct path. Used only when the upstream install.sh endpoint is
# unreachable. Override these to relocate.
VLLM_SR_VENV    ?= $(HOME)/.local/share/vllm-sr/venv
VLLM_SR_BIN_DIR ?= $(HOME)/.local/bin

setup: $(VENV)/bin/python
ifdef HAS_UV
	uv pip install --python $(PYTHON) -e ".[dev]"
else
	$(VENV)/bin/pip install -e ".[dev]"
endif
ifdef SKIP_ROUTER_INSTALL
	@echo "[setup] skipping vllm-sr install (SKIP_ROUTER_INSTALL is set)"
else
	@if command -v vllm-sr >/dev/null 2>&1; then \
	    echo "[setup] vllm-sr already present: $$(command -v vllm-sr)"; \
	else \
	    echo "[setup] installing vllm-sr from $(VLLM_SR_INSTALL_URL)"; \
	    curl -fsSL $(VLLM_SR_INSTALL_URL) | bash || true; \
	    if ! command -v vllm-sr >/dev/null 2>&1; then \
	        echo ""; \
	        echo "[setup] upstream installer did not produce vllm-sr on PATH."; \
	        echo "[setup] falling back to PyPI (pip install vllm-sr)..."; \
	        echo ""; \
	        $(MAKE) -s install-vllm-sr-pypi || true; \
	    fi; \
	    if command -v vllm-sr >/dev/null 2>&1; then \
	        echo "[setup] installed: $$(command -v vllm-sr)"; \
	    else \
	        echo ""; \
	        echo "[setup] WARN: vllm-sr install did not complete and the PyPI"; \
	        echo "        fallback also did not put it on PATH. vllm-sr is"; \
	        echo "        REQUIRED for \`make route\` -- the core routing pass"; \
	        echo "        this benchmark is built around. Setup will finish"; \
	        echo "        so you can still explore the committed dataset"; \
	        echo "        (\`make demo\`) and operate on already-routed rows"; \
	        echo "        (\`make answers\`, \`make evaluate\`, \`make export\`),"; \
	        echo "        but a fresh routing pass needs vllm-sr."; \
	        echo "        If $(VLLM_SR_BIN_DIR) exists but isn't on your PATH,"; \
	        echo "        add it via your shell rc and re-run \`make setup\`."; \
	        echo ""; \
	    fi; \
	fi
endif
	$(BENCHMARK) init-db --db $(DB)
	@echo ""
	@echo "[setup] complete. DB initialized at $(DB)."
	@if command -v vllm-sr >/dev/null 2>&1; then \
	    echo "[setup] vllm-sr ready."; \
	else \
	    echo "[setup] vllm-sr NOT installed -- \`make route\` (the routing pass)"; \
	    echo "        is blocked until you install it. Everything else works."; \
	    echo "        Re-run \`make setup\` to retry, or install manually."; \
	fi
	@# vllm-sr brings up a Docker stack (router + envoy + datastores), so the
	@# routing pass + the live interactive demo need a reachable Docker daemon.
	@# The cost replay (`make demo`), answers/evaluate/export do NOT.
	@if docker info >/dev/null 2>&1; then \
	    echo "[setup] Docker ready -- \`make route\` and the live demo can launch vllm-sr."; \
	else \
	    echo "[setup] Docker NOT reachable -- REQUIRED for \`make route\` and the live"; \
	    echo "        interactive demo (vllm-sr runs as a Docker stack). Not needed for"; \
	    echo "        \`make demo\` (cost replay), answers, evaluate, or export."; \
	    echo "        Start it:  sudo systemctl start docker   (then add your user to"; \
	    echo "        the docker group, re-login, and verify with \`docker ps\`)."; \
	fi

# Direct-from-PyPI install of vllm-sr. Mirrors what upstream install.sh
# does: isolated venv (so vllm-sr's deps don't conflict with this repo's
# pinned versions) + a launcher symlink on PATH. Used by `make setup` as
# a fallback when the upstream install.sh endpoint is unreachable; can
# also be invoked directly:
#     make install-vllm-sr-pypi
install-vllm-sr-pypi:
	@if ! command -v python3 >/dev/null 2>&1; then \
	    echo "[install-vllm-sr-pypi] ERROR: python3 not found on PATH."; \
	    exit 1; \
	fi
	@mkdir -p $(VLLM_SR_BIN_DIR)
	@if [ ! -x $(VLLM_SR_VENV)/bin/python ]; then \
	    echo "[install-vllm-sr-pypi] creating venv at $(VLLM_SR_VENV)"; \
	    python3 -m venv $(VLLM_SR_VENV); \
	fi
	@echo "[install-vllm-sr-pypi] installing vllm-sr from PyPI..."
	@$(VLLM_SR_VENV)/bin/python -m pip install --disable-pip-version-check --quiet --upgrade pip wheel setuptools
	@$(VLLM_SR_VENV)/bin/python -m pip install --disable-pip-version-check --quiet --upgrade vllm-sr
	@ln -sf $(VLLM_SR_VENV)/bin/vllm-sr $(VLLM_SR_BIN_DIR)/vllm-sr
	@echo "[install-vllm-sr-pypi] installed: $(VLLM_SR_BIN_DIR)/vllm-sr -> $(VLLM_SR_VENV)/bin/vllm-sr"
	@if ! command -v vllm-sr >/dev/null 2>&1; then \
	    echo "[install-vllm-sr-pypi] NOTE: $(VLLM_SR_BIN_DIR) is not on your"; \
	    echo "        current PATH. Add it to your shell rc (e.g. .bashrc):"; \
	    echo "          export PATH=\"$(VLLM_SR_BIN_DIR):\$$PATH\""; \
	    echo "        Then open a new shell or 'source' the rc and re-run setup."; \
	fi

# ---- router config (exemplar-based) ----
#
# Both `load` and `route` build config/router-config.yaml from
# config/router-exemplars.yaml + config/router-backends.yaml. The build
# step also validates that no exemplar prompt overlaps the eval set in
# data/queries.json — if it does, the build fails fast and the parent
# target is aborted.
EXEMPLARS := config/router-exemplars.yaml
BACKENDS := config/router-backends.yaml
ROUTER_CONFIG := config/router-config.yaml

# Mock URLs are derived from MOCK_PORT (defined in the mock section below).
# Two forms because the router (inside Docker) and the harness (on the host)
# reach the same mock at different addresses.
# Default well outside the typical 8000-8099 range (vllm 8000-8003,
# vllm-sr dashboard 8700, envoy 8899). Override on the command line
# (`make route MOCK_PORT=N`) if 18811 is taken on your machine.
MOCK_PORT ?= 18811
MOCK_FROM_ROUTER := host.docker.internal:$(MOCK_PORT)/v1
MOCK_FROM_HOST   := http://localhost:$(MOCK_PORT)/v1

# `MOCK=true` reroutes everything through the local OAI mock — used to
# verify the full pipeline before real backends are stood up. Applies to
# `make load`, `make route`, and `make answers`.
define BUILD_ROUTER_CONFIG
$(PYTHON) -m benchmark.build_router_config \
    --exemplars $(EXEMPLARS) \
    --backends $(BACKENDS) \
    --out $(ROUTER_CONFIG) \
    --check-against-eval data/queries.json \
    $(if $(filter true,$(MOCK)),--mock-endpoint $(MOCK_FROM_ROUTER),)
endef

# ---- data ----

# `make load` validates exemplars and (re)builds router-config.yaml as
# part of the queries.json → DB load. Overlap between exemplars and the
# eval set aborts the load.
load:
	$(BUILD_ROUTER_CONFIG)
	$(BENCHMARK) load --db $(DB)

# ---- production pass ----

# `make route` only needs the router's decision (x-vsr-selected-* headers),
# not real upstream completions. The router is an Envoy proxy: every
# routed request gets forwarded upstream, so without a mock we'd pay
# 1 completion token × ~110 queries × every per-vendor quirk
# (max_tokens vs max_completion_tokens, temperature=0 on gpt-5, etc.).
#
# `make route` routes every tier through the local OAI mock unconditionally.
# The mock ACKs cheaply with a tier-tagged canned reply while the
# routing classifier + decision rules still run end-to-end. The
# expensive real model calls happen in `make answers`, which bypasses
# the router and calls every model in the routed tier directly.
#
# RUN_NEW=true → drop pass1_results for the active run; re-seed.
#
# Depends on mock-bg so the mock is guaranteed running before the build
# emits a router-config that points at it. mock-bg is idempotent.
route: mock-bg
	@if ! command -v vllm-sr >/dev/null 2>&1; then \
	    echo "[route] ERROR: vllm-sr is not on PATH."; \
	    echo "        This target runs the routing pass through vllm-sr,"; \
	    echo "        which must be installed first. Run \`make setup\` to"; \
	    echo "        install it (re-run if the upstream install URL was"; \
	    echo "        previously unreachable), or install it manually from"; \
	    echo "        vllm-project/semantic-router releases and place it on PATH."; \
	    exit 1; \
	fi
	$(PYTHON) -m benchmark.build_router_config \
	    --exemplars $(EXEMPLARS) --backends $(BACKENDS) --out $(ROUTER_CONFIG) \
	    --check-against-eval data/queries.json \
	    --mock-endpoint $(MOCK_FROM_ROUTER)
	$(BENCHMARK) route --db $(DB) $(if $(filter true,$(RUN_NEW)),--run-new,)

# RUN_NEW=true → drop existing tier_answers for the active run and re-seed.
#                With TIER=<N>, only that tier's rows are deleted.
# MOCK=true    → answers calls every tier via the local mock URL.
# TIER=<N>     → restrict the worker to one tier level (1-5). Other tiers'
#                pending/error rows are left untouched; useful for exercising
#                a just-wired backend without re-hitting other tiers.
# CONC=<N>     → concurrency (parallel requests). Default 8 in the CLI.
#                Lower for CPU-bound local tiers (try 1-2) where parallel
#                requests saturate the cores. Higher (16-32) for vendor APIs.
# MAXTOK=<N>   → max_tokens per response (default 2048). Lower (e.g. 768)
#                caps worst-case CPU wall-clock so slow generations don't
#                hit the read timeout.
# Queries the router sent to the top tier are SKIPPED — no model calls.
# The top tier is the gold reference; comparisons are routed-vs-top,
# never top-vs-top. Its per-provider answers come from `make update-gold`
# and from `expected_answers[]` declared in queries.json.
#
# SMOKE=true   → connectivity probe only: tiny chat request to every
#                (tier, model) `make answers` would call, report OK/error
#                per endpoint, exit non-zero on any failure. No DB writes.
#                Use this BEFORE a real run to catch wrong URLs / bad API
#                keys / unknown model names. Honors TIER=<N> (probe just
#                that tier; useful for testing Tier 5 / update-gold creds).
answers:
	$(BENCHMARK) answers --db $(DB) \
	    $(if $(RUN),--run $(RUN),) \
	    $(if $(TIER),--tier $(TIER),) \
	    $(if $(CONC),--concurrency $(CONC),) \
	    $(if $(MAXTOK),--max-tokens $(MAXTOK),) \
	    $(if $(filter true,$(RUN_NEW)),--run-new,) \
	    $(if $(filter true,$(SMOKE)),--smoke,) \
	    $(if $(filter true,$(MOCK)),--mock-endpoint $(MOCK_FROM_HOST),)

# Judge the routed answers against the gold answers using one or more
# LLM evaluators configured via EVALUATOR_N_* env vars. Per-row resumable.
#
# RUN_NEW=true     → drop existing evaluation rows for the active run; re-seed.
# BATCH=<N>        → queries per judge call (default 50). Each query packs
#                    its (routed × gold) pairs into the same call, so the
#                    actual evaluations per call is ~6× this. Lower if
#                    your judge runs out of context or max_tokens budget.
#
# Each EVALUATOR_N_* slot is independent. Add as many as you want
# (EVALUATOR_1, EVALUATOR_2, …); see .env.example for the shape.
evaluate:
	$(BENCHMARK) evaluate --db $(DB) \
	    $(if $(RUN),--run $(RUN),) \
	    $(if $(BATCH),--batch-size $(BATCH),) \
	    $(if $(filter true,$(RUN_NEW)),--run-new,)

# Import externally-generated answers (e.g., manually prompted from a
# chat UI) into the tier_answers table, attributed to one model.
#   FILE=<path>      — markdown file with `## qNNNNN — Title` sections
#   TIER=<N>         — tier level (1-5) these answers represent
#   MODEL=<id>       — the model id these answers are from (per-tier key)
#   PROVIDER=<name>  — optional label (Anthropic/OpenAI/Google) → data/routed_queries_with_answers.json
# Idempotent: re-run to refresh the same (tier, model) rows.
import-answers:
	@if [ -z "$(FILE)" ] || [ -z "$(TIER)" ] || [ -z "$(MODEL)" ]; then \
	  echo "usage: make import-answers FILE=<path.md> TIER=<1-5> MODEL=<id> [PROVIDER=<name>]"; exit 2; \
	fi
	$(BENCHMARK) import-answers $(FILE) --tier $(TIER) --model $(MODEL) \
	    $(if $(PROVIDER),--provider $(PROVIDER),) \
	    $(if $(RUN),--run $(RUN),)

# Regenerate per-provider gold by calling EVERY top-tier model. Upserts
# one `gold_answers` row per (query, top-tier model). Scope, most → least
# specific:
#   QID=<id[,id,...]>  — just those queries
#   TIER=<1-5>         — every query with that expected_min_tier
#   (neither)          — ALL queries (prompts to confirm unless YES=true)
# `make answers` doesn't depend on this — top-tier-routed queries are
# already skipped — but downstream consumers read these rows as the
# per-provider `expected_answers[]` in data/routed_queries_with_answers.json.
update-gold:
	$(BENCHMARK) update-gold --db $(DB) \
	    $(foreach q,$(subst $(comma), ,$(QID)),--query-id $(q)) \
	    $(if $(TIER),--tier $(TIER),) \
	    $(if $(filter true,$(YES)),--yes,)

OUTPUT ?= data/routed_queries_with_answers.json
export:
	$(BENCHMARK) export --db $(DB) --output $(OUTPUT) $(if $(RUN),--run $(RUN),)

# ---- cost-routing replay demo ----
# Design goal: `make demo` is a SINGLE COMMAND that works on a bare clone
# with NO `make setup` — the committed demo dataset + a stdlib-only
# preprocessor + stdlib http.server mean it needs nothing but python3.
#
# DEMO_PY picks the venv python if `make setup` has run, else falls back
# to system python3. The demo never touches vllm-sr / the dev toolchain,
# so we deliberately don't force `make setup` (which would download the
# vllm-sr binary the demo doesn't use).
#
# `make demo` serves the COMMITTED demo_data.json as-is unless the source
# exports are newer (i.e. you re-ran the pipeline), in which case it
# rebuilds first — so a replicator's run shows their data while a fresh
# clone shows the canonical data. `make demo-data` forces a rebuild.
DEMO_PORT ?= 8000
ROUTED_JSON := data/routed_queries_with_answers.json
EVALS_JSON  := data/evaluations.json
DEMO_DATA   := demo/data/demo_data.json
DEMO_PY     := $(shell [ -x $(VENV)/bin/python ] && echo $(VENV)/bin/python || command -v python3)
# Cross-platform "open a URL in the default browser" (no-op if neither found).
OPEN_CMD    := $(shell command -v open 2>/dev/null || command -v xdg-open 2>/dev/null || echo true)

# Force a rebuild of the demo dataset from whatever exports are present.
demo-data:
	@if [ ! -f $(ROUTED_JSON) ] || [ ! -f $(EVALS_JSON) ]; then \
	    echo "[demo] source export(s) missing — running 'make export' to regenerate from the DB"; \
	    $(MAKE) export; \
	fi
	$(DEMO_PY) tools/build_demo_data.py $(if $(CONC),--concurrency $(CONC),)

demo:
	@if [ -z "$(DEMO_PY)" ]; then \
	    echo "[demo] no python3 found. Install Python 3, or run 'make setup'."; exit 1; \
	fi
	@# Rebuild only if the dataset is missing or the source exports changed
	@# (a fresh clone serves the committed canonical dataset untouched).
	@if [ ! -f $(DEMO_DATA) ] || [ $(ROUTED_JSON) -nt $(DEMO_DATA) ] \
	     || [ demo/pricing.json -nt $(DEMO_DATA) ]; then \
	    echo "[demo] building demo dataset ($(DEMO_PY))"; \
	    $(MAKE) demo-data; \
	else \
	    echo "[demo] using committed demo dataset (run 'make demo-data' to force a rebuild)"; \
	fi
	@echo "Serving demo at http://localhost:$(DEMO_PORT)/  (Ctrl-C to stop)"
	@( sleep 1 && $(OPEN_CMD) http://localhost:$(DEMO_PORT)/ >/dev/null 2>&1 & ) || true
	@$(DEMO_PY) tools/demo_server.py $(DEMO_PORT) --directory demo

# ---- contract gateway ----
# An OpenAI-compatible front door that adapts this repo's semantic router to the
# role-based contract an agent orchestrator expects (role names, x-llm-* headers,
# metadata.min_tier floor, strict structured output). Additive — does NOT change
# the router or the standalone demo. Standalone by default (zero real backends);
# pass ROUTER_URL=http://localhost:8801 to classify the worker via real vllm-sr.
GATEWAY_PORT ?= 8800
gateway:
	$(DEMO_PY) tools/router_gateway.py --port $(GATEWAY_PORT) \
	    $(if $(ROUTER_URL),--router-url $(ROUTER_URL),)

# ---- interactive routing demo ----
# A chat UI where you type a query, watch it scored + routed across tiers, and
# (with API keys added in Settings) get an answer. Tiers/models/exemplars/keys
# are all editable in one settings panel. Routing works with NO keys; answers
# need them. Separate from `make demo` (the cost replay) — both coexist.
INTERACTIVE_PORT ?= 8900
interactive:
	@echo "Interactive demo at http://localhost:$(INTERACTIVE_PORT)/  (Ctrl-C to stop)"
	@( sleep 1 && $(OPEN_CMD) http://localhost:$(INTERACTIVE_PORT)/ >/dev/null 2>&1 & ) || true
	@$(DEMO_PY) tools/interactive_server.py --port $(INTERACTIVE_PORT)

resume:
	$(BENCHMARK) resume --db $(DB) $(if $(RUN),--run $(RUN),)

# Diagnostic: show every query the router under-tiered for the latest run.
# Use this BEFORE tuning thresholds in config/router-exemplars.yaml — patterns
# in the output (e.g. all T4-expected queries routed to T2 with category=business)
# tell us where to nudge.
misroutes:
	$(BENCHMARK) misroutes --db $(DB) $(if $(RUN),--run $(RUN),)

# Deeper diagnostic: for each misroute, fetch per-signal scores from
# vllm-sr's /api/v1/eval endpoint. Requires the router stack to be up.
# Shows whether under-routes "just barely missed" (small negative gap) or
# "wildly missed" (large negative gap) the relevant signal threshold.
scores:
	$(BENCHMARK) scores --db $(DB) $(if $(RUN),--run $(RUN),)

# ---- utility ----

clean-results:
	$(BENCHMARK) clean-results --db $(DB)

PROMPT ?= What is 2+2?
router-smoke:
	$(BENCHMARK) router-smoke "$(PROMPT)"

# Tear down the vllm-sr Docker stack. Useful when the router got into a bad
# state (e.g. setup-mode bootstrap) and you want a clean slate before
# `make route` re-launches it with the checked-in config.
router-stop:
	vllm-sr stop

# ---- local mock backend ----
#
# Stands in for real LLM tier endpoints so the pipeline (route + answers +
# export) can be validated without spending tokens. Stdlib-only; no extra
# deps. Returns tier-tagged canned text so we can verify which tier served
# each row. MOCK_PORT is defined near the top alongside the MOCK=true URLs.

MOCK_LOG := logs/mock.log
MOCK_PID := logs/mock.pid

# All in one shell — `exit 0` in the "already running" branch only exits the
# subshell of that line, so the previous split-recipe version always tried to
# start a second mock and reported a spurious FAILED.
mock-bg:
	@mkdir -p logs
	@if [ -f $(MOCK_PID) ] && kill -0 $$(cat $(MOCK_PID)) 2>/dev/null; then \
	    echo "[mock] already running (pid $$(cat $(MOCK_PID)))"; \
	else \
	    nohup $(PYTHON) tools/oai_mock.py --port $(MOCK_PORT) > $(MOCK_LOG) 2>&1 & echo $$! > $(MOCK_PID); \
	    sleep 1; \
	    if kill -0 $$(cat $(MOCK_PID)) 2>/dev/null; then \
	        echo "[mock] listening on :$(MOCK_PORT) (pid $$(cat $(MOCK_PID))), log: $(MOCK_LOG)"; \
	    else \
	        echo "[mock] FAILED to start; see $(MOCK_LOG)"; \
	        rm -f $(MOCK_PID); \
	        exit 1; \
	    fi; \
	fi

mock-stop:
	@if [ -f $(MOCK_PID) ] && kill -0 $$(cat $(MOCK_PID)) 2>/dev/null; then \
	    kill $$(cat $(MOCK_PID)); \
	    rm -f $(MOCK_PID); \
	    echo "[mock] stopped"; \
	else \
	    echo "[mock] not running"; \
	    rm -f $(MOCK_PID); \
	fi

# ---- local-CPU tier backends ----
#
# `start_LLM` iterates config/tiers/*.yaml and brings up every tier whose
# `backend.kind` has a registered launcher. `stop_LLM` does the inverse.
# All knobs (image, NUMA pinning, KV size, ports, served-model-name) live
# in the per-tier YAML — edit those, not this Makefile.

start_LLM:
	$(BENCHMARK) start-llm

stop_LLM:
	$(BENCHMARK) stop-llm

test:
	$(VENV)/bin/pytest

fmt:
	$(VENV)/bin/ruff format src tests
	$(VENV)/bin/ruff check --fix src tests

lint:
	$(VENV)/bin/ruff check src tests
