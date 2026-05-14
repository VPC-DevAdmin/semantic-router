# Semantic Routing Demo — Makefile
#
# Production-pass workflow:
#   make setup        # one-time: venv, deps, DB, vllm-sr binary
#   make load         # data/queries.json → DB (idempotent)
#   make route        # for each query: ask the router which tier it picks
#   make answers      # for each query × tier: collect that tier's response  [TODO]
#   make export       # emit demo.json from the DB                            [TODO]

.PHONY: help setup load route answers export resume misroutes scores \
        clean-results router-smoke router-stop test fmt lint \
        mock-bg mock-stop start_LLM stop_LLM

VENV := .venv
PYTHON := $(VENV)/bin/python
BENCHMARK := $(VENV)/bin/benchmark
DB := benchmark.db

HAS_UV := $(shell command -v uv 2>/dev/null)

help:
	@echo "Production pass:"
	@echo "  setup                          venv + deps + init DB + install vllm-sr (if missing)"
	@echo "  load [MOCK=true]               validate exemplars; build router-config; load queries.json into DB"
	@echo "  route [MOCK=true] [RUN_NEW=true]   rebuild router-config; routing pass"
	@echo "  answers [MOCK=true] [RUN=<id>] [RUN_NEW=true] [TIER=<1-5>]  routed-tier answers; errors retry on next run"
	@echo "  export [RUN=<id>] [OUTPUT=<path>]  write demo.json (default: ./demo.json)"
	@echo "  misroutes [RUN=<id>]           diagnostic: list queries routed BELOW their min tier"
	@echo "  scores [RUN=<id>]              diagnostic: per-signal score + threshold gap for each misroute"
	@echo "  resume [RUN=<id>]              re-run pending/error rows; mark done if clean"
	@echo ""
	@echo "  MOCK=true   routes every tier through the local OAI mock (port \$$MOCK_PORT)."
	@echo "              Used for pipeline verification before real backends are stood up."
	@echo ""
	@echo "Backends:"
	@echo "  mock-bg                        start the local OAI mock (port \$$MOCK_PORT, default 8811)"
	@echo "  mock-stop                      stop the local OAI mock"
	@echo "  start_LLM                      launch local-CPU tier backends per config/tiers/*.yaml"
	@echo "  stop_LLM                       stop local-CPU tier backends"
	@echo ""
	@echo "Utility:"
	@echo "  clean-results                  wipe runs/results; preserves queries + gold"
	@echo "  router-smoke PROMPT='...'      diagnostic: one query through the router"
	@echo "  router-stop                    tear down the vllm-sr Docker stack"
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
	    curl -fsSL $(VLLM_SR_INSTALL_URL) | bash; \
	    if command -v vllm-sr >/dev/null 2>&1; then \
	        echo "[setup] installed: $$(command -v vllm-sr)"; \
	    else \
	        echo ""; \
	        echo "[setup] WARN: vllm-sr is not on PATH after install."; \
	        echo "The installer may have placed it in ~/.local/bin or similar."; \
	        echo "Add the install dir to PATH, then re-run \`make setup\`,"; \
	        echo "or set \`binary:\` to the full path in config/router.yaml."; \
	        exit 1; \
	    fi; \
	fi
endif
	$(BENCHMARK) init-db --db $(DB)

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
MOCK_PORT ?= 8811
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

# `make route` rebuilds router-config.yaml first so the router always
# launches with the current exemplars + backends reflected in its config.
# RUN_NEW=true → drop existing pass1_results for the active run and re-seed.
# MOCK=true    → router-config points every tier at the mock.
route:
	$(BUILD_ROUTER_CONFIG)
	$(BENCHMARK) route --db $(DB) $(if $(filter true,$(RUN_NEW)),--run-new,)

# RUN_NEW=true → drop existing tier_answers for the active run and re-seed.
#                With TIER=<N>, only that tier's rows are deleted.
# MOCK=true    → answers calls every tier via the local mock URL.
# TIER=<N>     → restrict the worker to one tier level (1-5). Other tiers'
#                pending/error rows are left untouched; useful for exercising
#                a just-wired backend without re-hitting other tiers.
answers:
	$(BENCHMARK) answers --db $(DB) \
	    $(if $(RUN),--run $(RUN),) \
	    $(if $(TIER),--tier $(TIER),) \
	    $(if $(filter true,$(RUN_NEW)),--run-new,) \
	    $(if $(filter true,$(MOCK)),--mock-endpoint $(MOCK_FROM_HOST),)

OUTPUT ?= demo.json
export:
	$(BENCHMARK) export --db $(DB) --output $(OUTPUT) $(if $(RUN),--run $(RUN),)

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
