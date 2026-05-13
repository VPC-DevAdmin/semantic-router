# Semantic Routing Demo — Makefile
#
# Production-pass workflow:
#   make setup        # one-time: venv, deps, DB, vllm-sr binary
#   make load         # data/queries.json → DB (idempotent)
#   make route        # for each query: ask the router which tier it picks
#   make answers      # for each query × tier: collect that tier's response  [TODO]
#   make export       # emit demo.json from the DB                            [TODO]

.PHONY: help setup install-router load route answers export resume \
        clean-results router-smoke router-stop test fmt lint

VENV := .venv
PYTHON := $(VENV)/bin/python
BENCHMARK := $(VENV)/bin/benchmark
DB := benchmark.db

HAS_UV := $(shell command -v uv 2>/dev/null)

help:
	@echo "Targets:"
	@echo "  setup                    venv + deps + init DB + install vllm-sr (if missing)"
	@echo "  install-router           install the vllm-sr binary (idempotent)"
	@echo "  load                     load data/queries.json into DB (idempotent)"
	@echo "  route                    for each query: capture the router's tier pick"
	@echo "  answers [RUN=<id>]       [TODO] for each query × tier: capture that tier's response"
	@echo "  export [RUN=<id>]        [TODO] write demo.json from DB"
	@echo "  resume [RUN=<id>]        re-run pending/error rows; mark done if clean"
	@echo "  clean-results            wipe runs/results; preserves queries + gold"
	@echo "  router-smoke PROMPT='...'  diagnostic: one query through the router"
	@echo "  router-stop              tear down the vllm-sr Docker stack"
	@echo "  test / fmt / lint"

# ---- one-time setup ----

$(VENV)/bin/python:
ifdef HAS_UV
	uv venv $(VENV)
else
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
endif

setup: $(VENV)/bin/python install-router
ifdef HAS_UV
	uv pip install --python $(PYTHON) -e ".[dev]"
else
	$(VENV)/bin/pip install -e ".[dev]"
endif
	$(BENCHMARK) init-db --db $(DB)

# Install the vllm-sr binary if it's not already on PATH. Idempotent.
# Override the source by passing VLLM_SR_INSTALL_URL=... or skip entirely with
# SKIP_ROUTER_INSTALL=1 if you manage it some other way.
VLLM_SR_INSTALL_URL ?= https://vllm-semantic-router.com/install.sh
install-router:
ifdef SKIP_ROUTER_INSTALL
	@echo "[install-router] skipped (SKIP_ROUTER_INSTALL is set)"
else
	@if command -v vllm-sr >/dev/null 2>&1; then \
	    echo "[install-router] vllm-sr already present: $$(command -v vllm-sr)"; \
	else \
	    echo "[install-router] installing from $(VLLM_SR_INSTALL_URL)"; \
	    curl -fsSL $(VLLM_SR_INSTALL_URL) | bash; \
	    if command -v vllm-sr >/dev/null 2>&1; then \
	        echo "[install-router] installed: $$(command -v vllm-sr)"; \
	    else \
	        echo ""; \
	        echo "[install-router] WARN: vllm-sr is not on PATH after install."; \
	        echo "The installer may have placed it in ~/.local/bin or similar."; \
	        echo "Add the install dir to PATH, then re-run \`make setup\`,"; \
	        echo "or set \`binary:\` to the full path in config/router.yaml."; \
	        exit 1; \
	    fi; \
	fi
endif

# ---- data ----

load:
	$(BENCHMARK) load --db $(DB)

# ---- production pass ----

route:
	$(BENCHMARK) route --db $(DB)

# `answers` and `export` not yet wired through the CLI; see PLAN.md §13.
answers:
	@echo "TODO: make answers — hit each tier endpoint per query. See PLAN.md §13."
	@exit 1

export:
	@echo "TODO: make export — emit demo.json from DB. See PLAN.md §13."
	@exit 1

resume:
	$(BENCHMARK) resume --db $(DB) $(if $(RUN),--run $(RUN),)

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

test:
	$(VENV)/bin/pytest

fmt:
	$(VENV)/bin/ruff format src tests
	$(VENV)/bin/ruff check --fix src tests

lint:
	$(VENV)/bin/ruff check src tests
