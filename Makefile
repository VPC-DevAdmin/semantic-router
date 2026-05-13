# Semantic Router Benchmark Harness
#
# Workflow:
#   make setup          # one-time: venv, deps, empty DB
#   make load           # load data/queries.json into DB (idempotent)
#   make route          # send queries through the router, capture routing decisions
#   make answer         # send queries through the router for full LLM responses
#   make judge          # LLM-as-judge scoring against gold
#   make review REVIEWER=alice   # human scoring TUI
#   make report         # aggregate stats; pass JSON=path or CSV=path to export

.PHONY: help setup load route answer resume judge review report \
        clean-results router-smoke test fmt lint

VENV := .venv
PYTHON := $(VENV)/bin/python
BENCHMARK := $(VENV)/bin/benchmark
DB := benchmark.db

HAS_UV := $(shell command -v uv 2>/dev/null)

help:
	@echo "Targets:"
	@echo "  setup                    venv + deps + init DB"
	@echo "  load                     load data/queries.json into DB (idempotent)"
	@echo "  route                    pass 1: routing decisions (no LLM generation)"
	@echo "  answer [RUN=<id>]        pass 2: full LLM responses (resumable)"
	@echo "  resume [RUN=<id>]        re-run pending/error rows; mark done if clean"
	@echo "  judge [RUN=<id>]         LLM-as-judge scoring of answers"
	@echo "  review REVIEWER=<id>     human scoring TUI"
	@echo "  report [RUN=<id>] [JSON=<path>] [CSV=<path>]"
	@echo "  clean-results            wipe runs/results/scores; preserves queries"
	@echo "  router-smoke PROMPT='...'  diagnostic: one query through the router"
	@echo "  test / fmt / lint"

# ---- one-time setup ----

$(VENV)/bin/python:
ifdef HAS_UV
	uv venv $(VENV)
else
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
endif

setup: $(VENV)/bin/python
ifdef HAS_UV
	uv pip install --python $(PYTHON) -e ".[dev]"
else
	$(VENV)/bin/pip install -e ".[dev]"
endif
	$(BENCHMARK) init-db --db $(DB)

# ---- data ----

load:
	$(BENCHMARK) load --db $(DB)

# ---- benchmark passes ----

route:
	$(BENCHMARK) route --db $(DB)

answer:
	$(BENCHMARK) answer --db $(DB) $(if $(RUN),--run $(RUN),)

resume:
	$(BENCHMARK) resume --db $(DB) $(if $(RUN),--run $(RUN),)

# ---- scoring ----

REVIEWER ?= $(USER)
judge:
	$(BENCHMARK) judge $(if $(RUN),--run $(RUN),)

review:
	$(BENCHMARK) review --reviewer $(REVIEWER) $(if $(RUN),--run $(RUN),) $(if $(SAMPLE),--sample $(SAMPLE),)

# ---- reporting ----

report:
	$(BENCHMARK) report $(if $(RUN),--run $(RUN),) $(if $(JSON),--json $(JSON),) $(if $(CSV),--csv $(CSV),)

# ---- utility ----

clean-results:
	$(BENCHMARK) clean-results --db $(DB)

PROMPT ?= What is 2+2?
router-smoke:
	$(BENCHMARK) router-smoke "$(PROMPT)"

test:
	$(VENV)/bin/pytest

fmt:
	$(VENV)/bin/ruff format src tests
	$(VENV)/bin/ruff check --fix src tests

lint:
	$(VENV)/bin/ruff check src tests
