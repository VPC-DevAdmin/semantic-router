# Semantic Router Benchmark Harness
#
# `make setup` and `make seed` are wired in M1.
# Other targets remain stubs until their respective milestones (see PLAN.md).

.PHONY: help setup seed gold run pass1 pass2 review judge report resume \
        clean-results test fmt lint validate-config router-smoke

VENV := .venv
PYTHON := $(VENV)/bin/python
BENCHMARK := $(VENV)/bin/benchmark
QUERIES := data/queries.yaml
DB := benchmark.db

# Prefer uv if available; fall back to stdlib venv + pip.
HAS_UV := $(shell command -v uv 2>/dev/null)

help:
	@echo "Targets:"
	@echo "  setup            venv + deps + init DB"
	@echo "  seed             upsert data/queries.yaml into DB (idempotent)"
	@echo "  validate-config  check config files without touching DB"
	@echo "  gold             [M2] generate/refresh gold answers"
	@echo "  router-smoke     [M3] boot router, send one prompt, print decision"
	@echo "  run              new run_id; boot router; pass1 + pass2; tear down"
	@echo "  pass1 [RUN=<id>] routing accuracy only (resumable)"
	@echo "  pass2 [RUN=<id>] response generation only (resumable)"
	@echo "  resume [RUN=<id>] resume pending/error rows; mark done if clean"
	@echo "  judge [RUN=<id>] LLM-as-judge scoring of pass-2 responses"
	@echo "  review [RUN=<id>] [SAMPLE=N] [REVIEWER=<id>] human scoring TUI"
	@echo "  report [RUN=<id>] [JSON=path] [CSV=path] aggregate stats + export"
	@echo "  clean-results    wipe runs/results/scores; preserves queries + gold"
	@echo "  test / fmt / lint"

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

seed:
	$(BENCHMARK) seed --queries $(QUERIES) --db $(DB)

validate-config:
	$(BENCHMARK) validate-config --queries $(QUERIES)

gold:
	$(BENCHMARK) gold --db $(DB)

# Smoke test: boots router, sends PROMPT, prints decision, tears down.
# Usage: make router-smoke PROMPT='What is 2+2?'
PROMPT ?= What is 2+2?
router-smoke:
	$(BENCHMARK) router-smoke "$(PROMPT)"

run:
	$(BENCHMARK) run --db $(DB)

pass1:
	$(BENCHMARK) pass1 --db $(DB) $(if $(RUN),--run $(RUN),)

pass2:
	$(BENCHMARK) pass2 --db $(DB) $(if $(RUN),--run $(RUN),)

resume:
	$(BENCHMARK) resume --db $(DB) $(if $(RUN),--run $(RUN),)

REVIEWER ?= $(USER)
review:
	$(BENCHMARK) review --reviewer $(REVIEWER) $(if $(RUN),--run $(RUN),) $(if $(SAMPLE),--sample $(SAMPLE),)

judge:
	$(BENCHMARK) judge $(if $(RUN),--run $(RUN),)

report:
	$(BENCHMARK) report $(if $(RUN),--run $(RUN),) $(if $(JSON),--json $(JSON),) $(if $(CSV),--csv $(CSV),)

clean-results:
	$(BENCHMARK) clean-results --db $(DB)

test:
	$(VENV)/bin/pytest

fmt:
	$(VENV)/bin/ruff format src tests
	$(VENV)/bin/ruff check --fix src tests

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/mypy src
