# Semantic Router Benchmark Harness
#
# `make setup` and `make seed` are wired in M1.
# Other targets remain stubs until their respective milestones (see PLAN.md).

.PHONY: help setup seed gold run pass1 pass2 review judge report resume \
        clean-results test fmt lint validate-config

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
	@echo "  run              [M3+M4] new run_id; pass1 + pass2"
	@echo "  pass1            [M4] routing accuracy only (resumable)"
	@echo "  pass2            [M4] response generation only (resumable)"
	@echo "  review           [M5] human scoring TUI"
	@echo "  judge            [M5] LLM-as-judge scoring"
	@echo "  report           [M6] aggregate stats + export"
	@echo "  resume RUN=<id>  [M4] resume a specific run"
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

run:
	@echo "TODO(M3+M4): boot router subprocess, run pass1 + pass2, tear down"

pass1:
	@echo "TODO(M4): pass 1 only (resumable)"

pass2:
	@echo "TODO(M4): pass 2 only (resumable)"

review:
	@echo "TODO(M5): human scoring TUI"

judge:
	@echo "TODO(M5): LLM-as-judge scoring"

report:
	@echo "TODO(M6): aggregate stats and export"

resume:
	@echo "TODO(M4): resume run $(RUN)"

clean-results:
	@echo "TODO(M1+): wipe runs/results/scores; preserve queries and gold"

test:
	$(VENV)/bin/pytest

fmt:
	$(VENV)/bin/ruff format src tests
	$(VENV)/bin/ruff check --fix src tests

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/mypy src
