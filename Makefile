# Semantic Router Benchmark Harness
#
# All targets are stubs until M1+ lands. See PLAN.md.

.PHONY: setup seed gold run pass1 pass2 review judge report resume clean-results test fmt lint help

help:
	@echo "Targets:"
	@echo "  setup          venv + deps + init DB"
	@echo "  seed           upsert data/queries.yaml into DB"
	@echo "  gold           generate/refresh gold answers"
	@echo "  run            new run_id; starts router; pass1 + pass2; tears down"
	@echo "  pass1          routing accuracy only (resumable)"
	@echo "  pass2          response generation only (resumable)"
	@echo "  review         human scoring TUI for pending pass-2 rows"
	@echo "  judge          LLM-as-judge scoring"
	@echo "  report         aggregate stats + CSV/JSON export"
	@echo "  resume RUN=<id>  resume a specific run"
	@echo "  clean-results  wipe runs/results/scores; preserves queries and gold"
	@echo "  test / fmt / lint"

setup:
	@echo "TODO(M1): create .venv with uv, install deps, init DB"

seed:
	@echo "TODO(M1): load data/queries.yaml into DB"

gold:
	@echo "TODO(M2): generate gold answers via configured gold tier"

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
	@echo "TODO(M1): wipe runs/results/scores; preserve queries and gold"

test:
	@echo "TODO: pytest"

fmt:
	@echo "TODO: ruff format + black"

lint:
	@echo "TODO: ruff + mypy"
