# Tier configuration — single source of truth

Each `tierN.yaml` file in this directory defines one tier end-to-end:
- **metadata** (`name`, `level`, `specializations`, `timeout_s`) — used by
  the routing-accuracy and tier-answer passes.
- **identity** (`router_alias`, `served_model_name`) — what the router emits
  vs. what the upstream serves. Same string for local tiers; differs for
  vendor APIs (e.g. `tier4` vs. `claude-sonnet-4-7`).
- **endpoint** (`url`, `api_key_env`) — direct OAI endpoint for `make answers`.
- **router_backend_refs** — Envoy backend cluster endpoints; multiple refs
  load-balance at the router layer.
- **backend** — how `make start_LLM` provisions this tier (or doesn't).

These YAMLs drive:
- `make answers` — direct OAI calls (uses `endpoint`, `served_model_name`).
- `make route` — tier lookup from router headers (uses `router_alias`).
- `make start_LLM` / `make stop_LLM` — local docker launch (uses `backend`).
- `make gen-router-config` — emits `config/vllm-sr.yaml` with the right
  `providers.models` block (uses `router_alias`, `served_model_name`,
  `router_backend_refs`).

To swap a tier's backend (local docker → remote URL, or vice versa), edit
**only this directory**. Run `make gen-router-config` and restart the
router stack to take effect.

## `backend.kind` dispatch

| kind | meaning | start_LLM behavior |
|------|---------|--------------------|
| `docker_vllm_dual_socket` | local vLLM, two NUMA-pinned replicas | docker run × 2 |
| `remote` | already-running endpoint (vendor API, external server) | no-op |
| `placeholder` | no real backend yet — used for parking before a runner is wired | no-op |

Adding a new kind = a new dispatcher branch in `src/benchmark/start_llm.py`.

## Secrets

`api_key_env` names an environment variable. Set its value in `.env` at
the repo root (gitignored) — it's auto-loaded by the CLI. For local
backends with no auth, set `api_key_env: null`.
