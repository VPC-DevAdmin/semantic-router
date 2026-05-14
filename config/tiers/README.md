# Tier configuration — single source of truth

Each `tierN.yaml` file in this directory defines one tier end-to-end:
- **metadata** (`name`, `level`, `specializations`, `timeout_s`) — used by
  the routing-accuracy and tier-answer passes.
- **identity** (`router_alias`, `served_model_name`) — what the router emits
  vs. what the upstream serves. Same string for local tiers; differs for
  vendor APIs (e.g. `tier4` vs. `claude-sonnet-4-7`).
- **endpoint** (`url`, `api_key_env`) — direct OAI endpoint for `make answers`.
- **backend** — how `make start_LLM` provisions this tier (or doesn't).

These YAMLs drive:
- `make answers` — direct OAI calls (uses `endpoint`, `served_model_name`).
- `make route` — tier lookup from router headers (uses `router_alias`).
- `make start_LLM` / `make stop_LLM` — local docker launch (uses `backend`).

The router itself reaches each tier via `config/router-backends.yaml`,
not via these files. To swap a tier's direct-call endpoint, edit
**this directory**; to swap a tier's router-side endpoint, edit
`config/router-backends.yaml` and restart the router stack.

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
