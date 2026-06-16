#!/usr/bin/env python3
"""Backend for the LIVE interactive routing demo (interactive/).

This is a thin client of a running **vllm-sr** — there is no local scorer and no
mock backend. The user types (or picks) a query; the server forwards it to
vllm-sr, which classifies it and forwards to the real model configured for the
chosen tier, and the UI shows the routing decision + the real answer.

Config lives in a SEPARATE overlay file so the canonical benchmark config is
never touched:
    config/live_demo.json         committed default (demo tiers, blank keys)
    config/live_demo.local.json   the user's saved version (gitignored; keys)

Endpoints:
  GET  /api/config            -> overlay (API keys masked; key_set flags)
  POST /api/config            -> persist overlay to live_demo.local.json
  POST /api/apply             -> build a router-config from the overlay (tier
                                 models/keys via TIER{N}_1_* env, edited cutoffs
                                 + signal banks) and (re)launch vllm-sr
  GET  /api/queries           -> the benchmark queries grouped by category
  POST /api/chat              -> proxy {query, mode} to vllm-sr; return
                                 {routing, answer | error}
  GET  /                      -> the static UI

Stdlib + httpx + pyyaml (deps). Run:  python tools/interactive_server.py --port 8900
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "interactive"
DEFAULT_OVERLAY = ROOT / "config" / "live_demo.json"
USER_OVERLAY = ROOT / "config" / "live_demo.local.json"
CANON_EXEMPLARS = ROOT / "config" / "router-exemplars.yaml"
CANON_BACKENDS = ROOT / "config" / "router-backends.yaml"
LIVE_EXEMPLARS = ROOT / "config" / "live_exemplars.local.yaml"
LIVE_ROUTER_CFG = ROOT / "config" / "router-config.yaml"


# ── Overlay config ───────────────────────────────────────────────────────────

def load_overlay() -> dict:
    path = USER_OVERLAY if USER_OVERLAY.exists() else DEFAULT_OVERLAY
    return json.loads(path.read_text())


def masked_overlay(ov: dict) -> dict:
    """Overlay for the browser: never send raw keys back; flag which tiers have one."""
    out = json.loads(json.dumps(ov))
    for t in out.get("tiers", []):
        t["key_set"] = bool((t.get("api_key") or "").strip())
        t["api_key"] = ""
    return out


def merge_overlay(incoming: dict) -> dict:
    """Persist the incoming overlay, preserving any existing key the UI left blank
    (the browser never receives raw keys, so blank means 'unchanged')."""
    prev = load_overlay()
    prev_keys = {t["id"]: t.get("api_key", "") for t in prev.get("tiers", [])}
    for t in incoming.get("tiers", []):
        if not (t.get("api_key") or "").strip():
            t["api_key"] = prev_keys.get(t["id"], "")
        t.pop("key_set", None)
    USER_OVERLAY.write_text(json.dumps(incoming, indent=2, ensure_ascii=False))
    return incoming


# ── Benchmark queries (the example picker) ────────────────────────────────────

def grouped_queries() -> dict:
    """Benchmark queries grouped by specialization/category for the picker."""
    qfile = ROOT / "data" / "queries.json"
    if not qfile.exists():
        return {"categories": {}}
    items = json.loads(qfile.read_text()).get("queries", [])
    by_cat: dict[str, list[str]] = defaultdict(list)
    for it in items:
        prompt = it.get("prompt") or it.get("query")
        if not prompt:
            continue
        cats = it.get("specializations") or ["general"]
        for c in cats:
            if len(by_cat[c]) < 25:        # cap per category for a tidy picker
                by_cat[c].append(prompt)
    return {"categories": dict(sorted(by_cat.items()))}


# ── vllm-sr proxy ──────────────────────────────────────────────────────────────

def _upstream_error_text(r: httpx.Response) -> str:
    """Pull the human-readable error out of an upstream error response — the
    OpenAI/Anthropic-style {"error": {"message": ...}} shape, falling back to
    raw text. Truncated; safe on non-JSON bodies."""
    try:
        j = r.json()
        e = j.get("error")
        if isinstance(e, dict) and e.get("message"):
            return str(e["message"])[:400]
        if e:
            return str(e)[:400]
        # Some adapters return a completion-shaped body (not an error object) on a
        # 4xx — dumping that raw JSON is noise, so summarize instead.
        if isinstance(j, dict) and "choices" in j:
            return "the upstream rejected the request (empty/blocked completion, no error detail)"
    except ValueError:
        pass
    return (r.text or "(no body)").strip()[:300]


def _provider_models(provider: str, base_url: str, api_key: str) -> list[str]:
    """Fetch available model ids from a provider's list endpoint.

    OpenAI and OpenAI-compatible providers (including Google's
    `/v1beta/openai` shim) answer `GET {base_url}/models` with a Bearer
    header; Anthropic answers `GET {base_url}/models` with `x-api-key` +
    `anthropic-version`. Returns sorted model ids; raises on failure."""
    base = (base_url or "").rstrip("/")
    if not base:
        raise ValueError("no base URL configured for this tier")
    if (provider or "").lower() == "anthropic" or "anthropic.com" in base:
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    else:
        headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=20.0) as c:
        r = c.get(f"{base}/models", headers=headers)
    r.raise_for_status()
    payload = r.json()
    rows = payload.get("data") or payload.get("models") or []
    ids = [(m.get("id") or m.get("name")) for m in rows if isinstance(m, dict)]
    return sorted({i for i in ids if i})


def list_models(payload: dict) -> dict:
    """Resolve a tier's credentials (form values, falling back to the saved
    overlay's key when the form key is blank/masked) and return its provider's
    model list, or {error}."""
    tier_id = payload.get("tier_id")
    provider = payload.get("provider") or ""
    base_url = payload.get("base_url") or ""
    api_key = (payload.get("api_key") or "").strip()
    if not api_key and tier_id:
        saved = next((t for t in load_overlay().get("tiers", [])
                      if t.get("id") == tier_id), None)
        if saved:
            api_key = (saved.get("api_key") or "").strip()
            base_url = base_url or saved.get("base_url") or ""
            provider = provider or saved.get("provider") or ""
    if not api_key:
        return {"error": "No API key for this tier — paste one (then it works "
                         "immediately) or Save first."}
    try:
        return {"models": _provider_models(provider, base_url, api_key)}
    except httpx.HTTPStatusError as exc:
        return {"error": f"HTTP {exc.response.status_code} from "
                         f"{provider or 'provider'}: {_upstream_error_text(exc.response)}"}
    except (httpx.HTTPError, ValueError) as exc:
        return {"error": str(exc)}


def _to_float(v):
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


# x-vsr-matched-* headers name the signals that fired for this query.
_MATCH_HEADERS = {
    "x-vsr-matched-embeddings": "embedding",
    "x-vsr-matched-complexity": "complexity",
    "x-vsr-matched-structure": "structure",
    "x-vsr-matched-projections": "projection",
}


def _matched_signals(h: dict) -> list:
    out = []
    for hdr, kind in _MATCH_HEADERS.items():
        for name in (h.get(hdr) or "").split(","):
            name = name.strip()
            if name:
                out.append({"type": kind, "name": name})
    return out


ROUTER_LOG_CONTAINER = "vllm-sr-router-container"


def _routing_scores() -> dict:
    """Best-effort: the per-signal confidences + request difficulty the router
    actually computed, pulled from the latest router_replay_start in the router
    container log. Returns {} on any failure (the UI falls back to the matched
    signal names from the response headers)."""
    try:
        # Tail generously: the router emits a lot between requests (perf,
        # traces), so the replay line can be well past the last few dozen lines.
        # The router writes its JSON logs to the container's STDERR, so merge
        # both streams (docker logs forwards container stderr to its stderr).
        res = subprocess.run(
            ["docker", "logs", "--tail", "400", ROUTER_LOG_CONTAINER],
            capture_output=True, text=True, timeout=6)
        out = (res.stdout or "") + (res.stderr or "")
        for line in reversed(out.splitlines()):
            if '"router_replay_start"' not in line or "{" not in line:
                continue
            j = json.loads(line[line.index("{"):])
            return {
                "request_difficulty": _to_float(
                    (j.get("projection_scores") or {}).get("request_difficulty")),
                "signal_confidences": {
                    k: _to_float(v)
                    for k, v in (j.get("signal_confidences") or {}).items()
                },
            }
    except (subprocess.SubprocessError, OSError, ValueError, KeyError):
        pass
    return {}


def _tier_for(overlay: dict, selected: str | None):
    """Map x-vsr-selected-model back to a tier. The live config names model cards
    by the REAL model id (vllm-sr forwards the name upstream), so the header is
    the model id, not the tier id — match on either."""
    if not selected:
        return None
    return next((t for t in overlay.get("tiers", [])
                 if t.get("id") == selected or t.get("model") == selected), None)


def _routing_from_headers(h: dict, overlay: dict, mode: str) -> dict:
    """The routing decision the router published on the response headers. Present
    on BOTH success and error responses (the router decides before forwarding),
    so the UI can show the rationale even when the upstream rejects the call."""
    selected = h.get("x-vsr-selected-model")
    tier = _tier_for(overlay, selected)
    return {
        "selected_tier_id": tier["id"] if tier else selected,
        "selected_tier_name": tier["name"] if tier else selected,
        "served_model": tier["model"] if tier else selected,
        "category": h.get("x-vsr-selected-category"),
        "reasoning": h.get("x-vsr-selected-reasoning"),
        "confidence": _to_float(h.get("x-vsr-selected-confidence")),
        "decision": h.get("x-vsr-selected-decision"),
        "matched": _matched_signals(h),
        "forced": mode != "auto",
        "cache_hit": "x-vsr-selected-model" not in h,
        **_routing_scores(),
    }


def vllm_chat(overlay: dict, query: str, mode: str) -> dict:
    """Forward one query to the live vllm-sr. mode='auto' routes; otherwise mode
    is a tier id to pin. Returns {routing, answer} or {routing, error}."""
    base = (overlay.get("vllm_sr_url") or "http://localhost:8899").rstrip("/")
    # mode 'auto' routes; otherwise it's a tier id to pin. The live config names
    # model cards by the real model id, so pin by that name (fall back to the id).
    if mode == "auto":
        model = "auto"
    else:
        pinned = next((t for t in overlay.get("tiers", []) if t.get("id") == mode), None)
        model = (pinned.get("model") if pinned else None) or mode
    # Use `max_tokens` (not max_completion_tokens): Anthropic *requires* it and
    # vllm-sr's OAI→Anthropic adapter keys off it; OpenAI/Google accept it too.
    # Omit temperature — some models (e.g. gpt-5 reasoning) reject temperature=0.
    body = {"model": model, "messages": [{"role": "user", "content": query}],
            "max_tokens": 1024}
    try:
        with httpx.Client(timeout=180.0) as c:
            r = c.post(f"{base}/v1/chat/completions", json=body)
    except httpx.HTTPError as exc:
        # Couldn't reach the router at all (connection refused, DNS, timeout).
        return {"error": f"vllm-sr not reachable at {base} ({exc}). Start it with "
                         f"`make route` (and configure your models in Settings → Apply)."}
    h = {k.lower(): v for k, v in r.headers.items()}
    routing = _routing_from_headers(h, overlay, mode)
    if r.status_code >= 400:
        # Routed fine, but the upstream rejected the call. Keep the rationale so
        # the UI still shows which tier + why, alongside the upstream error.
        who = routing.get("selected_tier_name") or h.get("x-vsr-selected-model") or "the tier"
        served = routing.get("served_model") or "?"
        return {"routing": routing,
                "error": f"{who} ({served}) returned HTTP {r.status_code}: "
                         f"{_upstream_error_text(r)} — check this tier's model id and "
                         f"API key in Settings."}
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return {"routing": routing, "answer": content}


# ── Apply overlay → vllm-sr config + reload ───────────────────────────────────

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _docker_ready() -> bool:
    """True if the Docker daemon is reachable (vllm-sr serve needs it)."""
    try:
        return subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    except (FileNotFoundError, OSError):
        return False


# The router's apiserver (separate from the Envoy frontend) answers /ready once
# the stack is fully up — including the first-launch embedding-model download.
ROUTER_READY_URL = "http://127.0.0.1:8080/ready"


def _router_ready() -> bool:
    """True if the router apiserver reports ready."""
    try:
        return httpx.get(ROUTER_READY_URL, timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


# ── Apply progress (background thread + polled status) ───────────────────────
# Apply is slow (stop → rebuild → launch → wait-for-ready). We run it off the
# request thread and publish step-by-step status the UI polls, so the user sees
# a progress bar that completes only when the router actually answers /ready.
APPLY_STEPS = [
    "Rebuilding router config",
    "Checking Docker",
    "Preparing routing model",
    "Stopping current router",
    "Launching vLLM Semantic Router",
    "Waiting for the router to come online",
]
_apply_lock = threading.Lock()
_apply_state = {
    "running": False, "done": False, "ok": None, "step": 0,
    "steps": APPLY_STEPS, "phase": "idle", "detail": "", "failed_step": None,
}


def _apply_set(**kw) -> None:
    with _apply_lock:
        _apply_state.update(kw)


def _apply_snapshot() -> dict:
    with _apply_lock:
        return dict(_apply_state)


def _wait_router_ready(deadline_s: float, progress: bool = False) -> bool:
    """Poll /ready up to deadline_s seconds. With progress=True, publish the
    elapsed wait to the apply status so the UI can show it live."""
    start = time.monotonic()
    while time.monotonic() - start < deadline_s:
        if _router_ready():
            return True
        if progress:
            _apply_set(detail=f"waiting for the router apiserver… {int(time.monotonic() - start)}s")
        time.sleep(1.5)
    return False


# The router downloads this embedding model on first launch; HF's Xet CDN is
# 403-blocked on many networks and vllm-sr won't forward HF_HUB_DISABLE_XET into
# its container, so we pre-seed the model into the bind-mounted config/models
# (Xet off) and the router skips the download. Mirrors `make fetch-router-model`.
ROUTER_EMBED_REPO = "llm-semantic-router/mmbert-embed-32k-2d-matryoshka"
# Pinned to the known-good release. v0.2.0 leaves the router's request path dead
# (doesn't start the postgres/redis backends it needs). Keep in sync with
# VLLM_SR_VERSION in the Makefile and the --image in config/router.yaml.
VLLM_SR_IMAGE = "ghcr.io/vllm-project/semantic-router/vllm-sr:v0.3.0"
MODELS_DIR = ROOT / "config" / "models"


def _ensure_router_model() -> None:
    """Pre-seed the router's embedding model into config/models if absent, so
    `vllm-sr serve` doesn't fail downloading it behind a firewalled Xet CDN.
    Idempotent and best-effort: on any failure the router still tries its own
    download at launch."""
    name = ROUTER_EMBED_REPO.split("/")[-1]
    if (MODELS_DIR / name / "model.safetensors").exists():
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["docker", "run", "--rm",
         "-e", "HF_HUB_DISABLE_XET=1", "-e", "HF_HUB_ENABLE_HF_TRANSFER=0",
         "-v", f"{MODELS_DIR}:/app/models",
         "--entrypoint", "python3", VLLM_SR_IMAGE,
         "-c", ("from huggingface_hub import snapshot_download; "
                f"snapshot_download('{ROUTER_EMBED_REPO}', "
                f"local_dir='/app/models/{name}')")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900)


def _clean_log(text: str) -> str:
    """Make a subprocess log fit for the UI: strip ANSI color codes (vllm-sr
    prints a colored banner) and surface the actionable error line if there is
    one, else the tail."""
    plain = _ANSI.sub("", text or "")
    lines = [ln.strip() for ln in plain.splitlines() if ln.strip()]
    flagged = [ln for ln in lines if "ERROR" in ln or "Failed" in ln or "error" in ln.lower()]
    out = " · ".join(flagged) if flagged else " ".join(lines)
    return out[-600:] if out else "(no output)"


def build_live_exemplars(overlay: dict) -> None:
    """Write a live exemplars YAML = canonical + the overlay's edited cutoffs and
    per-signal banks (weight/threshold/candidates). Canonical file untouched."""
    ex = yaml.safe_load(CANON_EXEMPLARS.read_text())
    if overlay.get("tier_cutoffs"):
        ex["tier_cutoffs"] = overlay["tier_cutoffs"]
    edits = {s["id"]: s for s in overlay.get("signals", [])}
    for sig in ex.get("embedding_signals", []):
        e = edits.get(sig["id"])
        if not e:
            continue
        if "weight" in e:
            sig["weight"] = e["weight"]
        if "threshold" in e:
            sig["threshold"] = e["threshold"]
        if e.get("candidates"):
            sig["candidates"] = e["candidates"]
    LIVE_EXEMPLARS.write_text(yaml.safe_dump(ex, sort_keys=False, allow_unicode=True))


def apply_overlay(overlay: dict) -> dict:
    """Build a router-config from the overlay and (re)launch vllm-sr.

    Tier model/base_url/key go in via TIER{N}_1_* env overrides (the same path
    `make route` uses); edited cutoffs + signal banks go via a live exemplars
    file. Requires vllm-sr installed + the harness venv.
    """
    env = dict(os.environ)
    # Force the classic HF download path: the embedding model the router pulls
    # on first launch is Xet-backed, and the Xet CDN is 403-blocked on many
    # networks. vllm-sr forwards HF_* into the router container. (See router.yaml
    # for the same defaults on the `make route` path.)
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    for i, t in enumerate(overlay.get("tiers", []), start=1):
        if t.get("model"):
            env[f"TIER{i}_1_MODEL"] = t["model"]
        if t.get("base_url"):
            env[f"TIER{i}_1_URL"] = t["base_url"]
        if (t.get("api_key") or "").strip():
            env[f"TIER{i}_1_API_KEY"] = t["api_key"]
        if t.get("provider"):
            env[f"TIER{i}_1_PROVIDER"] = t["provider"]
    build_live_exemplars(overlay)
    try:
        _apply_set(step=0, phase="building", detail="compiling exemplars + backends")
        subprocess.run(
            [sys.executable, "-m", "benchmark.build_router_config",
             "--exemplars", str(LIVE_EXEMPLARS), "--backends", str(CANON_BACKENDS),
             "--out", str(LIVE_ROUTER_CFG),
             # vllm-sr forwards the model card NAME upstream, so against real
             # providers the card must be named with the real model id (else
             # OpenAI 404s on "model tier2 does not exist").
             "--served-model-names", "real",
             # No promotion lanes: route purely by the difficulty band the UI
             # meter shows (lanes override it on soft embedding matches and
             # over-route ambiguous prompts to the frontier model).
             "--lanes", "off"],
            cwd=str(ROOT), env=env, check=True, capture_output=True, text=True)
        _apply_set(step=1, phase="docker", detail="checking the Docker daemon")
        if not _docker_ready():
            return {"ok": False, "step": "docker",
                    "detail": "Docker daemon not reachable. vllm-sr runs as a Docker "
                              "stack — start Docker (sudo systemctl start docker) and "
                              "ensure your user can access /var/run/docker.sock "
                              "(sudo usermod -aG docker $USER, then re-login)."}
        # Pre-seed the embedding model so the first launch doesn't fail trying to
        # download it behind a firewalled HF Xet CDN. Best-effort.
        _apply_set(step=2, phase="model", detail="ensuring the routing model is present")
        try:
            _ensure_router_model()
        except (subprocess.SubprocessError, OSError):
            pass
        # Apply must REPLACE the running stack so the new config takes effect.
        # `vllm-sr serve` on an already-running stack no-ops — it leaves the old
        # config live (e.g. the mock-backed config from `make route`), so the UI
        # would keep showing mock answers despite real models being configured.
        # Stop first to guarantee the rebuilt config is loaded on relaunch.
        _apply_set(step=3, phase="stopping", detail="tearing down the running stack")
        subprocess.run(["vllm-sr", "stop"], cwd=str(ROOT), env=env,
                       capture_output=True, text=True)
        _apply_set(step=4, phase="launching",
                   detail="starting containers (router, envoy, datastores)")
        serve = subprocess.run(["vllm-sr", "serve", "--config", str(LIVE_ROUTER_CFG),
                                "--minimal", "--image", VLLM_SR_IMAGE],
                               cwd=str(ROOT), env=env, capture_output=True, text=True)
        if serve.returncode != 0:
            return {"ok": False, "step": "serve",
                    "detail": _clean_log(serve.stderr or serve.stdout)}
        # `vllm-sr serve` exits 0 once the stack is *launched* — not once the
        # router is *ready*. Poll /ready (publishing progress) until it answers,
        # so the UI's bar completes only when the router can actually route.
        _apply_set(step=5, phase="waiting", detail="waiting for the router apiserver…")
        if _wait_router_ready(240.0, progress=True):
            return {"ok": True, "detail": "Router is live and serving."}
        return {"ok": True, "warming": True,
                "detail": "vllm-sr launched. On first launch it downloads its routing "
                          "model (~2-3 min) before it can route — your first query may "
                          "say 'not reachable' until that finishes. Retry shortly; watch "
                          "progress with `vllm-sr logs router`."}
    except FileNotFoundError as exc:
        return {"ok": False, "step": "launch",
                "detail": f"{exc}. Install vllm-sr (`make setup`) and run from the harness venv."}
    except subprocess.CalledProcessError as exc:
        return {"ok": False, "step": "build",
                "detail": _clean_log(exc.stderr or exc.stdout or str(exc))}


def _run_apply(overlay: dict) -> None:
    """Background runner: drive apply_overlay, publishing terminal status the UI
    polls. apply_overlay updates the per-step state as it goes."""
    _apply_set(running=True, done=False, ok=None, step=0, phase="start",
               detail="saving + rebuilding…", failed_step=None)
    try:
        result = apply_overlay(overlay)
        ok = bool(result.get("ok"))
        _apply_set(running=False, done=True, ok=ok, step=len(APPLY_STEPS),
                   phase="done" if ok else "error", detail=result.get("detail", ""),
                   failed_step=None if ok else result.get("step"))
    except Exception as exc:  # noqa: BLE001 — surface any unexpected failure to the UI
        _apply_set(running=False, done=True, ok=False, phase="error",
                   detail=f"{type(exc).__name__}: {exc}", failed_step="unexpected")


# ── HTTP server ──────────────────────────────────────────────────────────────

def _make_handler():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code: int, payload: dict):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/config":
                self._json(200, masked_overlay(load_overlay()))
                return
            if path == "/api/queries":
                self._json(200, grouped_queries())
                return
            if path == "/api/apply/status":
                self._json(200, _apply_snapshot())
                return
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            f = (STATIC_DIR / rel).resolve()
            if not str(f).startswith(str(STATIC_DIR.resolve())) or not f.is_file():
                self._json(404, {"error": "not found"})
                return
            ctype = {"html": "text/html", "css": "text/css", "js": "text/javascript"}.get(
                f.suffix.lstrip("."), "text/plain")
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            try:
                payload = self._body()
            except json.JSONDecodeError:
                self._json(400, {"error": "invalid JSON"})
                return
            if path == "/api/config":
                if payload.get("_reset"):
                    USER_OVERLAY.unlink(missing_ok=True)   # revert to committed default
                else:
                    merge_overlay(payload)
                self._json(200, {"ok": True})
            elif path == "/api/apply":
                # Non-blocking: launch the apply on a worker thread and let the
                # client poll /api/apply/status for the progress bar.
                if _apply_snapshot().get("running"):
                    self._json(200, {"started": False, "running": True})
                else:
                    overlay = load_overlay()
                    threading.Thread(target=_run_apply, args=(overlay,), daemon=True).start()
                    self._json(200, {"started": True})
            elif path == "/api/models":
                self._json(200, list_models(payload))
            elif path == "/api/chat":
                self._json(200, vllm_chat(load_overlay(), payload.get("query", ""),
                                          payload.get("mode", "auto")))
            else:
                self._json(404, {"error": "not found"})

    return Handler


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8900)
    args = p.parse_args()
    ov = load_overlay()
    httpd = ThreadingHTTPServer(("", args.port), _make_handler())
    print(f"live interactive demo on http://localhost:{args.port}/  "
          f"[routes via vllm-sr at {ov.get('vllm_sr_url')}]")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
