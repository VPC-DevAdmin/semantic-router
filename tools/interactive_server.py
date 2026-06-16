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

def vllm_chat(overlay: dict, query: str, mode: str) -> dict:
    """Forward one query to the live vllm-sr. mode='auto' routes; otherwise mode
    is a tier id to pin. Returns {routing, answer} or {routing?, error}."""
    base = (overlay.get("vllm_sr_url") or "http://localhost:8899").rstrip("/")
    model = "auto" if mode == "auto" else mode    # tier id pins a tier
    body = {"model": model, "messages": [{"role": "user", "content": query}],
            "temperature": 0.0, "max_completion_tokens": 800}
    try:
        with httpx.Client(timeout=180.0) as c:
            r = c.post(f"{base}/v1/chat/completions", json=body)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        return {"error": f"vllm-sr not reachable at {base} ({exc}). Start it with "
                         f"`make route` (and configure your models in Settings → Apply)."}
    data = r.json()
    h = {k.lower(): v for k, v in r.headers.items()}
    selected = h.get("x-vsr-selected-model") or data.get("model")
    tier = next((t for t in overlay["tiers"] if t["id"] == selected), None)
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return {
        "routing": {
            "selected_tier_id": selected,
            "selected_tier_name": tier["name"] if tier else selected,
            "served_model": tier["model"] if tier else selected,
            "category": h.get("x-vsr-selected-category"),
            "reasoning": h.get("x-vsr-selected-reasoning"),
            "forced": mode != "auto",
            "cache_hit": "x-vsr-selected-model" not in h,
        },
        "answer": content,
    }


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


def _wait_router_ready(deadline_s: float) -> bool:
    """Poll /ready up to deadline_s seconds. Bounded so the Apply request never
    hangs on the multi-minute first-run model download — we report honestly
    instead of blocking."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if _router_ready():
            return True
        time.sleep(1.0)
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
        subprocess.run(
            [sys.executable, "-m", "benchmark.build_router_config",
             "--exemplars", str(LIVE_EXEMPLARS), "--backends", str(CANON_BACKENDS),
             "--out", str(LIVE_ROUTER_CFG)],
            cwd=str(ROOT), env=env, check=True, capture_output=True, text=True)
        if not _docker_ready():
            return {"ok": False, "step": "docker",
                    "detail": "Docker daemon not reachable. vllm-sr runs as a Docker "
                              "stack — start Docker (sudo systemctl start docker) and "
                              "ensure your user can access /var/run/docker.sock "
                              "(sudo usermod -aG docker $USER, then re-login)."}
        # Pre-seed the embedding model so the first launch doesn't fail trying to
        # download it behind a firewalled HF Xet CDN. Best-effort.
        try:
            _ensure_router_model()
        except (subprocess.SubprocessError, OSError):
            pass
        # Apply must REPLACE the running stack so the new config takes effect.
        # `vllm-sr serve` on an already-running stack no-ops — it leaves the old
        # config live (e.g. the mock-backed config from `make route`), so the UI
        # would keep showing mock answers despite real models being configured.
        # Stop first to guarantee the rebuilt config is loaded on relaunch.
        subprocess.run(["vllm-sr", "stop"], cwd=str(ROOT), env=env,
                       capture_output=True, text=True)
        serve = subprocess.run(["vllm-sr", "serve", "--config", str(LIVE_ROUTER_CFG),
                                "--minimal", "--image", VLLM_SR_IMAGE],
                               cwd=str(ROOT), env=env, capture_output=True, text=True)
        if serve.returncode != 0:
            return {"ok": False, "step": "serve",
                    "detail": _clean_log(serve.stderr or serve.stdout)}
        # `vllm-sr serve` exits 0 once the stack is *launched* — not once the
        # router is *ready*. On first launch the router still has to download
        # its embedding model (~2-3 min), and it can crash mid-startup. Poll
        # /ready briefly so we report the truth instead of a premature "OK".
        if _wait_router_ready(20.0):
            return {"ok": True, "detail": "Router config rebuilt and vllm-sr is live."}
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
                self._json(200, apply_overlay(load_overlay()))
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
