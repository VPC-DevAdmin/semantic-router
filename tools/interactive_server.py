#!/usr/bin/env python3
"""Backend for the interactive routing demo (interactive/).

Serves the static UI and two JSON endpoints:

  POST /api/route  {query, tiers:[{id,name,exemplars,threshold}]}
       -> per-tier similarity scores, the closest matching exemplar per tier,
          the chosen tier, and a one-line reasoning. Works with NO API keys —
          this is the routing decision, the part worth showing for free.

  POST /api/chat   {query, mode:"auto"|<tier_id>, tiers:[... incl model/
                    provider/base_url/api_key ...], max_tier_id}
       -> routes (or honors a forced tier), calls that tier's model, and returns
          {routing, answer | no_api_key, tier}. "Get a deeper answer" re-calls
          with mode=max_tier_id.

Scoring is pluggable: real embeddings via `fastembed` if installed, else a
zero-dep lexical token-cosine fallback (clearly labeled in the response). Either
way you get per-tier scores + the closest exemplar so the routing UX is real.

Stdlib + httpx (already a dep). Run:  python tools/interactive_server.py --port 8900
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "interactive"


# ── Scoring ──────────────────────────────────────────────────────────────────

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> Counter:
    return Counter(_WORD.findall((s or "").lower()))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b[t] for t, v in a.items() if t in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# Optional real-embedding backend. Loaded lazily; falls back to lexical cosine.
_EMBED = {"model": None, "tried": False}


def _embedder():
    if _EMBED["tried"]:
        return _EMBED["model"]
    _EMBED["tried"] = True
    try:
        from fastembed import TextEmbedding  # type: ignore
        _EMBED["model"] = TextEmbedding()
    except Exception:
        _EMBED["model"] = None
    return _EMBED["model"]


def _embed_cosine(query: str, exemplars: list[str]) -> list[float] | None:
    model = _embedder()
    if model is None:
        return None
    import numpy as np  # fastembed pulls numpy
    vecs = list(model.embed([query] + list(exemplars)))
    q = vecs[0]
    out = []
    for e in vecs[1:]:
        denom = (float(np.linalg.norm(q)) * float(np.linalg.norm(e))) or 1.0
        out.append(float(np.dot(q, e)) / denom)
    return out


def score_tiers(query: str, tiers: list[dict]) -> dict:
    """Score the query against every tier's exemplars. Returns ranked tier scores
    (each with its closest exemplar), the chosen tier id, and a reasoning string."""
    use_embed = _embedder() is not None
    qtok = _tokens(query)
    results = []
    for t in tiers:
        exemplars = t.get("exemplars") or []
        sims: list[float]
        if use_embed and exemplars:
            sims = _embed_cosine(query, exemplars) or []
        else:
            sims = [_cosine(qtok, _tokens(e)) for e in exemplars]
        if sims:
            best_i = max(range(len(sims)), key=lambda i: sims[i])
            score, closest = sims[best_i], exemplars[best_i]
        else:
            score, closest = 0.0, None
        results.append({
            "id": t["id"], "name": t.get("name", t["id"]),
            "score": round(score, 4), "closest_exemplar": closest,
            "threshold": t.get("threshold", 0.0),
        })

    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    chosen = ranked[0] if ranked else None
    # Reasoning: the winner, its margin over the runner-up, and the matched exemplar.
    reasoning = ""
    if chosen and chosen["score"] > 0:
        margin = chosen["score"] - (ranked[1]["score"] if len(ranked) > 1 else 0.0)
        reasoning = (f"Routed to {chosen['name']} — highest similarity "
                     f"({chosen['score']:.2f}), {margin:.2f} above the next tier. "
                     f"Closest exemplar: \"{(chosen['closest_exemplar'] or '')[:90]}\".")
    elif chosen:
        reasoning = (f"No exemplar matched strongly; defaulting to {chosen['name']}. "
                     f"Add exemplars to sharpen routing.")
    return {
        "scorer": "embeddings (fastembed)" if use_embed else "lexical token-cosine",
        "tiers": results,
        "ranked_ids": [r["id"] for r in ranked],
        "chosen_id": chosen["id"] if chosen else None,
        "reasoning": reasoning,
    }


# ── Model call ───────────────────────────────────────────────────────────────

def call_model(tier: dict, query: str, max_tokens: int = 700) -> dict:
    """Call one tier's model. Returns {answer} or {no_api_key|error}."""
    key = (tier.get("api_key") or "").strip()
    if not key:
        return {"no_api_key": True}
    provider = (tier.get("provider") or "").lower()
    base = (tier.get("base_url") or "").rstrip("/")
    model = tier.get("model") or ""
    try:
        if provider == "anthropic":
            url = (base or "https://api.anthropic.com/v1") + "/messages"
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
            body = {"model": model, "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": query}]}
            with httpx.Client(timeout=120.0) as c:
                r = c.post(url, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            text = "".join(p.get("text", "") for p in data.get("content", []))
            return {"answer": text, "model": model}
        # OpenAI-compatible (OpenAI, Google OpenAI-compat, vLLM, …)
        url = (base or "https://api.openai.com/v1") + "/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
        body = {"model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": query}]}
        with httpx.Client(timeout=120.0) as c:
            r = c.post(url, headers=headers, json=body)
        r.raise_for_status()
        return {"answer": r.json()["choices"][0]["message"]["content"], "model": model}
    except httpx.HTTPStatusError as exc:
        return {"error": f"{exc.response.status_code} from {provider or 'provider'}: "
                         f"{exc.response.text[:200]}"}
    except Exception as exc:
        return {"error": str(exc)}


def _tier_by_id(tiers: list[dict], tid: str | None) -> dict | None:
    return next((t for t in tiers if t.get("id") == tid), None)


def handle_chat(payload: dict) -> dict:
    query = payload.get("query", "")
    tiers = payload.get("tiers", [])
    mode = payload.get("mode", "auto")
    routing = score_tiers(query, tiers)

    if mode == "auto":
        chosen_id = routing["chosen_id"]
    else:
        chosen_id = mode   # forced tier id (incl. max_tier_id for "deeper answer")
        routing["forced"] = True
    tier = _tier_by_id(tiers, chosen_id) or (tiers[0] if tiers else None)
    if tier is None:
        return {"routing": routing, "error": "no tiers configured"}

    result = call_model(tier, query)
    return {"routing": routing, "tier": {"id": tier["id"], "name": tier.get("name"),
            "model": tier.get("model")}, **result}


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

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            f = (STATIC_DIR / rel).resolve()
            if not str(f).startswith(str(STATIC_DIR.resolve())) or not f.is_file():
                self._json(404, {"error": "not found"})
                return
            ctype = {"html": "text/html", "css": "text/css", "js": "text/javascript",
                     "json": "application/json"}.get(f.suffix.lstrip("."), "text/plain")
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            try:
                payload = self._read_json()
            except json.JSONDecodeError:
                self._json(400, {"error": "invalid JSON"})
                return
            if path == "/api/route":
                self._json(200, score_tiers(payload.get("query", ""),
                                            payload.get("tiers", [])))
            elif path == "/api/chat":
                self._json(200, handle_chat(payload))
            else:
                self._json(404, {"error": "not found"})

    return Handler


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8900)
    args = p.parse_args()
    httpd = ThreadingHTTPServer(("", args.port), _make_handler())
    scorer = "embeddings (fastembed)" if _embedder() else "lexical token-cosine"
    print(f"interactive routing demo on http://localhost:{args.port}/  [scorer: {scorer}]")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
