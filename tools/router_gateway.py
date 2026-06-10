#!/usr/bin/env python3
"""Contract gateway: adapt this repo's semantic router to the OpenAI-compatible
contract an agent orchestrator expects, WITHOUT specializing the router.

It is an additive front door — the harness, `make route`, and `make demo` are
untouched. The gateway is role-agnostic: roles are defined in
config/gateway_roles.yaml (pinned vs routed), so any client can use its own
role names. What it adds on top of plain OpenAI chat-completions:

  • role names in `model` → pinned tier or the routed `worker` path
  • `metadata.min_tier` honored as a hard floor (served = max(classified, min))
  • x-llm-model-served / x-llm-route-decision / x-llm-cost-usd response headers
    (route-decision exposes classified vs served vs min so an escalation is
     visible: "classifier said L2, floor forced L3")
  • strict structured output: a generic JSON-Schema → minimal-valid-instance
    generator so `response_format: json_schema strict` always returns a body
    that validates (standalone mode), the same guarantee guided_json gives.

Two modes:
  • standalone (default): a self-contained mock classifier + schema-valid canned
    output, so an orchestrator's full loop runs with ZERO real backends.
  • --router-url <vllm-sr>: classify the worker via the real semantic router.

Stdlib only (+ pyyaml, already a dep). Run:
    python tools/router_gateway.py --port 8800
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

log = logging.getLogger("router_gateway")

TIERS = ["L1", "L2", "L3", "L4", "L5"]


# ── Pure helpers (unit-tested) ───────────────────────────────────────────────

def clamp_tier(classified: str, min_tier: str | None) -> str:
    """served = max(classified, min_tier). Unknown min_tier is ignored."""
    if min_tier not in TIERS:
        return classified
    return classified if TIERS.index(classified) >= TIERS.index(min_tier) else min_tier


def route_decision(role: str, model: str, *, classified: str | None = None,
                   served: str | None = None, min_tier: str | None = None) -> str:
    """x-llm-route-decision string. For routed roles, expose classified/served and
    min only when a floor was applied; for pinned roles just role@model."""
    base = f"{role}@{model}"
    if classified is None:
        return base
    parts = [f"classified={classified}", f"served={served}"]
    if min_tier and served != classified:
        parts.insert(1, f"min={min_tier}")
    return base + "?" + "&".join(parts)


def cost_usd(prompt_tokens: int, completion_tokens: int, price: dict | None) -> float:
    """USD for one call from a {in_per_1m, out_per_1m} price (self_hosted → 0)."""
    if not price or price.get("self_hosted"):
        return 0.0
    return round(prompt_tokens * price.get("in_per_1m", 0.0) / 1e6
                 + completion_tokens * price.get("out_per_1m", 0.0) / 1e6, 6)


def classify_heuristic(prompt: str) -> str:
    """Standalone mock classifier: a transparent length/signal heuristic that
    produces a tier spread. Real routing uses --router-url (vllm-sr); this is
    only the zero-backend fallback."""
    words = len(prompt.split())
    hard = ("prove", "design", "architect", "synthesize", "novel", "derive",
            "multi-step", "trade-off", "tradeoff", "strategy")
    tier = ("L1" if words < 15 else "L2" if words < 40
            else "L3" if words < 100 else "L4")
    if any(h in prompt.lower() for h in hard):
        tier = TIERS[min(TIERS.index(tier) + 1, 4)]
    return tier


def resolve_role(name: str, cfg: dict) -> tuple[str, dict]:
    """Map a requested `model` (role) to (role_name, behavior). Unknown names
    fall back to default_role so the gateway never 404s on a role."""
    roles = cfg["roles"]
    if name in roles:
        return name, roles[name]
    default = cfg.get("default_role", "worker")
    return default, roles.get(default, {"mode": "route"})


def _resolve_ref(node: dict, defs: dict) -> dict:
    ref = node.get("$ref")
    if ref and ref.startswith("#/$defs/"):
        return defs.get(ref.split("/")[-1], {})
    return node


def instance_from_schema(schema: dict, defs: dict | None = None) -> object:
    """Generic minimal value that validates against a (Pydantic v2) JSON Schema.

    Lets standalone mode satisfy `response_format: json_schema strict` for ANY
    schema — no knowledge of the client's models. Handles object/array/string/
    number/integer/boolean, enum/const, anyOf/oneOf (first non-null), and $ref."""
    defs = defs if defs is not None else schema.get("$defs", {})
    schema = _resolve_ref(schema, defs)

    if "const" in schema:
        return schema["const"]
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema and schema[key]:
            non_null = [s for s in schema[key] if _resolve_ref(s, defs).get("type") != "null"]
            return instance_from_schema((non_null or schema[key])[0], defs)

    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), t[0])

    if t == "object" or "properties" in schema:
        props = schema.get("properties", {})
        # strict mode requires all properties present; emit every one.
        return {k: instance_from_schema(v, defs) for k, v in props.items()}
    if t == "array":
        n = schema.get("minItems", 0)
        item = schema.get("items", {})
        return [instance_from_schema(item, defs) for _ in range(n)]
    if t == "string":
        return "x" * schema.get("minLength", 0)
    if t == "integer":
        return int(schema.get("minimum", 0))
    if t == "number":
        return float(schema.get("minimum", 0.0))
    if t == "boolean":
        return False
    if t == "null":
        return None
    return {}


# ── Gateway core ─────────────────────────────────────────────────────────────

class Gateway:
    def __init__(self, roles_cfg: dict, pricing: dict, router_url: str | None = None):
        self.cfg = roles_cfg
        self.tiers = roles_cfg["tiers"]
        self.pricing = pricing
        self.router_url = router_url

    def _classify(self, prompt: str) -> str:
        if self.router_url:
            try:
                return self._classify_via_router(prompt)
            except Exception as exc:   # fall back to heuristic if the router is down
                log.warning("router classify failed (%s) — using heuristic", exc)
        return classify_heuristic(prompt)

    def _classify_via_router(self, prompt: str) -> str:
        """Classify the worker prompt through the real vllm-sr frontend, mapping
        the selected model back to a tier. Requires --router-url."""
        import httpx
        body = {"model": "auto", "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 1, "temperature": 0.0}
        with httpx.Client(timeout=60.0) as c:
            r = c.post(f"{self.router_url}/v1/chat/completions", json=body)
        r.raise_for_status()
        selected = (r.headers.get("x-vsr-selected-model")
                    or r.json().get("model") or "")
        # Map the router's selected model name onto our tier table when possible;
        # otherwise default mid-tier. (The tier id may already be "tier3" etc.)
        for tier, meta in self.tiers.items():
            if selected.lower() in (meta["model"].lower(), tier.lower()):
                return tier
        digits = [ch for ch in selected if ch.isdigit()]
        return f"L{digits[0]}" if digits and f"L{digits[0]}" in TIERS else "L3"

    def handle(self, body: dict) -> tuple[dict, dict]:
        """Process one chat-completions request → (response_body, response_headers)."""
        model_field = body.get("model", "worker")
        messages = body.get("messages", [])
        prompt = "\n".join(m.get("content", "") if isinstance(m.get("content"), str)
                           else "" for m in messages)
        meta = body.get("metadata") or {}
        min_tier = meta.get("min_tier")
        response_format = body.get("response_format")

        role, behavior = resolve_role(model_field, self.cfg)

        if behavior.get("mode") == "pinned":
            served = behavior.get("tier", "L3")
            classified = None
        else:
            classified = self._classify(prompt)
            served = clamp_tier(classified, min_tier)

        tier_meta = self.tiers.get(served, {})
        served_model = tier_meta.get("model", served)

        # Generate content: schema-valid instance for strict structured output,
        # else canned text. (Real-backend forwarding is the production path.)
        if response_format and response_format.get("type") == "json_schema":
            schema = response_format["json_schema"]["schema"]
            content = json.dumps(instance_from_schema(schema))
        else:
            content = f"[gateway role={role} tier={served} model={served_model}] {prompt[:120]}"

        ptok = max(1, len(prompt.split()))
        ctok = max(1, len(content.split()))
        price = self.pricing.get(tier_meta.get("price", ""), {})
        resp = {
            "id": f"chatcmpl-gw-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served_model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": ptok, "completion_tokens": ctok,
                      "total_tokens": ptok + ctok},
        }
        headers = {
            "x-llm-model-served": served_model,
            "x-llm-route-decision": route_decision(
                role, served_model, classified=classified, served=served,
                min_tier=min_tier),
            "x-llm-cost-usd": str(cost_usd(ptok, ctok, price)),
        }
        return resp, headers


def load_gateway(roles_path: Path, pricing_path: Path, router_url: str | None) -> Gateway:
    roles_cfg = yaml.safe_load(roles_path.read_text())
    pricing = json.loads(pricing_path.read_text()).get("models", {})
    return Gateway(roles_cfg, pricing, router_url)


# ── HTTP server ──────────────────────────────────────────────────────────────

def _make_handler(gw: Gateway):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, payload: dict, extra_headers: dict | None = None):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path.rstrip("/") == "/healthz":
                self._send(200, {"status": "ok"})
            elif self.path.startswith("/v1/models"):
                self._send(200, {"object": "list", "data": [
                    {"id": r, "object": "model"} for r in gw.cfg["roles"]]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                self._send(404, {"error": {"message": "not found"}})
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": {"message": "invalid JSON"}})
                return
            try:
                resp, hdrs = gw.handle(body)
            except Exception as exc:   # contract: headers set even on error, cost 0
                log.exception("gateway error")
                self._send(500, {"error": {"message": str(exc)}},
                           {"x-llm-model-served": "", "x-llm-route-decision": "error",
                            "x-llm-cost-usd": "0"})
                return
            self._send(200, resp, hdrs)

    return Handler


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8800)
    root = Path(__file__).resolve().parent.parent
    p.add_argument("--roles", type=Path, default=root / "config" / "gateway_roles.yaml")
    p.add_argument("--pricing", type=Path, default=root / "demo" / "pricing.json")
    p.add_argument("--router-url", default=None,
                   help="Real vllm-sr frontend (e.g. http://localhost:8801) to "
                        "classify the worker role. Omit for standalone mock mode.")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    gw = load_gateway(args.roles, args.pricing, args.router_url)
    mode = f"router={args.router_url}" if args.router_url else "standalone (mock classify)"
    httpd = ThreadingHTTPServer(("", args.port), _make_handler(gw))
    log.info("contract gateway on :%d  [%s]", args.port, mode)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
