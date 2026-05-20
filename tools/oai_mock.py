"""Minimal OpenAI-compatible mock backend.

Purpose: stand in for real LLM tier endpoints so the harness pipeline
(`make route` → `make answers` → `make export`) can be validated end-to-end
at zero cost while real backends are being provisioned.

What it serves:
  - POST /v1/chat/completions  → tier-tagged canned text echoing the prompt
  - GET  /v1/models            → one entry per known tier (tier1..tier5)
  - GET  /healthz              → liveness probe

The response's `content` field embeds the model name from the request body
so downstream verification can confirm which tier each row was actually
served by (i.e. routing did the right thing in `make route`, and direct
dispatch did the right thing in `make answers`).

Stdlib only — no FastAPI dep, no uvicorn. ThreadingHTTPServer is plenty
for a 110-query benchmark.

Run:
    python tools/oai_mock.py --port 18811
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("oai_mock")

KNOWN_TIERS = ("tier1", "tier2", "tier3", "tier4", "tier5")


def _make_completion(model: str, prompt_excerpt: str) -> dict:
    """Build an OAI-compatible chat.completion response."""
    content = f"[mock tier={model}] response to: {prompt_excerpt}"
    prompt_tokens = max(1, len(prompt_excerpt.split()))
    completion_tokens = max(1, len(content.split()))
    return {
        "id": f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _extract_prompt(messages: list) -> str:
    """Pull the last user message text out of an OAI-style messages array."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "")
    return ""


class MockHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": t, "object": "model", "owned_by": "mock"}
                        for t in KNOWN_TIERS
                    ],
                },
            )
            return
        self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        length = int(self.headers.get("content-length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "invalid json", "type": "bad_request"}})
            return
        model = body.get("model") or "unknown"
        messages = body.get("messages") or []
        prompt = _extract_prompt(messages)
        excerpt = prompt[:80].replace("\n", " ")
        self._send_json(200, _make_completion(model, excerpt))

    def log_message(self, fmt: str, *args) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    # Default well outside the typical 8000-8099 LLM/dashboard range
    # (vllm 8000-8003, vllm-sr dashboard 8700, envoy 8899). Override
    # with `make ... MOCK_PORT=N` if 18811 is also taken.
    parser.add_argument("--port", type=int, default=18811)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        server = ThreadingHTTPServer((args.host, args.port), MockHandler)
    except OSError as e:
        # Most often EADDRINUSE (errno 98). Give a clear next step
        # instead of a bare stack trace.
        log.error(
            "failed to bind %s:%d (%s). Something else is on that port "
            "— run with `make mock-bg MOCK_PORT=N` (and the same for "
            "`make route`/`make answers MOCK=true`) to pick a free one.",
            args.host, args.port, e,
        )
        raise SystemExit(1) from e
    log.info("oai-mock listening on %s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("oai-mock shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
