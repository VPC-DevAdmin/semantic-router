"""OpenAI-compatible chat completion client.

Used by gold generation (M2), the router client (M3 wraps the router's OAI endpoint),
and tier calls during pass execution (M4). Works against any endpoint that speaks
`/v1/chat/completions` — local vLLM, hosted Anthropic/OpenAI, etc.

Auth model:
  - `api_key_env=None`  → no Authorization header (local servers)
  - `api_key_env="X"`, env unset/empty → raise at client construction
  - `api_key_env="X"`, env set        → `Authorization: Bearer <value>`

Multimodal: text and image attachments are supported via the OAI content-array
shape. Audio attachments are not yet supported (M5).
"""
from __future__ import annotations

import base64
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import Attachment, EndpointConfig, TierConfig


@dataclass
class ChatResult:
    content: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int
    raw: dict[str, Any]


class MissingApiKeyError(RuntimeError):
    pass


class OAIClient:
    def __init__(
        self,
        endpoint: str,
        model_id: str,
        *,
        api_key: str | None,
        timeout_s: float,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_id = model_id
        self._api_key = api_key
        self._timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        if self._api_key:
            h["authorization"] = f"Bearer {self._api_key}"
        return h

    async def chat(
        self,
        prompt: str,
        *,
        attachments: list[Attachment] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        system: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        messages = build_messages(prompt, attachments or [], system=system)
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if extra:
            body.update(extra)

        url = f"{self.endpoint}/chat/completions"
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(url, headers=self._headers(), json=body)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        content = choice["message"]["content"]
        # Some servers return content as a list of parts; flatten to text.
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))

        usage = data.get("usage") or {}
        return ChatResult(
            content=content or "",
            model=data.get("model", self.model_id),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            raw=data,
        )


def build_messages(
    prompt: str,
    attachments: list[Attachment],
    *,
    system: str | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})

    if not attachments:
        messages.append({"role": "user", "content": prompt})
        return messages

    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for att in attachments:
        parts.append(_attachment_to_part(att))
    messages.append({"role": "user", "content": parts})
    return messages


def _attachment_to_part(att: Attachment) -> dict[str, Any]:
    if att.type == "image":
        path = Path(att.path)
        if not path.exists():
            raise FileNotFoundError(f"image attachment not found: {path}")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        }
    if att.type == "audio":
        raise NotImplementedError("audio attachments are deferred to M5")
    raise ValueError(f"unknown attachment type: {att.type!r}")


def _resolve_api_key(api_key_env: str | None) -> str | None:
    if api_key_env is None:
        return None
    val = os.environ.get(api_key_env, "").strip()
    if not val:
        raise MissingApiKeyError(
            f"environment variable {api_key_env!r} is unset or empty; "
            f"either set it or remove `api_key_env` from the config for local endpoints"
        )
    return val


def client_from_tier(tier: TierConfig) -> OAIClient:
    return OAIClient(
        endpoint=tier.endpoint,
        model_id=tier.model_id,
        api_key=_resolve_api_key(tier.api_key_env),
        timeout_s=float(tier.timeout_s),
    )


def client_from_endpoint(cfg: EndpointConfig) -> OAIClient:
    return OAIClient(
        endpoint=cfg.endpoint,
        model_id=cfg.model_id,
        api_key=_resolve_api_key(cfg.api_key_env),
        timeout_s=float(cfg.timeout_s),
    )
