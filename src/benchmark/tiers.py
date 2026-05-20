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

from .config import Attachment, TierConfig, TierModel


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


class ChatError(RuntimeError):
    """A chat call failed, with a message that says *what*, *where*, and *why*.

    The answers worker stores `str(e)` verbatim into `tier_answers.error_msg`,
    so the richer this is, the less the operator has to spelunk docker logs.
    Distinguishes the three failure modes that actually happen here:
      - timeout (backend accepted but didn't finish — usually a <think>
        chain eating the whole token budget on CPU),
      - HTTP 4xx/5xx with the server's error body surfaced (e.g. a
        model-name mismatch → 404),
      - connection refused (backend not up).
    """


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
            # OpenAI's GPT-5 / o-series reasoning models require
            # `max_completion_tokens` instead of `max_tokens` (and
            # reject the latter with HTTP 400). Older OpenAI models
            # accept either, so it's safe to always use the new name
            # against api.openai.com. Other providers still use
            # `max_tokens` (Google's OAI-compat rejects unknown fields,
            # so we can't just send both).
            if "api.openai.com" in (self.endpoint or ""):
                body["max_completion_tokens"] = max_tokens
            else:
                body["max_tokens"] = max_tokens
        if extra:
            body.update(extra)

        url = f"{self.endpoint}/chat/completions"
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(url, headers=self._headers(), json=body)
                # OpenAI's gpt-5-nano (and possibly others in the reasoning
                # line) reject any temperature override — only the default
                # (1.0) is supported and a non-default value 400s. Retry
                # once without temperature so the caller doesn't have to
                # special-case it per model.
                if (
                    resp.status_code == 400
                    and "temperature" in body
                    and "temperature" in resp.text.lower()
                    and "does not support" in resp.text.lower()
                ):
                    body.pop("temperature")
                    resp = await client.post(url, headers=self._headers(), json=body)
        except httpx.TimeoutException as e:
            # ConnectTimeout subclasses both TimeoutException and ConnectError;
            # catching TimeoutException first reports it as the timeout it is.
            elapsed = time.perf_counter() - t0
            raise ChatError(
                f"{type(e).__name__} after {elapsed:.0f}s "
                f"(timeout budget {self._timeout_s:.0f}s) calling {url} "
                f"model={self.model_id!r}. The backend did not respond in "
                f"time — most often it accepted the request and is still "
                f"generating: a Qwen3 <think> chain can consume the entire "
                f"max_tokens budget at CPU speed before the visible answer. "
                f"Raise this tier's TIMEOUT / MAX_TOKENS in .env, or set "
                f"TIER{{N}}_THINKING=false, then retry."
            ) from e
        except httpx.ConnectError as e:
            raise ChatError(
                f"ConnectError calling {url} model={self.model_id!r}: could "
                f"not reach the backend — is it up? Check `docker ps` and "
                f"`make start_LLM`, and that the URL/port are right. "
                f"Underlying: {e!r}"
            ) from e
        except httpx.RequestError as e:
            elapsed = time.perf_counter() - t0
            raise ChatError(
                f"{type(e).__name__} after {elapsed:.0f}s calling {url} "
                f"model={self.model_id!r}: {e!r}"
            ) from e

        latency_ms = int((time.perf_counter() - t0) * 1000)

        if resp.status_code >= 400:
            excerpt = " ".join(resp.text.split())
            if len(excerpt) > 600:
                excerpt = excerpt[:600] + "…"
            hint = ""
            if resp.status_code in (400, 404) and "model" in excerpt.lower():
                hint = (
                    f" — this looks like a model-name mismatch: the request "
                    f"sent model={self.model_id!r}, which must exactly match "
                    f"what the server serves (`curl {self.endpoint}/models`)."
                )
            raise ChatError(
                f"HTTP {resp.status_code} from {url} model={self.model_id!r} "
                f"after {latency_ms} ms: {excerpt or '(empty response body)'}"
                f"{hint}"
            )

        try:
            data = resp.json()
        except ValueError as e:
            raise ChatError(
                f"HTTP {resp.status_code} from {url} model={self.model_id!r} "
                f"but the body was not JSON: {resp.text[:300]!r}"
            ) from e

        choice = data["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content")
        # Some servers return content as a list of parts; flatten to text.
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        content = (content or "").strip()

        usage = data.get("usage") or {}
        finish_reason = choice.get("finish_reason")

        # Qwen3 with `--reasoning-parser qwen3`: the <think> chain is split
        # into message.reasoning_content and only the post-</think> text
        # lands in message.content. If the model burns the whole token
        # budget inside <think> (finish_reason="length", no closing tag)
        # the parser puts *everything* in reasoning_content and content is
        # null — which we were silently storing as an empty "success"
        # (this is the q00076 bug). Fall back to the reasoning stream so a
        # thinking-only completion still yields its substance.
        if not content:
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                content = reasoning.strip()

        # Still nothing → a genuinely empty completion. Make it a loud
        # error (retried on the next pass, visible in the running list)
        # instead of a fake success.
        if not content:
            raise ChatError(
                f"empty completion from {url} model={self.model_id!r} "
                f"(finish_reason={finish_reason!r}, "
                f"completion_tokens={usage.get('completion_tokens')}, "
                f"{latency_ms} ms): the model returned no content and no "
                f"reasoning_content. If finish_reason='length' the budget "
                f"was exhausted — most often a Qwen3 <think> chain that "
                f"never closed. Disable thinking for this tier "
                f"(TIER{{N}}_THINKING=false) or raise MAX_TOKENS."
            )

        return ChatResult(
            content=content,
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
    """Back-compat: a client for the tier's slot-0 / legacy single model."""
    return OAIClient(
        endpoint=tier.endpoint.url,
        model_id=tier.served_model_name,
        api_key=_resolve_api_key(tier.endpoint.api_key_env),
        timeout_s=float(tier.timeout_s),
    )


def client_from_model(model: TierModel) -> OAIClient:
    """A client for one specific model slot within a tier."""
    return OAIClient(
        endpoint=model.url,
        model_id=model.served_model_name,
        api_key=_resolve_api_key(model.api_key_env),
        timeout_s=float(model.timeout_s),
    )
