"""Client for the router's OpenAI-compatible frontend.

The router exposes `POST /v1/chat/completions` on its Envoy frontend (default :8801).
Sending `model: "auto"` invokes routing. The actual selected model is returned
in the `x-vsr-selected-model` response header (also reflected in the response
body's `model` field). Two other headers carry routing context:

  - x-vsr-selected-category    e.g. "math", "computer science"
  - x-vsr-selected-reasoning   "on" | "off"
  - x-vsr-selected-model       e.g. "deepseek-v31"

Headers are added only on 2xx responses that did not hit the cache. Cache hits
won't have them; we treat that as `selected_model = response.body.model` fallback.

Pass-1 (routing accuracy) and Pass-2 (response quality) both use this client;
Pass 1 caps `max_tokens` to minimize generation cost.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import Attachment, ModelsConfig, RouterProcessConfig, load_models
from .tiers import build_messages

# Header names are lower-cased by httpx on read.
HDR_SELECTED_MODEL = "x-vsr-selected-model"
HDR_SELECTED_CATEGORY = "x-vsr-selected-category"
HDR_SELECTED_REASONING = "x-vsr-selected-reasoning"


@dataclass
class RoutingDecision:
    selected_model: str | None
    selected_tier: int | None
    selected_specs: list[str] | None
    category: str | None
    reasoning: str | None  # "on" | "off" | None
    cache_hit: bool  # True when we couldn't read x-vsr headers (cache or missing)


@dataclass
class RouterResult:
    content: str
    decision: RoutingDecision
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int
    raw_body: dict[str, Any]
    raw_headers: dict[str, str]


class TierLookup:
    """Map a router-selected model name to a tier level + specializations.

    Lookup is case-insensitive on `model_id`. Unknown models return (None, None)
    and are surfaced in pass-1 reporting so the operator can extend models.yaml.
    """

    def __init__(self, models: ModelsConfig) -> None:
        self._by_model_id: dict[str, tuple[int, list[str]]] = {}
        for tier in models.tiers:
            self._by_model_id[tier.model_id.lower()] = (tier.level, list(tier.specializations))

    def lookup(self, model_name: str | None) -> tuple[int | None, list[str] | None]:
        if not model_name:
            return None, None
        hit = self._by_model_id.get(model_name.lower())
        if hit is None:
            return None, None
        return hit


class RouterClient:
    def __init__(
        self,
        proc_cfg: RouterProcessConfig,
        tier_lookup: TierLookup,
        *,
        timeout_s: float = 180.0,
    ) -> None:
        self._cfg = proc_cfg
        self._lookup = tier_lookup
        self._timeout_s = timeout_s

    @property
    def base_url(self) -> str:
        return f"http://{self._cfg.frontend_host}:{self._cfg.frontend_port}"

    async def chat(
        self,
        prompt: str,
        *,
        attachments: list[Attachment] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        extra_body: dict[str, Any] | None = None,
    ) -> RouterResult:
        messages = build_messages(prompt, attachments or [])
        body: dict[str, Any] = {
            "model": self._cfg.auto_model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            # `max_completion_tokens` is the post-2025 OpenAI field. OpenAI's
            # gpt-5.x reasoning models REJECT the legacy `max_tokens`
            # ("Unsupported parameter: 'max_tokens' is not supported with
            # this model. Use 'max_completion_tokens' instead.") and the
            # vllm-sr Anthropic adapter (pkg/anthropic/client.go) explicitly
            # prefers `MaxCompletionTokens` over `MaxTokens`. Modern vLLM
            # and Google's OAI-compat endpoint accept it too. Since the
            # router chooses the upstream and we can't pre-pick the right
            # field, send the universally-accepted one.
            body["max_completion_tokens"] = max_tokens
        if extra_body:
            body.update(extra_body)

        url = f"{self.base_url}/v1/chat/completions"
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(url, json=body)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        resp.raise_for_status()

        data = resp.json()
        headers = {k.lower(): v for k, v in resp.headers.items()}

        decision = self._extract_decision(headers, data)
        choice = data["choices"][0]
        content = choice["message"]["content"]
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))

        usage = data.get("usage") or {}
        return RouterResult(
            content=content or "",
            decision=decision,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            raw_body=data,
            raw_headers=headers,
        )

    def _extract_decision(
        self, headers: dict[str, str], body: dict[str, Any]
    ) -> RoutingDecision:
        selected = headers.get(HDR_SELECTED_MODEL)
        category = headers.get(HDR_SELECTED_CATEGORY)
        reasoning = headers.get(HDR_SELECTED_REASONING)
        cache_hit = selected is None
        # Fallback: response body's `model` field reflects routed model on cache hits.
        if selected is None:
            selected = body.get("model")
        tier, specs = self._lookup.lookup(selected)
        return RoutingDecision(
            selected_model=selected,
            selected_tier=tier,
            selected_specs=specs,
            category=category,
            reasoning=reasoning,
            cache_hit=cache_hit,
        )


def make_tier_lookup(models_yaml: Path) -> TierLookup:
    return TierLookup(load_models(models_yaml))
