"""RouterClient header-extraction and TierLookup tests."""
from __future__ import annotations

import json

import httpx
import pytest

from benchmark.config import ModelsConfig, RouterProcessConfig, TierConfig
from benchmark.router_client import (
    HDR_SELECTED_CATEGORY,
    HDR_SELECTED_MODEL,
    HDR_SELECTED_REASONING,
    RouterClient,
    TierLookup,
)


def _models() -> ModelsConfig:
    return ModelsConfig(
        tiers=[
            TierConfig(
                name="t1",
                level=1,
                endpoint="http://x/v1",
                model_id="phi4",
                api_key_env=None,
                specializations=["general"],
            ),
            TierConfig(
                name="t4",
                level=4,
                endpoint="http://x/v1",
                model_id="DeepSeek-V31",
                api_key_env=None,
                specializations=["general", "code", "math", "reasoning"],
            ),
        ]
    )


def test_tier_lookup_case_insensitive() -> None:
    lookup = TierLookup(_models())
    assert lookup.lookup("phi4") == (1, ["general"])
    assert lookup.lookup("PHI4") == (1, ["general"])
    assert lookup.lookup("deepseek-v31") == (4, ["general", "code", "math", "reasoning"])


def test_tier_lookup_unknown_returns_none() -> None:
    lookup = TierLookup(_models())
    assert lookup.lookup("not-a-model") == (None, None)
    assert lookup.lookup(None) == (None, None)
    assert lookup.lookup("") == (None, None)


def _patch_httpx(monkeypatch, transport: httpx.MockTransport) -> None:
    real_init = httpx.AsyncClient.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = transport
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)


@pytest.mark.asyncio
async def test_extracts_decision_from_headers(monkeypatch) -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={
                HDR_SELECTED_MODEL: "DeepSeek-V31",
                HDR_SELECTED_CATEGORY: "math",
                HDR_SELECTED_REASONING: "on",
            },
            json={
                "model": "DeepSeek-V31",
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            },
        )

    _patch_httpx(monkeypatch, httpx.MockTransport(handler))

    proc_cfg = RouterProcessConfig(frontend_port=8801, auto_model_name="auto")
    client = RouterClient(proc_cfg, TierLookup(_models()))
    result = await client.chat("solve 2x+3=7", max_tokens=64)

    # Wire-format checks.
    assert captured["url"] == "http://127.0.0.1:8801/v1/chat/completions"
    assert captured["body"]["model"] == "auto"
    assert captured["body"]["max_tokens"] == 64

    # Decision extraction.
    d = result.decision
    assert d.selected_model == "DeepSeek-V31"
    assert d.selected_tier == 4
    assert d.selected_specs == ["general", "code", "math", "reasoning"]
    assert d.category == "math"
    assert d.reasoning == "on"
    assert d.cache_hit is False

    assert result.content == "answer"
    assert result.prompt_tokens == 8
    assert result.completion_tokens == 4


@pytest.mark.asyncio
async def test_cache_hit_falls_back_to_body_model(monkeypatch) -> None:
    """No x-vsr-* headers (cache hit). We fall back to body.model."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "phi4",
                "choices": [{"message": {"content": "cached"}}],
            },
        )

    _patch_httpx(monkeypatch, httpx.MockTransport(handler))
    proc_cfg = RouterProcessConfig()
    client = RouterClient(proc_cfg, TierLookup(_models()))
    result = await client.chat("hi")

    d = result.decision
    assert d.cache_hit is True
    assert d.selected_model == "phi4"
    assert d.selected_tier == 1
    assert d.category is None
    assert d.reasoning is None


@pytest.mark.asyncio
async def test_unknown_model_yields_none_tier(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={HDR_SELECTED_MODEL: "exotic-model-not-in-config"},
            json={
                "model": "exotic-model-not-in-config",
                "choices": [{"message": {"content": "x"}}],
            },
        )

    _patch_httpx(monkeypatch, httpx.MockTransport(handler))
    proc_cfg = RouterProcessConfig()
    client = RouterClient(proc_cfg, TierLookup(_models()))
    result = await client.chat("hi")

    d = result.decision
    assert d.selected_model == "exotic-model-not-in-config"
    assert d.selected_tier is None
    assert d.selected_specs is None
    assert d.cache_hit is False  # headers were present
