"""Tier client wire-format and auth tests using httpx.MockTransport."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from benchmark.config import (
    Attachment,
    TierConfig,
    TierModel,
    _env_bool,
    apply_tier_env_overrides,
)
from benchmark.tiers import (
    ChatError,
    MissingApiKeyError,
    OAIClient,
    _resolve_api_key,
    build_messages,
)


def _mock_response(captured: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"message": {"content": "hello world"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            },
        )

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_text_only_request_shape(monkeypatch):
    captured: dict = {}
    transport = _mock_response(captured)

    # Patch httpx.AsyncClient to use our mock transport.
    real_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = transport
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

    client = OAIClient(
        endpoint="https://api.example.com/v1",
        model_id="m1",
        api_key="sk-test",
        timeout_s=5.0,
    )
    result = await client.chat("hi", temperature=0.0, max_tokens=128)

    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer sk-test"
    assert captured["body"]["model"] == "m1"
    assert captured["body"]["temperature"] == 0.0
    assert captured["body"]["max_tokens"] == 128
    # Non-OpenAI endpoints use `max_tokens`, not `max_completion_tokens`.
    assert "max_completion_tokens" not in captured["body"]
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert result.content == "hello world"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 2


@pytest.mark.asyncio
async def test_openai_retries_without_temperature_on_unsupported_value(monkeypatch):
    """OpenAI's gpt-5-nano rejects temperature overrides — retry without
    `temperature` should succeed and the caller never has to special-case."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "temperature" in body:
            return httpx.Response(
                400,
                json={"error": {
                    "message": "Unsupported value: 'temperature' does not "
                               "support 0.0 with this model. Only the "
                               "default (1) value is supported.",
                    "type": "invalid_request_error",
                    "param": "temperature",
                    "code": "unsupported_value",
                }},
            )
        return httpx.Response(
            200,
            json={
                "model": "gpt-5-nano",
                "choices": [{"message": {"content": "pong"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    real_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

    client = OAIClient(
        endpoint="https://api.openai.com/v1",
        model_id="gpt-5-nano", api_key="sk-test", timeout_s=5.0,
    )
    result = await client.chat("ping", max_tokens=16, temperature=0.0)
    # Two POSTs: first with temperature (400), second without (200).
    assert len(calls) == 2
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]
    assert result.content == "pong"


@pytest.mark.asyncio
async def test_openai_endpoint_uses_max_completion_tokens(monkeypatch):
    """OpenAI's GPT-5 / o-series reject `max_tokens`; we must send
    `max_completion_tokens` instead for any api.openai.com endpoint."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gpt-5",
                "choices": [{"message": {"content": "pong"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    real_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

    client = OAIClient(
        endpoint="https://api.openai.com/v1",
        model_id="gpt-5", api_key="sk-test", timeout_s=5.0,
    )
    await client.chat("ping", max_tokens=16)
    assert captured["body"]["max_completion_tokens"] == 16
    assert "max_tokens" not in captured["body"]


def _patch_transport(monkeypatch, transport: httpx.MockTransport) -> None:
    real_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = transport
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)


@pytest.mark.asyncio
async def test_chat_http_error_surfaces_body_and_model_hint(monkeypatch) -> None:
    """A 404 from the server (e.g. wrong model name) must surface the
    server's error body AND a model-mismatch hint, not a bare status code."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": {"message": "The model `Qwen3-30B-A3B` does not exist."}},
        )

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    client = OAIClient(
        endpoint="http://localhost:8002/v1",
        model_id="Qwen3-30B-A3B",
        api_key=None,
        timeout_s=5.0,
    )
    with pytest.raises(ChatError) as ei:
        await client.chat("hi", max_tokens=16)
    msg = str(ei.value)
    assert "HTTP 404" in msg
    assert "Qwen3-30B-A3B" in msg
    assert "does not exist" in msg          # server body surfaced
    assert "model-name mismatch" in msg     # actionable hint
    assert "/v1/models" in msg              # tells them how to check


@pytest.mark.asyncio
async def test_chat_timeout_is_informative(monkeypatch) -> None:
    """A read timeout must say it's a timeout, name the budget, and point
    at the thinking/budget knobs — not surface as an empty ReadTimeout."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    client = OAIClient(
        endpoint="http://localhost:8001/v1",
        model_id="Qwen3-1.7B",
        api_key=None,
        timeout_s=600.0,
    )
    with pytest.raises(ChatError) as ei:
        await client.chat("hi", max_tokens=4096)
    msg = str(ei.value)
    assert "ReadTimeout" in msg
    assert "timeout budget 600s" in msg
    assert "think" in msg.lower()
    assert "MAX_TOKENS" in msg and "THINKING" in msg
    assert "Qwen3-1.7B" in msg


@pytest.mark.asyncio
async def test_chat_connect_error_is_informative(monkeypatch) -> None:
    """A refused connection must say the backend is unreachable, not raise
    a bare httpx.ConnectError the operator has to decode."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    client = OAIClient(
        endpoint="http://localhost:8002/v1",
        model_id="tier2",
        api_key=None,
        timeout_s=5.0,
    )
    with pytest.raises(ChatError) as ei:
        await client.chat("hi", max_tokens=16)
    msg = str(ei.value)
    assert "ConnectError" in msg
    assert "could not reach the backend" in msg
    assert "docker ps" in msg


@pytest.mark.asyncio
async def test_chat_falls_back_to_reasoning_content(monkeypatch) -> None:
    """Qwen3 + reasoning-parser puts a thinking-only answer in
    reasoning_content with content=null; we must surface it, not store ''."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "Qwen3-1.7B",
                "choices": [{
                    "finish_reason": "length",
                    "message": {
                        "content": None,
                        "reasoning_content": "Joyful, glad, cheerful — 'content' is calm.",
                    },
                }],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4096},
            },
        )

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    client = OAIClient(
        endpoint="http://localhost:8001/v1", model_id="Qwen3-1.7B",
        api_key=None, timeout_s=5.0,
    )
    result = await client.chat("What's a synonym for 'happy'?", max_tokens=4096)
    assert result.content == "Joyful, glad, cheerful — 'content' is calm."


@pytest.mark.asyncio
async def test_chat_empty_completion_raises(monkeypatch) -> None:
    """A truly empty completion must be a loud error, not a fake success
    (the q00076 bug: empty answer stored with status='success')."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "Qwen3-1.7B",
                "choices": [{
                    "finish_reason": "length",
                    "message": {"content": "   ", "reasoning_content": None},
                }],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4096},
            },
        )

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    client = OAIClient(
        endpoint="http://localhost:8001/v1", model_id="Qwen3-1.7B",
        api_key=None, timeout_s=5.0,
    )
    with pytest.raises(ChatError) as ei:
        await client.chat("What's a synonym for 'happy'?", max_tokens=4096)
    msg = str(ei.value)
    assert "empty completion" in msg
    assert "finish_reason='length'" in msg
    assert "THINKING=false" in msg and "MAX_TOKENS" in msg


def test_build_messages_with_image(tmp_path: Path) -> None:
    img = tmp_path / "tiny.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)  # not a real png; mime guesser uses .png

    messages = build_messages("describe this", [Attachment(type="image", path=str(img))])
    assert len(messages) == 1
    msg = messages[0]
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    assert msg["content"][0] == {"type": "text", "text": "describe this"}
    img_part = msg["content"][1]
    assert img_part["type"] == "image_url"
    assert img_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_messages_audio_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        build_messages("say it", [Attachment(type="audio", path="/no/such.wav")])


def test_resolve_api_key_unset_env_raises(monkeypatch) -> None:
    monkeypatch.delenv("THIS_IS_DEFINITELY_NOT_SET", raising=False)
    with pytest.raises(MissingApiKeyError):
        _resolve_api_key("THIS_IS_DEFINITELY_NOT_SET")


def test_resolve_api_key_none_returns_none() -> None:
    assert _resolve_api_key(None) is None


def test_resolve_api_key_set(monkeypatch) -> None:
    monkeypatch.setenv("FOO_KEY", "abc123")
    assert _resolve_api_key("FOO_KEY") == "abc123"


# ---- per-tier env overrides: timeout / max_tokens / thinking ----

def _tier(level: int, *, timeout_s: int = 60) -> TierConfig:
    """Minimal TierConfig with no env-discovered slots — used by tests that
    want to drive `apply_tier_env_overrides` from a known empty state."""
    return TierConfig(
        name=f"tier{level}",
        level=level,
        specializations=["general"],
        timeout_s=timeout_s,
        router_alias=f"tier{level}",
    )


def _clear_tier_env(monkeypatch, level: int) -> None:
    suffixes = ("URL", "MODEL", "API_KEY", "TIMEOUT", "MAX_TOKENS", "THINKING", "PROVIDER")
    for suffix in suffixes:
        monkeypatch.delenv(f"TIER{level}_{suffix}", raising=False)
    # Also clear indexed slots 1..4 so a stray TIER{n}_2_MODEL in the
    # dev shell can't make these tests non-deterministic.
    for i in range(1, 5):
        for suffix in suffixes:
            monkeypatch.delenv(f"TIER{level}_{i}_{suffix}", raising=False)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("0", False), ("no", False), ("off", False),
        ("maybe", None), ("", None),
    ],
)
def test_env_bool(raw: str, expected) -> None:
    assert _env_bool(raw) is expected


def test_env_per_slot_timeout_and_max_tokens(monkeypatch) -> None:
    _clear_tier_env(monkeypatch, 1)
    monkeypatch.setenv("TIER1_1_URL", "http://example/v1")
    monkeypatch.setenv("TIER1_1_MODEL", "m1")
    monkeypatch.setenv("TIER1_1_TIMEOUT", "600")
    monkeypatch.setenv("TIER1_1_MAX_TOKENS", "4096")
    t = apply_tier_env_overrides(_tier(1, timeout_s=180))
    assert t.models[0].timeout_s == 600
    assert t.models[0].max_tokens == 4096


def test_env_unset_yields_no_models(monkeypatch) -> None:
    """No env slots → tier.models stays empty; resolved_models() is []."""
    _clear_tier_env(monkeypatch, 2)
    t = apply_tier_env_overrides(_tier(2, timeout_s=300))
    assert t.models == []
    assert t.resolved_models() == []


def test_env_timeout_non_integer_raises(monkeypatch) -> None:
    _clear_tier_env(monkeypatch, 1)
    monkeypatch.setenv("TIER1_1_URL", "http://example/v1")
    monkeypatch.setenv("TIER1_1_MODEL", "m1")
    monkeypatch.setenv("TIER1_1_TIMEOUT", "fast")
    with pytest.raises(ValueError, match="TIER1_1_TIMEOUT must be an integer"):
        apply_tier_env_overrides(_tier(1))


def test_env_thinking_true_creates_extra_body(monkeypatch) -> None:
    _clear_tier_env(monkeypatch, 1)
    monkeypatch.setenv("TIER1_1_URL", "http://example/v1")
    monkeypatch.setenv("TIER1_1_MODEL", "Qwen3-1.7B")
    monkeypatch.setenv("TIER1_1_THINKING", "true")
    t = apply_tier_env_overrides(_tier(1))
    assert t.models[0].extra_body == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_env_thinking_invalid_raises(monkeypatch) -> None:
    _clear_tier_env(monkeypatch, 1)
    monkeypatch.setenv("TIER1_1_URL", "http://example/v1")
    monkeypatch.setenv("TIER1_1_MODEL", "m1")
    monkeypatch.setenv("TIER1_1_THINKING", "sometimes")
    with pytest.raises(ValueError, match="TIER1_1_THINKING must be a boolean"):
        apply_tier_env_overrides(_tier(1))


# ---- multiple models per tier (indexed slots only — no slot 0) ----

def test_bare_tier_env_var_raises_with_migration_hint(monkeypatch) -> None:
    """A stale single-model TIER{N}_* env var must fail loud, not silently."""
    _clear_tier_env(monkeypatch, 1)
    monkeypatch.setenv("TIER1_MODEL", "leftover-bare-model")
    with pytest.raises(ValueError, match="TIER1_MODEL is not a supported env var"):
        apply_tier_env_overrides(_tier(1))


def test_env_indexed_slots_with_providers(monkeypatch) -> None:
    """Indexed slots add models; each carries its optional provider label."""
    _clear_tier_env(monkeypatch, 5)
    monkeypatch.setenv("TIER5_1_URL", "https://api.anthropic.com/v1")
    monkeypatch.setenv("TIER5_1_MODEL", "claude-opus-4-7")
    monkeypatch.setenv("TIER5_1_API_KEY", "sk-ant-xxx")
    monkeypatch.setenv("TIER5_1_PROVIDER", "Anthropic")
    monkeypatch.setenv("TIER5_2_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("TIER5_2_MODEL", "gpt-5")
    monkeypatch.setenv("TIER5_2_API_KEY", "sk-openai")
    monkeypatch.setenv("TIER5_2_PROVIDER", "OpenAI")

    t = apply_tier_env_overrides(_tier(5))
    assert [(m.slot, m.served_model_name, m.provider, m.url) for m in t.models] == [
        (1, "claude-opus-4-7", "Anthropic", "https://api.anthropic.com/v1"),
        (2, "gpt-5", "OpenAI", "https://api.openai.com/v1"),
    ]
    assert t.models[0].api_key_env == "TIER5_1_API_KEY"
    assert t.models[1].api_key_env == "TIER5_2_API_KEY"


def test_env_indexed_slots_stop_at_gap(monkeypatch) -> None:
    """Discovery stops at the first missing slot — a gap ends the list."""
    _clear_tier_env(monkeypatch, 3)
    monkeypatch.setenv("TIER3_1_URL", "http://example/v1")
    monkeypatch.setenv("TIER3_1_MODEL", "m1")
    monkeypatch.setenv("TIER3_2_URL", "http://example/v1")
    monkeypatch.setenv("TIER3_2_MODEL", "m2")
    # no slot 3
    monkeypatch.setenv("TIER3_4_URL", "http://example/v1")
    monkeypatch.setenv("TIER3_4_MODEL", "m4")  # unreachable: gap at 3
    t = apply_tier_env_overrides(_tier(3))
    assert [m.served_model_name for m in t.models] == ["m1", "m2"]


def test_env_slot_url_without_model_raises(monkeypatch) -> None:
    _clear_tier_env(monkeypatch, 3)
    monkeypatch.setenv("TIER3_1_URL", "https://x/v1")  # URL but no MODEL
    with pytest.raises(ValueError, match="TIER3_1_MODEL is not"):
        apply_tier_env_overrides(_tier(3))


def test_env_slot_model_without_url_raises(monkeypatch) -> None:
    """Both URL and MODEL are required per slot — no YAML fallback for URL."""
    _clear_tier_env(monkeypatch, 3)
    monkeypatch.setenv("TIER3_1_MODEL", "model-with-no-url")
    with pytest.raises(ValueError, match="TIER3_1_URL is not"):
        apply_tier_env_overrides(_tier(3))


def test_env_duplicate_model_name_within_tier_raises(monkeypatch) -> None:
    _clear_tier_env(monkeypatch, 3)
    monkeypatch.setenv("TIER3_1_URL", "http://a/v1")
    monkeypatch.setenv("TIER3_1_MODEL", "dup")
    monkeypatch.setenv("TIER3_2_URL", "http://b/v1")
    monkeypatch.setenv("TIER3_2_MODEL", "dup")
    with pytest.raises(ValueError, match="duplicate model name 'dup'"):
        apply_tier_env_overrides(_tier(3))


def test_env_per_slot_thinking_and_budget(monkeypatch) -> None:
    """Per-slot TIMEOUT/MAX_TOKENS/THINKING; slots inherit tier defaults
    independently for fields they don't override."""
    _clear_tier_env(monkeypatch, 1)
    monkeypatch.setenv("TIER1_1_URL", "http://example/v1")
    monkeypatch.setenv("TIER1_1_MODEL", "Qwen3-1.7B")
    monkeypatch.setenv("TIER1_2_URL", "http://example/v1")
    monkeypatch.setenv("TIER1_2_MODEL", "Qwen3-1.7B-think")
    monkeypatch.setenv("TIER1_2_THINKING", "true")
    monkeypatch.setenv("TIER1_2_MAX_TOKENS", "8192")
    # tier timeout 300 — slot 2 omits TIMEOUT → inherits.
    t = apply_tier_env_overrides(_tier(1, timeout_s=300))
    s1, s2 = t.models
    assert s1.served_model_name == "Qwen3-1.7B"
    assert s2.timeout_s == 300          # inherited from tier
    assert s2.max_tokens == 8192        # per-slot override
    assert s2.extra_body["chat_template_kwargs"]["enable_thinking"] is True
    # slot 1 untouched by slot 2's THINKING flag.
    assert s1.extra_body is None


def test_tier_model_is_pydantic() -> None:
    m = TierModel(slot=1, url="http://x/v1", served_model_name="m")
    assert m.api_key_env is None and m.provider is None and m.max_tokens is None


# ---- recipe-merge: local_models.yaml extras into localhost slots ----

def test_recipe_extras_merge_only_into_localhost_slots() -> None:
    """A recipe's extra_body (e.g. Qwen3 sampler) merges into the slot's
    extra_body when the slot hits localhost AND the recipe exists.
    Vendor slots (non-localhost) are not touched — even if the recipe
    exists, the vendor wouldn't accept its fields."""
    from benchmark.config import (
        LocalLauncherSpec,
        LocalModelLibrary,
        LocalModelRecipe,
        apply_recipe_extras,
    )

    library = LocalModelLibrary(recipes={
        "Qwen3-1.7B": LocalModelRecipe(
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "top_k": 20,
            },
            amd=LocalLauncherSpec(start=["true"], stop=["true"]),
        ),
    })

    tier = _tier(1)
    tier.models = [
        TierModel(
            slot=1, url="http://localhost:8001/v1",
            served_model_name="Qwen3-1.7B",
        ),
        TierModel(
            slot=2, url="https://api.openai.com/v1",
            served_model_name="Qwen3-1.7B",  # same name, but vendor URL
        ),
    ]
    apply_recipe_extras(tier, library)

    # localhost slot picked up the recipe's extras.
    assert tier.models[0].extra_body == {
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": 20,
    }
    # vendor slot left untouched.
    assert tier.models[1].extra_body is None


def test_recipe_extras_slot_env_wins_on_leaf_conflict() -> None:
    """A per-slot env override (TIER{N}_{i}_THINKING=true) must win over
    the recipe's default (enable_thinking=false) on a deep-merge."""
    from benchmark.config import (
        LocalLauncherSpec,
        LocalModelLibrary,
        LocalModelRecipe,
        apply_recipe_extras,
    )

    library = LocalModelLibrary(recipes={
        "Qwen3-1.7B": LocalModelRecipe(
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "top_k": 20,  # sibling — recipe wins (no slot override)
            },
            amd=LocalLauncherSpec(start=["true"], stop=["true"]),
        ),
    })

    tier = _tier(1)
    tier.models = [
        TierModel(
            slot=1, url="http://localhost:8001/v1",
            served_model_name="Qwen3-1.7B",
            # Slot already has the per-slot THINKING env override applied.
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        ),
    ]
    apply_recipe_extras(tier, library)

    merged = tier.models[0].extra_body
    # Leaf conflict: slot's enable_thinking=True wins.
    assert merged["chat_template_kwargs"]["enable_thinking"] is True
    # Sibling key from recipe survives.
    assert merged["top_k"] == 20
