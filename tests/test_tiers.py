"""Tier client wire-format and auth tests using httpx.MockTransport."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from benchmark.config import (
    Attachment,
    BackendSpec,
    TierConfig,
    TierEndpoint,
    _env_bool,
    apply_tier_env_overrides,
)
from benchmark.tiers import (
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
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert result.content == "hello world"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 2


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

def _tier(level: int, *, timeout_s: int = 60, extra_body: dict | None = None) -> TierConfig:
    backend_kwargs: dict = {"kind": "remote"}
    if extra_body is not None:
        backend_kwargs["extra_body"] = extra_body
    return TierConfig(
        name=f"tier{level}",
        level=level,
        specializations=["general"],
        timeout_s=timeout_s,
        router_alias=f"tier{level}",
        served_model_name=f"tier{level}",
        endpoint=TierEndpoint(url=f"http://localhost:800{level}/v1"),
        backend=BackendSpec(**backend_kwargs),
    )


def _clear_tier_env(monkeypatch, level: int) -> None:
    for suffix in ("URL", "MODEL", "API_KEY", "TIMEOUT", "MAX_TOKENS", "THINKING"):
        monkeypatch.delenv(f"TIER{level}_{suffix}", raising=False)


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


def test_env_override_timeout_and_max_tokens(monkeypatch) -> None:
    _clear_tier_env(monkeypatch, 1)
    monkeypatch.setenv("TIER1_TIMEOUT", "600")
    monkeypatch.setenv("TIER1_MAX_TOKENS", "4096")
    t = apply_tier_env_overrides(_tier(1, timeout_s=180))
    assert t.timeout_s == 600
    assert t.max_tokens == 4096


def test_env_override_unset_keeps_yaml_defaults(monkeypatch) -> None:
    _clear_tier_env(monkeypatch, 2)
    t = apply_tier_env_overrides(_tier(2, timeout_s=300))
    assert t.timeout_s == 300
    assert t.max_tokens is None  # no TIER2_MAX_TOKENS → stays unset


def test_env_override_timeout_non_integer_raises(monkeypatch) -> None:
    _clear_tier_env(monkeypatch, 1)
    monkeypatch.setenv("TIER1_TIMEOUT", "fast")
    with pytest.raises(ValueError, match="TIER1_TIMEOUT must be an integer"):
        apply_tier_env_overrides(_tier(1))


def test_env_thinking_true_creates_extra_body(monkeypatch) -> None:
    _clear_tier_env(monkeypatch, 1)
    monkeypatch.setenv("TIER1_THINKING", "true")
    t = apply_tier_env_overrides(_tier(1))  # no extra_body in YAML
    assert t.backend.model_dump()["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_env_thinking_false_overrides_yaml_and_preserves_siblings(monkeypatch) -> None:
    _clear_tier_env(monkeypatch, 1)
    monkeypatch.setenv("TIER1_THINKING", "false")
    # YAML had enable_thinking=true plus an unrelated sibling key.
    t = _tier(1, extra_body={
        "chat_template_kwargs": {"enable_thinking": True, "other": 1},
        "top_p": 0.9,
    })
    apply_tier_env_overrides(t)
    dumped = t.backend.model_dump()["extra_body"]
    assert dumped["chat_template_kwargs"] == {"enable_thinking": False, "other": 1}
    assert dumped["top_p"] == 0.9  # unrelated extra_body keys untouched


def test_env_thinking_invalid_raises(monkeypatch) -> None:
    _clear_tier_env(monkeypatch, 1)
    monkeypatch.setenv("TIER1_THINKING", "sometimes")
    with pytest.raises(ValueError, match="TIER1_THINKING must be a boolean"):
        apply_tier_env_overrides(_tier(1))
