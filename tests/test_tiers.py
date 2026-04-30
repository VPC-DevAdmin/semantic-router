"""Tier client wire-format and auth tests using httpx.MockTransport."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from benchmark.config import Attachment
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
