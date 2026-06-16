"""Live interactive demo backend (routes via vllm-sr). Tests the verifiable
logic: overlay key masking/preservation, query grouping, the vllm-sr response
parsing, and the overlay→exemplars translation. The live vllm-sr proxy + reload
require a running stack and are exercised on the box, not here."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import interactive_server as srv  # noqa: E402

OVERLAY = {
    "vllm_sr_url": "http://localhost:8801",
    "tiers": [
        {"id": "tier1", "name": "Tier 1", "model": "m1", "provider": "Google",
         "base_url": "b1", "api_key": "SECRET1"},
        {"id": "tier2", "name": "Tier 2", "model": "m2", "provider": "OpenAI",
         "base_url": "b2", "api_key": ""},
    ],
    "tier_cutoffs": [0.1, 0.2],
    "signals": [{"id": "trivial_lookup", "weight": -0.2, "threshold": 0.65,
                 "description": "d", "candidates": ["who painted X?", "capital of Y?"]}],
}


def test_masked_overlay_hides_keys_and_flags():
    m = srv.masked_overlay(OVERLAY)
    assert m["tiers"][0]["api_key"] == "" and m["tiers"][0]["key_set"] is True
    assert m["tiers"][1]["api_key"] == "" and m["tiers"][1]["key_set"] is False
    # original not mutated
    assert OVERLAY["tiers"][0]["api_key"] == "SECRET1"


def test_merge_overlay_preserves_blank_keys(tmp_path, monkeypatch):
    user = tmp_path / "live_demo.local.json"
    monkeypatch.setattr(srv, "USER_OVERLAY", user)
    monkeypatch.setattr(srv, "DEFAULT_OVERLAY", ROOT / "config" / "live_demo.json")
    # seed an existing saved overlay with a key
    user.write_text(json.dumps({"tiers": [{"id": "tier1", "api_key": "KEEPME"}]}))
    incoming = {"vllm_sr_url": "x", "tiers": [{"id": "tier1", "name": "T1", "api_key": "",
                "key_set": True}]}
    srv.merge_overlay(incoming)
    saved = json.loads(user.read_text())
    assert saved["tiers"][0]["api_key"] == "KEEPME"   # blank incoming preserved existing
    assert "key_set" not in saved["tiers"][0]          # stripped before persist


def test_grouped_queries_shape():
    g = srv.grouped_queries()
    assert "categories" in g and g["categories"]
    # every category maps to a capped list of prompt strings
    for _cat, prompts in g["categories"].items():
        assert isinstance(prompts, list) and len(prompts) <= 25
        assert all(isinstance(p, str) for p in prompts)


def test_build_live_exemplars_applies_edits(tmp_path, monkeypatch):
    out = tmp_path / "live_exemplars.local.yaml"
    monkeypatch.setattr(srv, "LIVE_EXEMPLARS", out)
    srv.build_live_exemplars(OVERLAY)
    import yaml
    ex = yaml.safe_load(out.read_text())
    assert ex["tier_cutoffs"] == [0.1, 0.2]            # overlay cutoffs applied
    sig = next(s for s in ex["embedding_signals"] if s["id"] == "trivial_lookup")
    assert sig["weight"] == -0.2
    assert sig["candidates"] == ["who painted X?", "capital of Y?"]   # bank replaced


class _Resp:
    def __init__(self, headers, body, status_code=200, text=""):
        self.headers = headers
        self._body = body
        self.status_code = status_code
        self.text = text
    def raise_for_status(self): pass
    def json(self): return self._body


def test_vllm_chat_parses_decision(monkeypatch):
    body = {"model": "tier3", "choices": [{"message": {"content": "the answer"}}]}
    headers = {"x-vsr-selected-model": "tier3", "x-vsr-selected-category": "math",
               "x-vsr-selected-reasoning": "on"}

    class _Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, json): return _Resp(headers, body)

    monkeypatch.setattr(srv.httpx, "Client", _Client)
    ov = {"vllm_sr_url": "http://localhost:8801",
          "tiers": [{"id": "tier3", "name": "Tier 3", "model": "gpt-5.4-mini"}]}
    out = srv.vllm_chat(ov, "what is 2+2 with a twist", "auto")
    assert out["answer"] == "the answer"
    r = out["routing"]
    assert r["selected_tier_id"] == "tier3"
    assert r["selected_tier_name"] == "Tier 3"
    assert r["served_model"] == "gpt-5.4-mini"
    assert r["category"] == "math" and r["reasoning"] == "on"
    assert r["forced"] is False


def test_vllm_chat_unreachable_returns_error(monkeypatch):
    class _Boom:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, *a, **k): raise srv.httpx.ConnectError("refused")

    monkeypatch.setattr(srv.httpx, "Client", _Boom)
    out = srv.vllm_chat({"vllm_sr_url": "http://localhost:8801", "tiers": []}, "q", "auto")
    assert "not reachable" in out["error"]


def _models_client(captured, body):
    class _C:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url, headers):
            captured["url"] = url
            captured["headers"] = headers
            return _Resp({}, body)
    return _C


def test_list_models_openai_bearer_and_sorted(monkeypatch):
    cap = {}
    monkeypatch.setattr(srv.httpx, "Client",
                        _models_client(cap, {"data": [{"id": "gpt-x"}, {"id": "gpt-a"}]}))
    out = srv.list_models({"tier_id": "tier2", "provider": "OpenAI",
                           "base_url": "https://api.openai.com/v1", "api_key": "sk-x"})
    assert out["models"] == ["gpt-a", "gpt-x"]            # sorted, deduped
    assert cap["url"] == "https://api.openai.com/v1/models"
    assert cap["headers"]["Authorization"] == "Bearer sk-x"


def test_list_models_anthropic_uses_x_api_key(monkeypatch):
    cap = {}
    monkeypatch.setattr(srv.httpx, "Client",
                        _models_client(cap, {"data": [{"id": "claude-opus-4-7"}]}))
    out = srv.list_models({"provider": "Anthropic",
                           "base_url": "https://api.anthropic.com/v1", "api_key": "k"})
    assert out["models"] == ["claude-opus-4-7"]
    assert cap["headers"]["x-api-key"] == "k"
    assert "anthropic-version" in cap["headers"]


def test_list_models_no_key_is_friendly_error():
    out = srv.list_models({"provider": "OpenAI",
                           "base_url": "https://api.openai.com/v1", "api_key": ""})
    assert "API key" in out["error"]


def test_vllm_chat_upstream_error_surfaces_reason(monkeypatch):
    # The router routed fine but the upstream model 404'd (e.g. a bad model id).
    # The UI must show the upstream reason + the routed tier, NOT "not reachable".
    err_body = {"error": {"message": "The model `gpt-5.4-nano` does not exist"}}
    headers = {"x-vsr-selected-model": "tier2"}

    class _Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, json): return _Resp(headers, err_body, status_code=404)

    monkeypatch.setattr(srv.httpx, "Client", _Client)
    ov = {"vllm_sr_url": "http://localhost:8899",
          "tiers": [{"id": "tier2", "name": "Tier 2", "model": "gpt-5.4-nano"}]}
    out = srv.vllm_chat(ov, "q", "auto")
    assert "not reachable" not in out["error"]
    assert "404" in out["error"]
    assert "does not exist" in out["error"]
    assert "tier2" in out["error"]
