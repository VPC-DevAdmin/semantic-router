"""Interactive demo backend: tier scoring (closest exemplar + chosen tier +
reasoning) and the no-API-key chat path."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import interactive_server as srv  # noqa: E402


TIERS = [
    {"id": "tier1", "name": "Tier 1", "threshold": 0.3,
     "exemplars": ["What is 17 + 26?", "Capital of France?", "Define photosynthesis"]},
    {"id": "tier2", "name": "Tier 2", "threshold": 0.3,
     "exemplars": ["Summarize this paragraph in two sentences", "Convert 5km to miles"]},
    {"id": "tier3", "name": "Tier 3", "threshold": 0.3,
     "exemplars": ["Analyze the tradeoffs of a microservices migration strategy",
                   "Design a caching layer for a high-traffic API"]},
]


def test_cosine_basic():
    a, b = srv._tokens("the quick brown fox"), srv._tokens("the quick brown fox")
    assert srv._cosine(a, b) == 1.0
    assert srv._cosine(srv._tokens("apple"), srv._tokens("orange")) == 0.0


def test_score_tiers_picks_closest_and_reports_exemplar():
    out = srv.score_tiers("What is 42 + 58?", TIERS)
    assert out["chosen_id"] == "tier1"                       # arithmetic → Tier 1
    t1 = next(t for t in out["tiers"] if t["id"] == "tier1")
    assert "+" in (t1["closest_exemplar"] or "")             # matched the math exemplar
    assert out["tiers"] and "reasoning" in out and out["reasoning"]
    assert out["ranked_ids"][0] == "tier1"


def test_score_tiers_routes_complex_to_higher_tier():
    out = srv.score_tiers("Analyze the tradeoffs of a microservices migration", TIERS)
    assert out["chosen_id"] == "tier3"
    chosen = next(t for t in out["tiers"] if t["id"] == "tier3")
    assert "microservices" in (chosen["closest_exemplar"] or "").lower()


def test_chat_without_key_returns_no_api_key():
    payload = {"query": "What is 2+2?", "mode": "auto",
               "tiers": [{**t, "model": "m", "provider": "OpenAI", "api_key": ""} for t in TIERS]}
    res = srv.handle_chat(payload)
    assert res["no_api_key"] is True
    assert res["routing"]["chosen_id"]            # routing still computed
    assert res["tier"]["id"]                       # served tier reported


def test_chat_forced_mode_overrides_routing():
    payload = {"query": "What is 2+2?", "mode": "tier3",   # force Tier 3
               "tiers": [{**t, "model": "m", "provider": "OpenAI", "api_key": ""} for t in TIERS],
               "max_tier_id": "tier3"}
    res = srv.handle_chat(payload)
    assert res["tier"]["id"] == "tier3"            # forced served tier
    assert res["routing"]["chosen_id"] == "tier1"  # auto would have picked Tier 1
    assert res["routing"]["forced"] is True
