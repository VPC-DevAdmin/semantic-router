"""Contract gateway: tier clamp, route-decision string, cost, role resolution,
the generic schema-instance generator, and an end-to-end strict-schema request."""
from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import router_gateway as gw  # noqa: E402


ROLES_CFG = {
    "roles": {
        "planner": {"mode": "pinned", "tier": "L5"},
        "worker": {"mode": "route"},
    },
    "default_role": "worker",
    "tiers": {
        "L1": {"model": "tiny", "price": "tiny"},
        "L2": {"model": "small", "price": "small"},
        "L3": {"model": "mid", "price": "mid"},
        "L4": {"model": "big", "price": "big"},
        "L5": {"model": "frontier", "price": "frontier"},
    },
}
PRICING = {
    "tiny": {"self_hosted": True},
    "small": {"in_per_1m": 0.2, "out_per_1m": 1.25},
    "frontier": {"in_per_1m": 5.0, "out_per_1m": 25.0},
}


def test_clamp_tier():
    assert gw.clamp_tier("L2", "L3") == "L3"      # floor raises it
    assert gw.clamp_tier("L4", "L3") == "L4"      # already above floor
    assert gw.clamp_tier("L2", None) == "L2"      # no floor
    assert gw.clamp_tier("L2", "bogus") == "L2"   # bad floor ignored


def test_route_decision_visibility():
    # No clamp: classified == served, min absent.
    assert gw.route_decision("worker", "small", classified="L2", served="L2") \
        == "worker@small?classified=L2&served=L2"
    # Clamp fired: min is shown, making the escalation visible.
    assert gw.route_decision("worker", "mid", classified="L2", served="L3",
                             min_tier="L3") == "worker@mid?classified=L2&min=L3&served=L3"
    # Pinned role: just role@model.
    assert gw.route_decision("planner", "frontier") == "planner@frontier"


def test_cost_usd():
    assert gw.cost_usd(1000, 1000, {"in_per_1m": 1.0, "out_per_1m": 2.0}) == 0.003
    assert gw.cost_usd(1000, 1000, {"self_hosted": True}) == 0.0
    assert gw.cost_usd(1000, 1000, None) == 0.0


def test_resolve_role_falls_back_to_default():
    assert gw.resolve_role("planner", ROLES_CFG)[0] == "planner"
    # unknown role -> default_role
    name, behavior = gw.resolve_role("summarizer", ROLES_CFG)
    assert name == "worker" and behavior["mode"] == "route"


class _Scores(BaseModel):
    coverage: float = Field(0.0, ge=0.0, le=1.0)


class _Plan(BaseModel):
    plan_id: str
    n: int
    flags: list[str] = []
    scores: _Scores
    ok: bool


def test_instance_from_schema_validates_against_pydantic():
    schema = _Plan.model_json_schema()
    inst = gw.instance_from_schema(schema)
    # The generated minimal instance must validate against the model — this is
    # exactly the strict guided_json guarantee, for any schema.
    _Plan.model_validate(inst)


def test_gateway_worker_clamp_and_headers():
    g = gw.Gateway(ROLES_CFG, PRICING)
    body = {"model": "worker",
            "messages": [{"role": "user", "content": "hi"}],   # short -> L1 classify
            "metadata": {"min_tier": "L3"}}
    resp, hdrs = g.handle(body)
    # classified L1, floored to L3
    assert "classified=L1" in hdrs["x-llm-route-decision"]
    assert "served=L3" in hdrs["x-llm-route-decision"]
    assert "min=L3" in hdrs["x-llm-route-decision"]
    assert hdrs["x-llm-model-served"] == "mid"
    assert float(hdrs["x-llm-cost-usd"]) >= 0.0


def test_gateway_pinned_role():
    g = gw.Gateway(ROLES_CFG, PRICING)
    resp, hdrs = g.handle({"model": "planner",
                           "messages": [{"role": "user", "content": "plan this"}]})
    assert hdrs["x-llm-model-served"] == "frontier"
    assert hdrs["x-llm-route-decision"] == "planner@frontier"   # no tier params


def test_gateway_strict_output_roundtrips():
    g = gw.Gateway(ROLES_CFG, PRICING)
    body = {"model": "verifier",
            "messages": [{"role": "user", "content": "verify"}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "Plan",
                                                "schema": _Plan.model_json_schema(),
                                                "strict": True}}}
    resp, _ = g.handle(body)
    content = resp["choices"][0]["message"]["content"]
    _Plan.model_validate_json(content)   # the returned body validates
