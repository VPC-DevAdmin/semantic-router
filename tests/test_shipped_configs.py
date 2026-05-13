"""Regression tests that the YAML config files shipped under config/ actually parse.

The pydantic loaders are exercised by other tests, but those use inline fixtures.
This test catches spec-name drift between code and the shipped configs.
"""
from __future__ import annotations

from pathlib import Path

from benchmark.config import (
    load_models,
    load_queries,
    load_router_process,
)

ROOT = Path(__file__).parent.parent


def test_tier_yamls_parse() -> None:
    m = load_models(ROOT / "config" / "tiers")
    assert len(m.tiers) >= 5
    levels = sorted(t.level for t in m.tiers)
    assert levels == sorted(set(levels)), "duplicate tier levels"
    for t in m.tiers:
        assert t.specializations, f"tier {t.name} has empty specializations"
        assert t.router_alias, f"tier {t.name} missing router_alias"
        assert t.served_model_name, f"tier {t.name} missing served_model_name"
        assert t.endpoint.url, f"tier {t.name} missing endpoint.url"
        assert t.backend.kind, f"tier {t.name} missing backend.kind"


def test_router_yaml_parses() -> None:
    r = load_router_process(ROOT / "config" / "router.yaml")
    assert r.binary
    assert 1 <= r.apiserver_port <= 65535
    assert 1 <= r.frontend_port <= 65535


def test_queries_json_parses() -> None:
    q = load_queries(ROOT / "data" / "queries.json")
    assert len(q.queries) >= 100
    assert all(qq.expected_answer for qq in q.queries), "every shipped query should have gold"


def test_vllm_sr_routing_template_parses() -> None:
    """The hand-maintained routing template (listeners/signals/decisions/global)
    must be valid YAML with the expected top-level structure. Provider models
    are generated from config/tiers/*.yaml at `make gen-router-config` time;
    we don't assert on them here."""
    import yaml

    path = ROOT / "config" / "vllm-sr.routing.yaml"
    if not path.exists():
        # Generator not yet wired; skip this check. The generator task will
        # create this file.
        return
    with path.open() as f:
        cfg = yaml.safe_load(f)
    for key in ("version", "listeners", "routing", "global"):
        assert key in cfg, f"routing template missing top-level key {key!r}"
