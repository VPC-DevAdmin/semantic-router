"""Regression tests that the YAML config files shipped under config/ actually parse.

The pydantic loaders are exercised by other tests, but those use inline fixtures.
This test catches spec-name drift between code and the example configs.
"""
from __future__ import annotations

from pathlib import Path

from benchmark.config import (
    load_endpoint,
    load_models,
    load_queries,
    load_router_process,
    load_scoring,
)

ROOT = Path(__file__).parent.parent


def test_models_yaml_parses() -> None:
    m = load_models(ROOT / "config" / "models.yaml")
    assert len(m.tiers) >= 5
    levels = sorted({t.level for t in m.tiers})
    assert levels == sorted(set(levels))
    # Every tier's specializations must be in the whitelist; if this fails,
    # update either the whitelist or models.yaml — they must match.
    for t in m.tiers:
        assert t.specializations, f"tier {t.name} has empty specializations"


def test_judge_yaml_parses() -> None:
    cfg = load_endpoint(ROOT / "config" / "judge.yaml")
    assert cfg.endpoint
    assert cfg.model_id


def test_scoring_yaml_parses() -> None:
    sc = load_scoring(ROOT / "config" / "scoring.yaml")
    assert sc.rubric_version
    assert len(sc.scale) >= 3


def test_router_yaml_parses() -> None:
    r = load_router_process(ROOT / "config" / "router.yaml")
    assert r.binary
    assert 1 <= r.apiserver_port <= 65535
    assert 1 <= r.frontend_port <= 65535


def test_queries_json_parses() -> None:
    q = load_queries(ROOT / "data" / "queries.json")
    assert len(q.queries) >= 100
    assert all(qq.expected_answer for qq in q.queries), "every shipped query should have gold"
