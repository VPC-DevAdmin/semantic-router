"""Regression tests that the YAML config files shipped under config/ actually parse.

The pydantic loaders are exercised by other tests, but those use inline fixtures.
This test catches spec-name drift between code and the example configs.
"""
from __future__ import annotations

from pathlib import Path

from benchmark.config import (
    load_models,
    load_queries,
    load_router_process,
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


def test_router_yaml_parses() -> None:
    r = load_router_process(ROOT / "config" / "router.yaml")
    assert r.binary
    assert 1 <= r.apiserver_port <= 65535
    assert 1 <= r.frontend_port <= 65535


def test_queries_json_parses() -> None:
    q = load_queries(ROOT / "data" / "queries.json")
    assert len(q.queries) >= 100
    assert all(qq.expected_answer for qq in q.queries), "every shipped query should have gold"


def test_vllm_sr_config_parses_as_yaml() -> None:
    """Smoke check that config/vllm-sr.yaml is valid YAML with the expected
    top-level structure. The router itself will validate its schema; here we
    just guard against typos and ensure the model names line up with
    models.yaml model_ids — that alignment is what makes TierLookup work."""
    import yaml

    path = ROOT / "config" / "vllm-sr.yaml"
    with path.open() as f:
        cfg = yaml.safe_load(f)

    # Top-level structure the router expects.
    for key in ("version", "listeners", "providers", "routing"):
        assert key in cfg, f"vllm-sr.yaml missing top-level key {key!r}"

    router_model_names = {m["name"] for m in cfg["providers"]["models"]}
    harness_model_ids = {t.model_id for t in load_models(ROOT / "config" / "models.yaml").tiers}

    # Every router-declared model must have a corresponding entry in models.yaml
    # so TierLookup can translate the `x-vsr-selected-model` header back to a
    # numeric tier. Drift here is the most common config bug.
    missing = router_model_names - harness_model_ids
    assert not missing, (
        f"router declares models {sorted(missing)} that have no entry in "
        f"config/models.yaml; add them or rename so TierLookup works"
    )
