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


def test_router_exemplars_build_cleanly() -> None:
    """Verify config/router-exemplars.yaml + config/router-backends.yaml build
    without contaminating the eval set. `make load` runs this same build;
    if this fails, `make load` would too.

    Catches three failure modes that have actually bitten us:
      1. An exemplar prompt drifting into data/queries.json (would teach the
         router to memorize the prompt rather than learn a generalization).
      2. A backends entry missing for a declared tier.
      3. A rule routing to a tier that doesn't exist.
    """
    from benchmark.build_router_config import build

    cfg = build(
        exemplars_path=ROOT / "config" / "router-exemplars.yaml",
        backends_path=ROOT / "config" / "router-backends.yaml",
        eval_set_path=ROOT / "data" / "queries.json",
    )

    for key in ("version", "listeners", "providers", "signals", "decision_rules"):
        assert key in cfg, f"built router config missing top-level key {key!r}"

    # Every backend named in the generated config must have a matching
    # tier in config/tiers/ so the harness's TierLookup can translate
    # x-vsr-selected-model back to a numeric tier level.
    router_tier_names = {endpoint["name"] for endpoint in cfg["providers"]["endpoints"]}
    harness_router_aliases = {
        t.router_alias for t in load_models(ROOT / "config" / "tiers").tiers
    }
    missing = router_tier_names - harness_router_aliases
    assert not missing, (
        f"exemplar-based config declares tiers {sorted(missing)} with no "
        f"matching router_alias in config/tiers/; TierLookup will record "
        f"them as unknown_tier"
    )


def test_exemplar_routing_reaches_every_tier() -> None:
    """The exemplar-based decision_rules emitted by build_router_config must
    likewise cover all 5 tiers. The original supplied rule set made T5
    unreachable (every R/E/J band combination was caught by
    trivial/light/hard_technical/judgment_heavy, so fallthrough_to_t5 was
    dead). The improved exemplars file added a frontier_synthesis_to_t5
    rule and constrained judgment_advice_to_t4 to fix it.
    """
    from benchmark.build_router_config import build

    cfg = build(
        exemplars_path=ROOT / "config" / "router-exemplars.yaml",
        backends_path=ROOT / "config" / "router-backends.yaml",
        eval_set_path=None,
    )
    rules = cfg["decision_rules"]

    def _matches_atom(cond: dict, bands: dict[str, str]) -> bool:
        ((key, spec),) = cond.items()
        if key == "always":
            return True
        axis = key.removeprefix("complexity_")
        return bands.get(axis) in spec["in"]

    def _matches_compound(c: dict, bands: dict[str, str]) -> bool:
        if "all" in c:
            return all(_matches_atom(cc, bands) for cc in c["all"])
        if "any" in c:
            return any(_matches_atom(cc, bands) for cc in c["any"])
        return _matches_atom(c, bands)

    def _matches_when(when: dict, bands: dict[str, str]) -> bool:
        if not when or when == {"always": True}:
            return True
        if "all" in when:
            return all(
                _matches_compound(c, bands) if isinstance(c, dict) and ("all" in c or "any" in c)
                else _matches_atom(c, bands)
                for c in when["all"]
            )
        if "any" in when:
            return any(_matches_atom(c, bands) for c in when["any"])
        return _matches_atom(when, bands)

    bands = ["low", "medium", "high"]
    reached: set[str] = set()
    for r_ in bands:
        for e_ in bands:
            for j_ in bands:
                query_bands = {"reasoning": r_, "expertise": e_, "judgment": j_}
                for rule in rules:
                    if _matches_when(rule["when"], query_bands):
                        reached.add(rule["route_to"])
                        break

    expected = {"tier1", "tier2", "tier3", "tier4", "tier5"}
    assert reached == expected, f"unreachable tiers in exemplar rules: {expected - reached}"
