"""Regression tests that the YAML config files shipped under config/ actually parse.

The pydantic loaders are exercised by other tests, but those use inline fixtures.
This test catches spec-name drift between code and the shipped configs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

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


# ─────────────────────────────────────────────────────────────────────────
# Builder tests — make sure config/router-config.yaml is shaped for vllm-sr v0.3.
# ─────────────────────────────────────────────────────────────────────────

def test_router_exemplars_build_v03_shape() -> None:
    """Verify the builder emits a v0.3-shaped config:
      - routing.signals.embeddings[] (not top-level signals)
      - routing.decisions[] (not top-level decision_rules)
      - providers.models[] with backend_refs[] (not providers.endpoints[])

    Also runs the eval-overlap check that `make load` runs — if any exemplar
    prompt drifted into data/queries.json, the build aborts here.
    """
    from benchmark.build_router_config import build

    cfg = build(
        exemplars_path=ROOT / "config" / "router-exemplars.yaml",
        backends_path=ROOT / "config" / "router-backends.yaml",
        eval_set_path=ROOT / "data" / "queries.json",
    )

    # Top level.
    for key in ("version", "listeners", "providers", "routing"):
        assert key in cfg, f"built router config missing top-level key {key!r}"
    assert cfg["version"] == "v0.3"

    # No leftover invented keys.
    for stale in ("signals", "decision_rules", "signals_builtin"):
        assert stale not in cfg, (
            f"top-level {stale!r} is the OLD invented schema; v0.3 wants this "
            f"nested under routing:"
        )

    # Providers: must use models[].backend_refs[], not endpoints[].
    assert "endpoints" not in cfg["providers"], (
        "providers.endpoints[] was the OLD shape; v0.3 wants providers.models[]"
    )
    assert "models" in cfg["providers"]
    for m in cfg["providers"]["models"]:
        assert "name" in m and "provider_model_id" in m and "api_format" in m
        assert m["backend_refs"], f"model {m['name']} has no backend_refs"
        ref = m["backend_refs"][0]
        # OAI-compatible refs use endpoint+protocol; anthropic refs use base_url+provider.
        assert "endpoint" in ref or "base_url" in ref

    # Routing block.
    routing = cfg["routing"]
    assert "modelCards" in routing
    assert routing["modelCards"], "modelCards is empty"
    assert "signals" in routing
    assert "embeddings" in routing["signals"]
    assert routing["signals"]["embeddings"], "no embedding signals emitted"
    assert "decisions" in routing
    assert routing["decisions"], "no decisions emitted"

    # Embedding signals: two per axis (hard + easy), with threshold + candidates.
    sig_names = {s["name"] for s in routing["signals"]["embeddings"]}
    for axis in ("reasoning", "expertise", "judgment"):
        assert f"{axis}_hard" in sig_names, f"missing embedding signal {axis}_hard"
        assert f"{axis}_easy" in sig_names, f"missing embedding signal {axis}_easy"
    for sig in routing["signals"]["embeddings"]:
        assert "threshold" in sig and isinstance(sig["threshold"], int | float)
        assert sig["candidates"], f"signal {sig['name']} has no candidates"
        assert sig.get("aggregation_method") == "max"

    # Tier alignment: every model name in providers.models must have a matching
    # router_alias in config/tiers/. Drift here means TierLookup records the
    # router's pick as unknown_tier.
    router_tier_names = {m["name"] for m in cfg["providers"]["models"]}
    harness_router_aliases = {
        t.router_alias for t in load_models(ROOT / "config" / "tiers").tiers
    }
    missing = router_tier_names - harness_router_aliases
    assert not missing, (
        f"router-config.yaml declares tiers {sorted(missing)} with no matching "
        f"router_alias in config/tiers/; TierLookup will record them as unknown_tier"
    )


def test_exemplar_routing_reaches_every_tier() -> None:
    """Synthetically evaluate each band combination against the v0.3 decisions
    and confirm every tier is reachable.

    Modeling: for a query in band X on axis A, the `<A>_hard` signal "matches"
    iff X == high, and `<A>_easy` signal "matches" iff X == low. Then we walk
    the decision conditions (with AND/OR/NOT) and pick the first decision
    whose rules evaluate true.

    This is the same test as before, just rewritten for v0.3 conditions.
    """
    from benchmark.build_router_config import build

    cfg = build(
        exemplars_path=ROOT / "config" / "router-exemplars.yaml",
        backends_path=ROOT / "config" / "router-backends.yaml",
        eval_set_path=None,
    )
    decisions = sorted(cfg["routing"]["decisions"], key=lambda d: -d["priority"])

    def eval_cond(c: dict, bands: dict[str, str]) -> bool:
        """Evaluate one v0.3 condition or condition tree against synthetic bands."""
        # Atom: {type: embedding, name: "<axis>_<side>"}
        if "type" in c and "name" in c and "operator" not in c:
            name: str = c["name"]
            axis, _, side = name.rpartition("_")
            assert side in ("hard", "easy"), f"unrecognized signal name {name!r}"
            band = bands.get(axis)
            return (side == "hard" and band == "high") or (side == "easy" and band == "low")
        # Composite: {operator: AND|OR|NOT, conditions: [...]}
        op = c["operator"]
        children = c["conditions"]
        if op == "AND":
            return all(eval_cond(ch, bands) for ch in children)
        if op == "OR":
            return any(eval_cond(ch, bands) for ch in children)
        if op == "NOT":
            assert len(children) == 1
            return not eval_cond(children[0], bands)
        raise AssertionError(f"unknown operator {op!r}")

    def eval_rules(rules: dict, bands: dict[str, str]) -> bool:
        # rules itself looks like a composite (operator + conditions).
        if not rules["conditions"]:
            return True  # unconditional fallthrough
        return eval_cond(rules, bands)

    bands_values = ["low", "medium", "high"]
    reached: set[str] = set()
    for r_ in bands_values:
        for e_ in bands_values:
            for j_ in bands_values:
                query_bands = {"reasoning": r_, "expertise": e_, "judgment": j_}
                for dec in decisions:
                    if eval_rules(dec["rules"], query_bands):
                        reached.add(dec["modelRefs"][0]["model"])
                        break

    expected = {"tier1", "tier2", "tier3", "tier4", "tier5"}
    assert reached == expected, f"unreachable tiers: {expected - reached}"


# ─────────────────────────────────────────────────────────────────────────
# Unit tests for the band→condition translation table.
# These are the load-bearing piece of the builder — verify in isolation.
# ─────────────────────────────────────────────────────────────────────────

def test_axis_condition_table() -> None:
    """Smoke-check each row of the band→condition table by name/structure."""
    from benchmark.build_router_config import _axis_condition

    # all three bands → no constraint
    assert _axis_condition("reasoning", ["low", "medium", "high"]) is None

    # [high] → hard
    c = _axis_condition("reasoning", ["high"])
    assert c == {"type": "embedding", "name": "reasoning_hard"}

    # [low] → easy AND NOT hard
    c = _axis_condition("reasoning", ["low"])
    assert c["operator"] == "AND"
    assert {"type": "embedding", "name": "reasoning_easy"} in c["conditions"]
    has_not_hard = any(
        cond.get("operator") == "NOT"
        and cond["conditions"] == [{"type": "embedding", "name": "reasoning_hard"}]
        for cond in c["conditions"]
    )
    assert has_not_hard

    # [low, medium] → NOT hard
    c = _axis_condition("reasoning", ["low", "medium"])
    assert c == {
        "operator": "NOT",
        "conditions": [{"type": "embedding", "name": "reasoning_hard"}],
    }

    # [medium, high] → NOT easy
    c = _axis_condition("reasoning", ["medium", "high"])
    assert c == {
        "operator": "NOT",
        "conditions": [{"type": "embedding", "name": "reasoning_easy"}],
    }

    # [medium] → NOT hard AND NOT easy
    c = _axis_condition("reasoning", ["medium"])
    assert c["operator"] == "AND"
    assert len(c["conditions"]) == 2
    assert all(cc["operator"] == "NOT" for cc in c["conditions"])


def _walk(node: Any) -> list[Any]:
    """Flatten a condition tree to a list of every dict node (for searching)."""
    out: list[Any] = [node]
    if isinstance(node, dict) and "conditions" in node:
        for ch in node["conditions"]:
            out.extend(_walk(ch))
    return out


def test_emit_decision_preserves_route_target() -> None:
    """Every rule's `route_to` ends up as the model in `modelRefs`."""
    from benchmark.build_router_config import _emit_decision

    rule = {
        "name": "test",
        "route_to": "tier3",
        "when": {"reasoning": ["high"]},
        "description": "test rule",
    }
    d = _emit_decision(rule, priority=42)
    assert d["name"] == "test"
    assert d["priority"] == 42
    assert d["modelRefs"] == [{"model": "tier3", "use_reasoning": False}]
    assert d["description"] == "test rule"
    # rules.conditions should contain at least one reference to reasoning_hard.
    flat = _walk(d["rules"])
    assert any(n.get("name") == "reasoning_hard" for n in flat if isinstance(n, dict))


def test_emit_decision_empty_when_is_fallthrough() -> None:
    """An empty `when` yields an empty conditions list (unconditional match)."""
    from benchmark.build_router_config import _emit_decision

    rule = {"name": "fallthrough", "route_to": "tier5", "when": {}}
    d = _emit_decision(rule, priority=0)
    assert d["rules"]["operator"] == "AND"
    assert d["rules"]["conditions"] == []


def test_builder_mock_endpoint_overrides_every_backend() -> None:
    """With --mock-endpoint set, every providers.models[].backend_refs[]
    points at the mock, and api_format is forced to openai (no Anthropic
    translation, since the mock speaks plain OAI)."""
    from benchmark.build_router_config import build

    mock = "host.docker.internal:8811/v1"
    cfg = build(
        exemplars_path=ROOT / "config" / "router-exemplars.yaml",
        backends_path=ROOT / "config" / "router-backends.yaml",
        eval_set_path=None,
        mock_endpoint=mock,
    )
    models = cfg["providers"]["models"]
    assert models, "no provider models emitted"
    for m in models:
        assert m["api_format"] == "openai", f"{m['name']}: api_format should be openai in mock mode"
        refs = m["backend_refs"]
        assert len(refs) == 1
        ref = refs[0]
        assert ref["endpoint"] == mock, f"{m['name']}: endpoint not redirected to mock"
        # No api_key — the mock doesn't enforce auth.
        assert "api_key_env" not in ref
        # provider_model_id should be the tier name, not e.g. claude-opus-4-7.
        assert m["provider_model_id"] == m["name"]


def test_emit_decision_requires_any_high() -> None:
    """`requires_any_high: [a, b]` adds an OR over the *_hard signals."""
    from benchmark.build_router_config import _emit_decision

    rule = {
        "name": "any_high",
        "route_to": "tier3",
        "when": {"judgment": ["low", "medium"]},
        "requires_any_high": ["reasoning", "expertise"],
    }
    d = _emit_decision(rule, priority=50)
    # Find the OR block.
    flat = _walk(d["rules"])
    or_nodes = [n for n in flat if isinstance(n, dict) and n.get("operator") == "OR"]
    assert or_nodes, "no OR block emitted for requires_any_high"
    or_block = or_nodes[0]
    or_names = {c["name"] for c in or_block["conditions"]}
    assert or_names == {"reasoning_hard", "expertise_hard"}
