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


# ─────────────────────────────────────────────────────────────────────────
# Builder tests — verify the projections-shape config:
#   routing.signals.complexity[]
#   routing.projections.scores.request_difficulty (weighted_sum)
#   routing.projections.mappings.tier_band (threshold_bands, 5 outputs)
#   routing.decisions[] — one per tier, each conditioning on its band
# ─────────────────────────────────────────────────────────────────────────

def test_router_exemplars_build_projections_shape() -> None:
    """Verify the builder emits the canonical v0.3 projections-based shape.

    Also runs the eval-overlap check that `make load` runs — if any
    candidate prompt drifted into data/queries.json, the build aborts here.
    """
    from benchmark.build_router_config import build

    cfg = build(
        exemplars_path=ROOT / "config" / "router-exemplars.yaml",
        backends_path=ROOT / "config" / "router-backends.yaml",
        eval_set_path=ROOT / "data" / "queries.json",
    )

    # Top level.
    for key in ("version", "listeners", "providers", "routing", "global"):
        assert key in cfg, f"built router config missing top-level key {key!r}"
    assert cfg["version"] == "v0.3"

    # Semantic cache must be explicitly disabled.
    assert cfg["global"]["stores"]["semantic_cache"]["enabled"] is False
    # Complexity prototype scoring needs to be on for the new signals to work.
    assert (
        cfg["global"]["model_catalog"]["modules"]["complexity"]["prototype_scoring"]["enabled"]
        is True
    )

    # No leftover invented keys from the OLD design.
    for stale in ("signals", "decision_rules", "signals_builtin", "projections"):
        assert stale not in cfg, (
            f"top-level {stale!r} is the OLD shape; v0.3 wants this nested under routing:"
        )

    # Providers: still uses models[].backend_refs[] — unchanged.
    assert "models" in cfg["providers"]
    for m in cfg["providers"]["models"]:
        assert "name" in m and "provider_model_id" in m and "api_format" in m
        assert m["backend_refs"], f"model {m['name']} has no backend_refs"

    routing = cfg["routing"]
    assert "modelCards" in routing and routing["modelCards"]

    # Signals: complexity (not embeddings).
    assert "signals" in routing
    assert "complexity" in routing["signals"], (
        "signals.complexity[] is the projections-design entry point"
    )
    assert "embeddings" not in routing["signals"], (
        "embeddings[] was the OLD DIY hard/easy design; should be replaced by complexity[]"
    )
    complexity_signals = routing["signals"]["complexity"]
    assert complexity_signals, "no complexity signals emitted"
    for sig in complexity_signals:
        assert "name" in sig and "threshold" in sig
        assert sig["hard"]["candidates"], f"signal {sig['name']} missing hard candidates"
        assert sig["easy"]["candidates"], f"signal {sig['name']} missing easy candidates"
    sig_names = {s["name"] for s in complexity_signals}
    assert sig_names == {"needs_reasoning", "needs_expertise", "needs_judgment"}

    # Projections: scores.request_difficulty + mappings.tier_band.
    assert "projections" in routing
    scores = routing["projections"]["scores"]
    assert len(scores) == 1
    rd = scores[0]
    assert rd["name"] == "request_difficulty"
    assert rd["method"] == "weighted_sum"
    # Each signal contributes TWO inputs: `<id>:medium` (half weight) and
    # `<id>:hard` (full weight). Without :medium, queries that match only
    # at the medium level contribute 0 — which collapsed request_difficulty
    # to 0 for most of the eval set in the first projections roll-out.
    # Per upstream canonical config, complexity-input names take the form
    # `<signal_id>:hard` or `<signal_id>:medium`; bare `<signal_id>` does
    # not bind.
    input_names = {i["name"] for i in rd["inputs"]}
    expected_names = (
        {f"{n}:hard" for n in sig_names} | {f"{n}:medium" for n in sig_names}
    )
    assert input_names == expected_names
    for inp in rd["inputs"]:
        assert inp["type"] == "complexity"
        assert inp["name"].endswith((":hard", ":medium")), (
            f"complexity input {inp['name']!r} missing ':hard'/':medium' suffix"
        )
        # IMPORTANT: do NOT set `value_source` on complexity inputs. Omitting
        # it gets the binary default (match=1.0 / miss=0.0). Setting it to
        # `confidence` returns the contrastive margin (~0.0-0.05), which
        # collapses the projected score to ~0 and lands everything in
        # tier1_band.
        assert "value_source" not in inp, (
            f"complexity input {inp['name']!r} should not set value_source — "
            "the binary default is what we want"
        )
        assert isinstance(inp["weight"], int | float)
    # Per signal: :medium weight should be exactly half the :hard weight.
    by_name = {i["name"]: i for i in rd["inputs"]}
    for n in sig_names:
        hard_w = by_name[f"{n}:hard"]["weight"]
        med_w = by_name[f"{n}:medium"]["weight"]
        assert abs(med_w - hard_w * 0.5) < 1e-9, (
            f"{n}: medium weight {med_w} should be half hard weight {hard_w}"
        )
    # The :hard weights alone should sum to ~1.0 — caps request_difficulty
    # at 1.0 (medium/hard are mutually exclusive per signal per query).
    hard_weight_sum = sum(by_name[f"{n}:hard"]["weight"] for n in sig_names)
    assert abs(hard_weight_sum - 1.0) < 1e-6, (
        f":hard weights sum to {hard_weight_sum} — should be ~1.0"
    )

    mappings = routing["projections"]["mappings"]
    assert len(mappings) == 1
    tb = mappings[0]
    assert tb["name"] == "tier_band"
    assert tb["source"] == "request_difficulty"
    assert tb["method"] == "threshold_bands"
    assert len(tb["outputs"]) == 5, "5 tier bands expected"

    # Decisions: one per tier, each conditioning on its band.
    decisions = routing["decisions"]
    assert len(decisions) == 5
    band_to_decision = {
        d["rules"]["conditions"][0]["name"]: d for d in decisions
    }
    for tier_id in ("tier1", "tier2", "tier3", "tier4", "tier5"):
        band_name = f"{tier_id}_band"
        assert band_name in band_to_decision, f"no decision conditioning on {band_name}"
        d = band_to_decision[band_name]
        assert d["modelRefs"][0]["model"] == tier_id
        assert d["rules"]["conditions"][0]["type"] == "projection"
        # vllm-sr's schema validator requires `description` on every decision.
        assert d.get("description"), f"decision for {tier_id} missing description"

    # Tier alignment with config/tiers/.
    router_tier_names = {m["name"] for m in cfg["providers"]["models"]}
    harness_router_aliases = {
        t.router_alias for t in load_models(ROOT / "config" / "tiers").tiers
    }
    missing = router_tier_names - harness_router_aliases
    assert not missing, (
        f"router-config.yaml declares tiers {sorted(missing)} with no matching "
        f"router_alias in config/tiers/; TierLookup will record them as unknown_tier"
    )


def test_tier_bands_cover_unit_interval_without_gaps_or_overlap() -> None:
    """The five tier bands together must cover [0, 1] with no gaps or overlap."""
    from benchmark.build_router_config import build

    cfg = build(
        exemplars_path=ROOT / "config" / "router-exemplars.yaml",
        backends_path=ROOT / "config" / "router-backends.yaml",
        eval_set_path=None,
    )
    outputs = cfg["routing"]["projections"]["mappings"][0]["outputs"]
    assert len(outputs) == 5

    # First band: no lower bound (must have only lte).
    assert "gt" not in outputs[0]
    assert "lte" in outputs[0]
    # Last band: no upper bound.
    assert "gt" in outputs[-1]
    assert "lte" not in outputs[-1]
    # Adjacent bands meet: outputs[i].lte == outputs[i+1].gt.
    for i in range(len(outputs) - 1):
        assert outputs[i]["lte"] == outputs[i + 1]["gt"], (
            f"gap or overlap between {outputs[i]['name']} and {outputs[i + 1]['name']}"
        )


def test_score_at_each_cutoff_lands_in_expected_tier() -> None:
    """Sanity check: a request_difficulty score just above each cutoff
    maps to the next tier up. Catches off-by-one in lte/gt boundary handling."""
    from benchmark.build_router_config import build

    cfg = build(
        exemplars_path=ROOT / "config" / "router-exemplars.yaml",
        backends_path=ROOT / "config" / "router-backends.yaml",
        eval_set_path=None,
    )
    outputs = cfg["routing"]["projections"]["mappings"][0]["outputs"]

    def band_for_score(score: float) -> str | None:
        for o in outputs:
            gt_ok = ("gt" not in o) or (score > o["gt"])
            lte_ok = ("lte" not in o) or (score <= o["lte"])
            if gt_ok and lte_ok:
                return o["name"]
        return None

    # Sample scores. Cutoffs = [0.20, 0.40, 0.60, 0.80].
    cases = [
        (0.00, "tier1_band"),
        (0.20, "tier1_band"),   # cutoff inclusive on the lte side
        (0.25, "tier2_band"),
        (0.50, "tier3_band"),
        (0.70, "tier4_band"),
        (0.95, "tier5_band"),
        (1.00, "tier5_band"),
    ]
    for score, expected in cases:
        actual = band_for_score(score)
        assert actual == expected, f"score {score} → {actual}, expected {expected}"
