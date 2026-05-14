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


def test_tier_env_overrides_win_over_yaml(monkeypatch) -> None:
    """`.env`-style overrides should beat YAML defaults for every tier."""
    monkeypatch.setenv("TIER4_URL", "https://test.example.com/v1")
    monkeypatch.setenv("TIER4_MODEL", "claude-test-model-id")
    monkeypatch.setenv("TIER4_API_KEY", "test-key-value")
    m = load_models(ROOT / "config" / "tiers")
    t4 = m.by_level(4)
    assert t4.endpoint.url == "https://test.example.com/v1"
    assert t4.served_model_name == "claude-test-model-id"
    # api_key_env should be set to the env-var NAME, not the value itself.
    # Downstream readers do os.environ[api_key_env] to get the real key.
    assert t4.endpoint.api_key_env == "TIER4_API_KEY"


def test_tier_env_overrides_ignore_blank(monkeypatch) -> None:
    """Empty `TIER{N}_*` env vars should NOT override the YAML defaults.

    Otherwise shipping `TIER3_URL=` in `.env.example` would silently
    blank out tier3.yaml's url.
    """
    # Capture the YAML default before any override is applied.
    baseline = load_models(ROOT / "config" / "tiers").by_level(4)
    yaml_url = baseline.endpoint.url
    yaml_model = baseline.served_model_name

    monkeypatch.setenv("TIER4_URL", "")
    monkeypatch.setenv("TIER4_MODEL", "   ")  # whitespace-only also ignored
    monkeypatch.setenv("TIER4_API_KEY", "")
    m = load_models(ROOT / "config" / "tiers")
    t4 = m.by_level(4)
    assert t4.endpoint.url == yaml_url
    assert t4.served_model_name == yaml_model


def test_openai_https_backend_emits_provider_openai(monkeypatch, tmp_path) -> None:
    """An HTTPS non-Anthropic backend should emit `provider: openai`
    with Bearer auth headers, not the `protocol: http` localhost shape."""
    monkeypatch.setenv("TIER3_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("TIER3_MODEL", "gpt-5-mini")
    monkeypatch.setenv("TIER3_API_KEY", "sk-test")
    from benchmark.build_router_config import build
    cfg = build(
        exemplars_path=ROOT / "config" / "router-exemplars.yaml",
        backends_path=ROOT / "config" / "router-backends.yaml",
        eval_set_path=None,
    )
    t3 = next(m for m in cfg["providers"]["models"] if m["name"] == "tier3")
    assert t3["provider_model_id"] == "gpt-5-mini"
    assert t3["api_format"] == "openai"
    ref = t3["backend_refs"][0]
    assert ref["base_url"] == "https://api.openai.com/v1"
    assert ref["provider"] == "openai"
    assert ref["auth_header"] == "Authorization"
    assert ref["auth_prefix"] == "Bearer"
    assert ref["api_key_env"] == "TIER3_API_KEY"
    # Should NOT carry the localhost-style fields.
    assert "endpoint" not in ref
    assert "protocol" not in ref


def test_google_oai_compat_backend_emits_provider_openai(monkeypatch) -> None:
    """Google Gemini's OAI-compatible endpoint should flow through the
    same `provider: openai` + Bearer-auth path as OpenAI itself, since
    Google designed that endpoint to be OAI-format-equivalent."""
    monkeypatch.setenv(
        "TIER3_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    monkeypatch.setenv("TIER3_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("TIER3_API_KEY", "AIza-test")
    from benchmark.build_router_config import build
    cfg = build(
        exemplars_path=ROOT / "config" / "router-exemplars.yaml",
        backends_path=ROOT / "config" / "router-backends.yaml",
        eval_set_path=None,
    )
    t3 = next(m for m in cfg["providers"]["models"] if m["name"] == "tier3")
    assert t3["provider_model_id"] == "gemini-2.5-flash"
    assert t3["api_format"] == "openai"
    ref = t3["backend_refs"][0]
    assert ref["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert ref["provider"] == "openai"
    assert ref["auth_header"] == "Authorization"
    assert ref["auth_prefix"] == "Bearer"
    assert ref["api_key_env"] == "TIER3_API_KEY"


def test_anthropic_backend_still_takes_anthropic_path(monkeypatch) -> None:
    """Sanity: anthropic.com URLs route through the Anthropic adapter,
    not the generic openai HTTPS path. The Anthropic adapter handles the
    OAI→Anthropic shape translation."""
    monkeypatch.setenv("TIER4_URL", "https://api.anthropic.com/v1")
    monkeypatch.setenv("TIER4_MODEL", "claude-sonnet-4-5")
    monkeypatch.setenv("TIER4_API_KEY", "sk-ant-test")
    from benchmark.build_router_config import build
    cfg = build(
        exemplars_path=ROOT / "config" / "router-exemplars.yaml",
        backends_path=ROOT / "config" / "router-backends.yaml",
        eval_set_path=None,
    )
    t4 = next(m for m in cfg["providers"]["models"] if m["name"] == "tier4")
    assert t4["api_format"] == "anthropic"
    ref = t4["backend_refs"][0]
    assert ref["provider"] == "anthropic"
    # Anthropic adapter strips the /v1 suffix; vllm-sr appends it itself.
    assert ref["base_url"] == "https://api.anthropic.com"


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

    # Signals: complexity is the discrete-bucket axis; embeddings (when
    # present) feeds continuous evidence into the same weighted_sum.
    # The shipped config exercises the upstream mixed-source pattern.
    assert "signals" in routing
    assert "complexity" in routing["signals"], (
        "signals.complexity[] is the discrete-axis entry point"
    )
    assert "embeddings" in routing["signals"], (
        "signals.embeddings[] should ship the continuous frontier_synthesis "
        "signal alongside the complexity signals"
    )
    complexity_signals = routing["signals"]["complexity"]
    assert complexity_signals, "no complexity signals emitted"
    for sig in complexity_signals:
        assert "name" in sig and "threshold" in sig
        assert sig["hard"]["candidates"], f"signal {sig['name']} missing hard candidates"
        assert sig["easy"]["candidates"], f"signal {sig['name']} missing easy candidates"
    sig_names = {s["name"] for s in complexity_signals}
    assert sig_names == {
        "needs_reasoning",
        "needs_expertise",
        "needs_judgment",
        "demands_commitment",
    }

    # Projections: scores.request_difficulty + mappings.tier_band.
    assert "projections" in routing
    scores = routing["projections"]["scores"]
    assert len(scores) == 1
    rd = scores[0]
    assert rd["name"] == "request_difficulty"
    assert rd["method"] == "weighted_sum"
    # Complexity inputs. Each signal contributes a `:hard` input; `:medium`
    # inputs are conditional on `medium_weight_factor > 0`. The shipped
    # config currently has medium_weight_factor=0 (the :medium inputs
    # fire 100% of the time across all queries with no discriminative
    # power — confirmed by routing-distribution diagnostic — so they're
    # dropped from the weighted_sum).
    complexity_inputs = [i for i in rd["inputs"] if i["type"] == "complexity"]
    embedding_inputs = [i for i in rd["inputs"] if i["type"] == "embedding"]
    assert complexity_inputs, "expected at least one complexity weighted_sum input"
    complexity_names = {i["name"] for i in complexity_inputs}

    import yaml
    exemplars = yaml.safe_load(
        (ROOT / "config" / "router-exemplars.yaml").read_text()
    )
    medium_factor = float(exemplars.get("medium_weight_factor", 0.6))
    expected_hard = {f"{n}:hard" for n in sig_names}
    expected_medium = {f"{n}:medium" for n in sig_names} if medium_factor > 0 else set()
    expected_complexity_names = expected_hard | expected_medium
    assert complexity_names == expected_complexity_names, (
        f"complexity inputs mismatch: got {complexity_names}, "
        f"expected {expected_complexity_names} "
        f"(medium_weight_factor={medium_factor})"
    )

    for inp in complexity_inputs:
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

    # Embedding signals (optional, but present in this shipped config) feed
    # continuous confidence into the weighted_sum — upstream mixed-source
    # pattern. They MUST set `value_source: confidence` (without it, the
    # binary default would discard the continuous signal).
    for inp in embedding_inputs:
        assert inp.get("value_source") == "confidence", (
            f"embedding input {inp['name']!r} must set value_source: confidence "
            "(omitting it falls back to binary, defeating the continuous evidence)"
        )

    # If :medium inputs are present, each pair must respect the
    # medium_weight_factor ratio. (Skipped when factor=0 since :medium
    # inputs are omitted entirely.)
    by_name = {i["name"]: i for i in rd["inputs"]}
    if medium_factor > 0:
        for n in sig_names:
            hard_w = by_name[f"{n}:hard"]["weight"]
            med_w = by_name[f"{n}:medium"]["weight"]
            assert abs(med_w - hard_w * medium_factor) < 1e-9, (
                f"{n}: medium weight {med_w} should equal hard {hard_w} × "
                f"medium_weight_factor {medium_factor}"
            )

    # Sanity bound: max possible score should fit in band-cutoff range.
    # With :hard latent capacity that may not fire, the practical score
    # ceiling is the embedding weight sum; the :hard channel is a future
    # path. We just check the sum is in a sane range (≤ 2.0 keeps things
    # from running away).
    max_score = (
        sum(by_name[f"{n}:hard"]["weight"] for n in sig_names)
        + sum(i["weight"] for i in embedding_inputs)
    )
    assert 0.5 <= max_score <= 2.0, (
        f"max-possible request_difficulty is {max_score}; expected in [0.5, 2.0] "
        "(combined :hard channel + embedding weights)"
    )

    mappings = routing["projections"]["mappings"]
    assert len(mappings) == 1
    tb = mappings[0]
    assert tb["name"] == "tier_band"
    assert tb["source"] == "request_difficulty"
    assert tb["method"] == "threshold_bands"
    assert len(tb["outputs"]) == 5, "5 tier bands expected"

    # Decisions: one band-only per tier (5) + at least one lane decision.
    decisions = routing["decisions"]
    band_only = [d for d in decisions if len(d["rules"]["conditions"]) == 1]
    lane = [d for d in decisions if len(d["rules"]["conditions"]) > 1]
    assert len(band_only) == 5, "expected one band-only decision per tier"
    assert lane, "expected at least one lane (Boolean-qualified) decision"

    band_to_decision = {
        d["rules"]["conditions"][0]["name"]: d for d in band_only
    }
    for tier_id in ("tier1", "tier2", "tier3", "tier4", "tier5"):
        band_name = f"{tier_id}_band"
        assert band_name in band_to_decision, f"no decision conditioning on {band_name}"
        d = band_to_decision[band_name]
        assert d["modelRefs"][0]["model"] == tier_id
        assert d["rules"]["conditions"][0]["type"] == "projection"
        # vllm-sr's schema validator requires `description` on every decision.
        assert d.get("description"), f"decision for {tier_id} missing description"

    # Lane decision sanity: must include `description`, priority must beat
    # the plain band decisions (otherwise the band-only wins on a tie),
    # and the conditions must include a projection plus at least one
    # complexity qualifier — otherwise it's just a renamed band decision.
    for d in lane:
        assert d.get("description"), f"lane decision {d['name']!r} missing description"
        assert d["priority"] > band_only[0]["priority"], (
            f"lane {d['name']!r} priority {d['priority']} must beat band priority "
            f"{band_only[0]['priority']} to override the plain band decision"
        )
        types_in_lane = {c["type"] for c in d["rules"]["conditions"]}
        assert "projection" in types_in_lane, (
            f"lane {d['name']!r} must include a projection condition"
        )
        # Each lane must combine the band projection with at least one
        # non-projection qualifier — otherwise it's a renamed band decision.
        assert types_in_lane - {"projection"}, (
            f"lane {d['name']!r} has only projection conditions — needs a "
            "complexity or embedding qualifier to justify being a lane"
        )

    # The two named lanes specifically. If either changes shape, downstream
    # callers reading the generated config should know about it.
    frontier = next((d for d in lane if d["name"] == "route_tier5_frontier"), None)
    assert frontier is not None, "expected a route_tier5_frontier lane decision"
    assert frontier["modelRefs"][0]["model"] == "tier5"
    cond_names = {c["name"] for c in frontier["rules"]["conditions"]}
    assert cond_names == {
        "tier4_band", "needs_reasoning:hard", "needs_expertise:hard"
    }

    committed = next(
        (d for d in lane if d["name"] == "route_tier5_committed_judgment"), None
    )
    assert committed is not None, (
        "expected a route_tier5_committed_judgment lane decision"
    )
    assert committed["modelRefs"][0]["model"] == "tier5"
    cond_names = {c["name"] for c in committed["rules"]["conditions"]}
    assert cond_names == {
        "tier4_band", "needs_judgment:hard", "demands_commitment:hard"
    }

    # Embedding-driven frontier lane: emitted only when a
    # `frontier_synthesis` embedding signal exists in the exemplars file.
    # The shipped config has one, so this lane must be present.
    emb_frontier = next(
        (d for d in lane if d["name"] == "route_tier5_embedding_frontier"), None
    )
    assert emb_frontier is not None, (
        "expected a route_tier5_embedding_frontier lane decision (the "
        "exemplars file ships a frontier_synthesis embedding signal)"
    )
    assert emb_frontier["modelRefs"][0]["model"] == "tier5"
    types = {c["type"] for c in emb_frontier["rules"]["conditions"]}
    assert types == {"projection", "embedding"}

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
    maps to the next tier up. Catches off-by-one in lte/gt boundary handling.

    Cutoff values are read from the shipped exemplars config rather than
    hardcoded, so this test follows tuning changes automatically.
    """
    import yaml as _yaml

    from benchmark.build_router_config import build

    cfg = build(
        exemplars_path=ROOT / "config" / "router-exemplars.yaml",
        backends_path=ROOT / "config" / "router-backends.yaml",
        eval_set_path=None,
    )
    outputs = cfg["routing"]["projections"]["mappings"][0]["outputs"]
    exemplars = _yaml.safe_load(
        (ROOT / "config" / "router-exemplars.yaml").read_text()
    )
    cutoffs: list[float] = exemplars["tier_cutoffs"]
    assert len(cutoffs) == 4, "expected four cutoffs to partition into five bands"
    c1, c2, c3, c4 = cutoffs

    def band_for_score(score: float) -> str | None:
        for o in outputs:
            gt_ok = ("gt" not in o) or (score > o["gt"])
            lte_ok = ("lte" not in o) or (score <= o["lte"])
            if gt_ok and lte_ok:
                return o["name"]
        return None

    # Test both sides of every cutoff: just-on (lte side) and just-over (gt side).
    eps = 1e-6
    cases = [
        (0.0, "tier1_band"),
        (c1, "tier1_band"),         # lte-inclusive
        (c1 + eps, "tier2_band"),
        (c2, "tier2_band"),
        (c2 + eps, "tier3_band"),
        (c3, "tier3_band"),
        (c3 + eps, "tier4_band"),
        (c4, "tier4_band"),
        (c4 + eps, "tier5_band"),
        (1.0, "tier5_band"),
    ]
    for score, expected in cases:
        actual = band_for_score(score)
        assert actual == expected, f"score {score} → {actual}, expected {expected}"
