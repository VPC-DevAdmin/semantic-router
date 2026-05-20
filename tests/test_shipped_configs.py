"""Regression tests that the YAML config files shipped under config/ actually parse.

The pydantic loaders are exercised by other tests, but those use inline fixtures.
This test catches spec-name drift between code and the shipped configs.
"""
from __future__ import annotations

import os
from pathlib import Path

from benchmark.config import (
    load_models,
    load_queries,
    load_router_process,
)

ROOT = Path(__file__).parent.parent


def test_tier_env_overrides_win_over_yaml(monkeypatch) -> None:
    """`.env`-style indexed slots produce per-tier callable models."""
    monkeypatch.setenv("TIER4_1_URL", "https://test.example.com/v1")
    monkeypatch.setenv("TIER4_1_MODEL", "claude-test-model-id")
    monkeypatch.setenv("TIER4_1_API_KEY", "test-key-value")
    monkeypatch.setenv("TIER4_1_PROVIDER", "TestProvider")
    m = load_models(ROOT / "config" / "tiers")
    t4 = m.by_level(4)
    s1 = t4.models[0]
    assert (s1.slot, s1.url, s1.served_model_name) == (
        1, "https://test.example.com/v1", "claude-test-model-id",
    )
    # api_key_env is the env-var NAME, not the value.
    assert s1.api_key_env == "TIER4_1_API_KEY"
    assert s1.provider == "TestProvider"


def test_tier_env_overrides_ignore_blank(monkeypatch) -> None:
    """Empty `TIER{N}_{i}_*` env vars should NOT discover a slot — the
    tier ends up with no callable models (so a blank line in .env.example
    can't silently misconfigure a tier)."""
    monkeypatch.setenv("TIER4_1_URL", "")
    monkeypatch.setenv("TIER4_1_MODEL", "   ")  # whitespace-only also ignored
    monkeypatch.setenv("TIER4_1_API_KEY", "")
    m = load_models(ROOT / "config" / "tiers")
    t4 = m.by_level(4)
    # No env slots → no callable models (tier YAML carries no fallback).
    assert t4.models == []
    assert t4.resolved_models() == []


def test_localhost_backend_url_translated_to_host_docker_internal(monkeypatch) -> None:
    """`.env`'s TIER{N}_1_URL is host-side ('localhost'). The router
    runs in docker, where localhost is the container's loopback —
    Envoy 503s every request because the configured upstream isn't
    reachable. The build must rewrite localhost → host.docker.internal
    in router-backends.yaml so the router container can reach the host."""
    monkeypatch.setenv("TIER1_1_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("TIER1_1_MODEL", "Qwen3-1.7B")
    monkeypatch.setenv("TIER2_1_URL", "http://127.0.0.1:8002/v1")
    monkeypatch.setenv("TIER2_1_MODEL", "Qwen3-30B-A3B")
    monkeypatch.setenv("TIER3_1_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("TIER3_1_MODEL", "gpt-5-mini")
    monkeypatch.setenv("TIER3_1_API_KEY", "sk-test")

    from benchmark.build_router_config import build
    cfg = build(
        exemplars_path=ROOT / "config" / "router-exemplars.yaml",
        backends_path=ROOT / "config" / "router-backends.yaml",
        eval_set_path=None,
    )

    def ref_for(tier_name):
        m = next(m for m in cfg["providers"]["models"] if m["name"] == tier_name)
        ref = m["backend_refs"][0]
        # Local-HTTP shape uses `endpoint:` (host:port/path); HTTPS/vendor
        # shapes use `base_url:`. Concatenate both for the substring check.
        return ref.get("endpoint", "") + " " + ref.get("base_url", "")

    # Both localhost variants translated.
    assert "host.docker.internal:8001" in ref_for("tier1")
    assert "host.docker.internal:8002" in ref_for("tier2")
    # Vendor URL untouched.
    assert "api.openai.com" in ref_for("tier3")
    assert "host.docker.internal" not in ref_for("tier3")


def test_external_model_ids_emitted_for_every_tier(monkeypatch) -> None:
    """`external_model_ids: {vllm: <real model>}` is what makes vllm-sr
    rewrite the outgoing `model:` field in the request body. Without it,
    the router forwards the router-side alias (`tier3`) verbatim and
    OpenAI/Anthropic 404 with "model `tier3` does not exist".

    Source: src/semantic-router/pkg/config/helper.go:ResolveExternalModelID
    — looks up modelConfig.ExternalModelIDs[endpointType]; endpointType
    defaults to "vllm" when the backend_ref omits `type:`.

    Regression for a 417/417 error pass where every tier3+ query 404'd.
    """
    monkeypatch.setenv("TIER1_1_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("TIER1_1_MODEL", "Qwen3-1.7B")
    monkeypatch.setenv("TIER3_1_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("TIER3_1_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("TIER3_1_API_KEY", "sk-test")
    monkeypatch.setenv("TIER4_1_URL", "https://api.anthropic.com/v1")
    monkeypatch.setenv("TIER4_1_MODEL", "claude-sonnet-4-5")
    monkeypatch.setenv("TIER4_1_API_KEY", "sk-ant-test")

    from benchmark.build_router_config import build
    cfg = build(
        exemplars_path=ROOT / "config" / "router-exemplars.yaml",
        backends_path=ROOT / "config" / "router-backends.yaml",
        eval_set_path=None,
    )
    by_name = {m["name"]: m for m in cfg["providers"]["models"]}

    # Local HTTP backend: rewrite tier1 → Qwen3-1.7B.
    assert by_name["tier1"]["external_model_ids"] == {"vllm": "Qwen3-1.7B"}
    # HTTPS OpenAI: rewrite tier3 → gpt-5.4-mini.
    assert by_name["tier3"]["external_model_ids"] == {"vllm": "gpt-5.4-mini"}
    # Anthropic: rewrite tier4 → claude-sonnet-4-5.
    assert by_name["tier4"]["external_model_ids"] == {"vllm": "claude-sonnet-4-5"}


def test_build_module_loads_dotenv(tmp_path) -> None:
    """`python -m benchmark.build_router_config` (how the Makefile invokes
    the build) must load .env, otherwise `_apply_backend_env_overrides`
    sees no TIER{N}_1_MODEL etc. and the YAML placeholder `model: tier1`
    flows verbatim into router-config.yaml. The router then forwards
    `model: "tier1"` to OpenAI/Anthropic and gets a 404.

    Regression for a 417/797 failure pass where every tier3+ query 404'd
    with "The model `tier3` does not exist."
    """
    import subprocess
    import sys

    # Probe: a .env in cwd should be picked up at module import time. The
    # subprocess prints the resolved var; if dotenv didn't run, the var is
    # empty.
    (tmp_path / ".env").write_text("BUILD_DOTENV_PROBE=loaded\n")
    code = (
        "import benchmark.build_router_config; "
        "import os; "
        "print(os.environ.get('BUILD_DOTENV_PROBE', ''))"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("BUILD_DOTENV_PROBE")}
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == "loaded", (
        "build_router_config must call load_dotenv at import time so the "
        "TIER{N}_1_* env overrides apply when invoked via "
        f"`python -m benchmark.build_router_config` (got {res.stdout!r}, stderr={res.stderr!r})"
    )


def test_api_key_inlined_at_build_time(monkeypatch) -> None:
    """The API key value (not just the env-var name) must be inlined into
    the compiled router-config.yaml. `vllm-sr serve`'s compose template
    does NOT propagate `TIER{N}_{i}_API_KEY` env vars into the router
    container, so an `api_key_env: TIER3_1_API_KEY` ref resolves to an
    empty value there and Envoy 401s. Resolve-and-inline at build time
    fixes that — the router reads the literal key from its config."""
    monkeypatch.setenv("TIER3_1_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("TIER3_1_MODEL", "gpt-5-mini")
    monkeypatch.setenv("TIER3_1_API_KEY", "sk-build-time-resolved")

    from benchmark.build_router_config import build
    cfg = build(
        exemplars_path=ROOT / "config" / "router-exemplars.yaml",
        backends_path=ROOT / "config" / "router-backends.yaml",
        eval_set_path=None,
    )
    t3 = next(m for m in cfg["providers"]["models"] if m["name"] == "tier3")
    ref = t3["backend_refs"][0]
    # The literal value, not the env-var name, is what the router reads.
    assert ref["api_key"] == "sk-build-time-resolved"
    assert "api_key_env" not in ref


def test_api_key_falls_back_to_env_name_when_unset(monkeypatch) -> None:
    """If the env var named by `api_key_env` is unset/empty at build
    time, keep `api_key_env: <NAME>` in the ref. The router will 401
    with a clear pointer at the missing variable instead of silently
    using an empty key."""
    monkeypatch.delenv("TIER3_1_API_KEY", raising=False)

    from benchmark.build_router_config import _apply_api_key
    ref: dict = {}
    _apply_api_key(ref, {"api_key_env": "TIER3_1_API_KEY"})
    assert ref == {"api_key_env": "TIER3_1_API_KEY"}
    assert "api_key" not in ref

    # No api_key_env at all (local HTTP) is a no-op.
    ref2: dict = {}
    _apply_api_key(ref2, {})
    assert ref2 == {}


def test_translate_host_unit() -> None:
    """Direct test of the URL rewrite — covers edge cases without
    going through the full router-config build."""
    from benchmark.build_router_config import _translate_host_for_router_container as t
    # localhost/127.0.0.1 → host.docker.internal, preserving port & path.
    assert t("http://localhost:8001/v1") == "http://host.docker.internal:8001/v1"
    assert t("http://127.0.0.1:8002/v1") == "http://host.docker.internal:8002/v1"
    # Case-insensitive host match.
    assert t("http://LOCALHOST:8001/v1") == "http://host.docker.internal:8001/v1"
    # Non-loopback hosts pass through unchanged.
    assert t("https://api.openai.com/v1") == "https://api.openai.com/v1"
    assert t("http://192.168.1.10:8001/v1") == "http://192.168.1.10:8001/v1"
    # No port → still rewritten.
    assert t("http://localhost/v1") == "http://host.docker.internal/v1"


def test_openai_https_backend_emits_provider_openai(monkeypatch, tmp_path) -> None:
    """An HTTPS non-Anthropic backend should emit `provider: openai`
    with Bearer auth headers, not the `protocol: http` localhost shape."""
    monkeypatch.setenv("TIER3_1_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("TIER3_1_MODEL", "gpt-5-mini")
    monkeypatch.setenv("TIER3_1_API_KEY", "sk-test")
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
    # Key is resolved at build time and inlined as `api_key`,
    # so the router service reads it from the config (its docker
    # container doesn't have TIER{N}_{i}_API_KEY env vars).
    assert ref["api_key"] == "sk-test" or ref["api_key"] == "AIza-test"
    assert "api_key_env" not in ref
    # Should NOT carry the localhost-style fields.
    assert "endpoint" not in ref
    assert "protocol" not in ref


def test_google_oai_compat_backend_emits_provider_openai(monkeypatch) -> None:
    """Google Gemini's OAI-compatible endpoint should flow through the
    same `provider: openai` + Bearer-auth path as OpenAI itself, since
    Google designed that endpoint to be OAI-format-equivalent."""
    monkeypatch.setenv(
        "TIER3_1_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    monkeypatch.setenv("TIER3_1_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("TIER3_1_API_KEY", "AIza-test")
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
    # Key is resolved at build time and inlined as `api_key`,
    # so the router service reads it from the config (its docker
    # container doesn't have TIER{N}_{i}_API_KEY env vars).
    assert ref["api_key"] == "sk-test" or ref["api_key"] == "AIza-test"
    assert "api_key_env" not in ref


def test_anthropic_backend_still_takes_anthropic_path(monkeypatch) -> None:
    """Sanity: anthropic.com URLs route through the Anthropic adapter,
    not the generic openai HTTPS path. The Anthropic adapter handles the
    OAI→Anthropic shape translation."""
    monkeypatch.setenv("TIER4_1_URL", "https://api.anthropic.com/v1")
    monkeypatch.setenv("TIER4_1_MODEL", "claude-sonnet-4-5")
    monkeypatch.setenv("TIER4_1_API_KEY", "sk-ant-test")
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
        assert t.timeout_s > 0, f"tier {t.name} missing timeout_s"


def test_router_yaml_parses() -> None:
    r = load_router_process(ROOT / "config" / "router.yaml")
    assert r.binary
    assert 1 <= r.apiserver_port <= 65535
    assert 1 <= r.frontend_port <= 65535


def test_queries_json_parses() -> None:
    q = load_queries(ROOT / "data" / "queries.json")
    assert len(q.queries) >= 100
    assert all(qq.expected_answers for qq in q.queries), \
        "every shipped query should have at least one expected_answers entry"


# ─────────────────────────────────────────────────────────────────────────
# Builder tests — verify the projections-shape config:
#   routing.signals.complexity[]
#   routing.projections.scores.request_difficulty (weighted_sum)
#   routing.projections.mappings.tier_band (threshold_bands, 5 outputs)
#   routing.decisions[] — one per tier, each conditioning on its band
# ─────────────────────────────────────────────────────────────────────────

def test_build_accepts_wrapped_queries_json(tmp_path) -> None:
    """`build(... eval_set_path=...)` must accept both shapes of queries.json:
    bare `[{...}]` and wrapped `{"queries": [{...}]}` (same rule as
    config.load_queries). Regression for a build failure on the wrapped
    form."""
    import json

    from benchmark.build_router_config import build

    wrapped = {
        "queries": [
            {
                "id": "q00001",
                "prompt": "What is 17 + 26?",
                "expected_answers": [
                    {"answer": "43.", "model": "Opus", "provider": "Anthropic"},
                ],
                "expected_min_tier": 1,
                "specializations": ["general"],
            }
        ]
    }
    qp = tmp_path / "queries.json"
    qp.write_text(json.dumps(wrapped))

    cfg = build(
        exemplars_path=ROOT / "config" / "router-exemplars.yaml",
        backends_path=ROOT / "config" / "router-backends.yaml",
        eval_set_path=qp,
    )
    assert cfg["version"] == "v0.3"  # built successfully


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

    # Signals: the canonical v0.3 mix uses multiple signal types.
    # We don't hardcode names — we assert structural properties so the
    # test follows config changes. The shipped config currently uses:
    #   complexity (1: query_difficulty), embeddings (2: detractor +
    #   promoter), structure (5), context (1).
    assert "signals" in routing
    assert "complexity" in routing["signals"], (
        "signals.complexity[] is required (one signal at minimum)"
    )
    complexity_signals = routing["signals"]["complexity"]
    assert complexity_signals, "no complexity signals emitted"
    for sig in complexity_signals:
        assert "name" in sig and "threshold" in sig
        assert sig["hard"]["candidates"], f"signal {sig['name']} missing hard candidates"
        assert sig["easy"]["candidates"], f"signal {sig['name']} missing easy candidates"
    complexity_sig_names = {s["name"] for s in complexity_signals}

    # If embedding signals are configured, they should follow upstream
    # schema (name/threshold/candidates with optional aggregation_method).
    emb_section = routing["signals"].get("embeddings", [])
    for sig in emb_section:
        assert "name" in sig and "threshold" in sig and "candidates" in sig
        assert sig["candidates"], f"embedding {sig['name']} has no candidates"

    # Structure signals (if present) need a feature.type + source block.
    struct_section = routing["signals"].get("structure", [])
    for sig in struct_section:
        assert "name" in sig and "feature" in sig
        assert sig["feature"].get("type"), f"structure {sig['name']} missing feature.type"
        assert sig["feature"].get("source"), f"structure {sig['name']} missing feature.source"

    # Context signals (if present) need at minimum a name.
    ctx_section = routing["signals"].get("context", [])
    for sig in ctx_section:
        assert "name" in sig

    # Projections: scores.request_difficulty + mappings.tier_band.
    assert "projections" in routing
    scores = routing["projections"]["scores"]
    assert len(scores) == 1
    rd = scores[0]
    assert rd["name"] == "request_difficulty"
    assert rd["method"] == "weighted_sum"

    # Inputs by type — each type carries different evidence.
    inputs_by_type: dict[str, list[dict]] = {}
    for inp in rd["inputs"]:
        inputs_by_type.setdefault(inp["type"], []).append(inp)
    assert "complexity" in inputs_by_type, (
        "expected at least one complexity input in the weighted_sum"
    )

    # Read tuning knobs from the exemplars file so the test follows them.
    import yaml as _yaml
    exemplars = _yaml.safe_load(
        (ROOT / "config" / "router-exemplars.yaml").read_text()
    )
    medium_factor = float(exemplars.get("medium_weight_factor", 0.6))

    # Complexity inputs: each signal contributes a `:hard` input; `:medium`
    # inputs are conditional on `medium_weight_factor > 0`.
    expected_hard = {f"{n}:hard" for n in complexity_sig_names}
    expected_medium = {f"{n}:medium" for n in complexity_sig_names} if medium_factor > 0 else set()
    complexity_input_names = {i["name"] for i in inputs_by_type["complexity"]}
    assert complexity_input_names == (expected_hard | expected_medium), (
        f"complexity inputs mismatch: got {complexity_input_names}, "
        f"expected {expected_hard | expected_medium} "
        f"(medium_weight_factor={medium_factor})"
    )
    for inp in inputs_by_type["complexity"]:
        assert inp["name"].endswith((":hard", ":medium"))
        # Complexity inputs MUST use binary mode (omit value_source).
        # `confidence` returns the contrastive margin, which is ~0.0-0.05
        # — too small to drive a [0, 1] weighted_sum.
        assert "value_source" not in inp, (
            f"complexity input {inp['name']!r} must not set value_source"
        )

    # Embedding inputs (if present) MUST use value_source: confidence so
    # the continuous similarity score feeds the weighted_sum. Without it,
    # the binary default discards the continuous signal.
    for inp in inputs_by_type.get("embedding", []):
        assert inp.get("value_source") == "confidence", (
            f"embedding input {inp['name']!r} must set value_source: confidence"
        )

    # Structure / context inputs (if present) should be binary
    # — no value_source — and their weight should be a number.
    for type_name in ("structure", "context"):
        for inp in inputs_by_type.get(type_name, []):
            assert "value_source" not in inp, (
                f"{type_name} input {inp['name']!r} should not set value_source"
            )
            assert isinstance(inp["weight"], int | float)

    # Sanity bound on score range. Negative weights are allowed (detractor
    # embeddings push trivial queries down). The score range derived from
    # all weights should be in [-0.5, 2.0] — bigger means cutoffs need
    # rescaling, smaller means the score can't span all 5 bands.
    pos_sum = sum(i["weight"] for i in rd["inputs"] if i["weight"] > 0)
    neg_sum = sum(i["weight"] for i in rd["inputs"] if i["weight"] < 0)
    assert 0.3 <= pos_sum <= 2.0, (
        f"positive weight sum is {pos_sum:.3f}; expected in [0.3, 2.0]"
    )
    assert -1.0 <= neg_sum <= 0.0, (
        f"negative weight sum is {neg_sum:.3f}; expected in [-1.0, 0.0]"
    )

    mappings = routing["projections"]["mappings"]
    assert len(mappings) == 1
    tb = mappings[0]
    assert tb["name"] == "tier_band"
    assert tb["source"] == "request_difficulty"
    assert tb["method"] == "threshold_bands"
    assert len(tb["outputs"]) == 5, "5 tier bands expected"

    # Decisions: one band-only per tier (5) + zero or more lanes. Lanes
    # are conditional on which signals are configured.
    decisions = routing["decisions"]
    band_only = [d for d in decisions if len(d["rules"]["conditions"]) == 1]
    lane = [d for d in decisions if len(d["rules"]["conditions"]) > 1]
    assert len(band_only) == 5, "expected one band-only decision per tier"

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

    # Each lane decision (if any) must have description, higher priority
    # than band-only, and combine a projection condition with at least
    # one qualifier of another type.
    for d in lane:
        assert d.get("description"), f"lane decision {d['name']!r} missing description"
        assert d["priority"] > band_only[0]["priority"], (
            f"lane {d['name']!r} priority {d['priority']} must beat band priority "
            f"{band_only[0]['priority']}"
        )
        types_in_lane = {c["type"] for c in d["rules"]["conditions"]}
        assert "projection" in types_in_lane
        assert types_in_lane - {"projection"}, (
            f"lane {d['name']!r} has only projection conditions — needs a qualifier"
        )

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
