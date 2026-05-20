#!/usr/bin/env python3
"""
Build a vllm-sr v0.3 config from the public training data + backend infra.

Inputs:
  router-exemplars.yaml   — public training-set artifact (complexity signals
                            + weights + tier cutoffs)
  router-backends.yaml    — infrastructure (endpoints, ports, auth)

Output:
  router-config.yaml      — vllm-sr v0.3 config, ready to load

Routing model — projections (the canonical vllm-sr v0.3 pattern):

  1. `routing.signals.complexity[]` — one entry per complexity signal,
     each with `hard` and `easy` candidate banks. The router's complexity
     classifier produces a contrastive confidence in [0, 1] per signal.

  2. `routing.projections.scores.request_difficulty` — weighted_sum of
     the per-signal confidences. Yields one continuous difficulty score
     per query in [0, 1] (assuming weights sum to 1.0).

  3. `routing.projections.mappings.tier_band` — threshold_bands partitioning
     the score into 5 mutually-exclusive bands (tier1_band .. tier5_band).

  4. `routing.decisions[]` — one decision per tier, condition is
     `{type: projection, name: tierN_band}`. Bands are exclusive so
     exactly one decision fires per query; priority order doesn't matter.

This replaces an earlier DIY design (hard/easy embedding-pair signals +
AND/OR/NOT rule tree) that hit a structural ceiling at ~84% routing
accuracy because vllm-sr's `matched_signals` uses single-winner
semantics — only the top-scoring signal counts as matched, regardless
of how many cleared their threshold. Projections sidestep that by
combining all signals into a continuous score before band-based
classification.

Usage:
  python -m benchmark.build_router_config \\
      --exemplars config/router-exemplars.yaml \\
      --backends config/router-backends.yaml \\
      --out config/router-config.yaml

  vllm-sr serve --config config/router-config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")

# Load .env BEFORE _apply_backend_env_overrides runs. Without this, running
# the build as `python -m benchmark.build_router_config` (the way the
# Makefile invokes it) sees no TIER{N}_1_* vars and the YAML placeholders
# (`model: tier1`, etc.) flow verbatim into the compiled router-config —
# the router then forwards `model: "tier1"` upstream and OpenAI/Anthropic
# 404 with "model not found". The CLI loads dotenv too; this keeps the
# two entry points in sync.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=False)


# ─────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────

# Per-signal "matched" threshold used for matched_signals reporting and
# composer evaluation. Doesn't gate the projection — that uses raw
# confidence regardless.
DEFAULT_SIGNAL_THRESHOLD = 0.55

# Bands are mutually exclusive so band-only decisions can all share one
# priority. Lane decisions (Boolean qualifiers on top of a band match)
# need a HIGHER priority so they outrank the plain band decision they
# override — e.g. a tier5_lane decision conditioning on tier4_band must
# beat the plain route_tier4 when both match.
DEFAULT_DECISION_PRIORITY = 50
LANE_DECISION_PRIORITY = 100


# ─────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────

def _validate_exemplars(ex: dict, eval_prompts: set[str] | None = None) -> None:
    """Sanity-check the exemplars file.

    If `eval_prompts` is provided, refuse to build if any candidate overlaps
    an eval-set prompt — that would contaminate the demo's eval.
    """
    required_top_level = ("tiers", "complexity_signals", "tier_cutoffs")
    missing_keys = [k for k in required_top_level if k not in ex]
    if missing_keys:
        raise ValueError(f"exemplars file missing required sections: {missing_keys}")

    tier_ids = [t["id"] for t in ex["tiers"]]
    if not tier_ids:
        raise ValueError("no tiers defined")
    if len(tier_ids) != len(set(tier_ids)):
        raise ValueError("duplicate tier ids")

    cutoffs = ex["tier_cutoffs"]
    if len(cutoffs) != len(tier_ids) - 1:
        raise ValueError(
            f"tier_cutoffs must have len = num_tiers - 1 "
            f"(got {len(cutoffs)} cutoffs, {len(tier_ids)} tiers)"
        )
    for prev, nxt in zip(cutoffs, cutoffs[1:], strict=False):
        if prev >= nxt:
            raise ValueError(f"tier_cutoffs must be strictly increasing: {cutoffs}")

    signals = ex["complexity_signals"]
    if not signals:
        raise ValueError("at least one complexity_signal required")
    signal_ids = [s["id"] for s in signals]
    if len(signal_ids) != len(set(signal_ids)):
        raise ValueError("duplicate complexity_signal ids")
    for sig in signals:
        for side in ("hard", "easy"):
            cs = sig.get(side, {}).get("candidates", [])
            if len(cs) < 5:
                print(
                    f"WARN: signal {sig['id']!r} has only {len(cs)} {side} candidates; "
                    "5+ recommended for stable scoring",
                    file=sys.stderr,
                )
            if eval_prompts:
                overlap = set(cs) & eval_prompts
                if overlap:
                    raise ValueError(
                        f"signal {sig['id']!r} side {side!r} contains "
                        f"{len(overlap)} prompts that overlap the eval set; "
                        f"this would contaminate the demo. Example: {next(iter(overlap))!r}"
                    )

    # Optional embedding signals — continuous evidence sources fed into
    # the weighted_sum alongside the complexity signals.
    emb_signals = ex.get("embedding_signals", [])
    if emb_signals:
        emb_ids = [s["id"] for s in emb_signals]
        if len(emb_ids) != len(set(emb_ids)):
            raise ValueError("duplicate embedding_signal ids")
        # Embedding ids and complexity ids share a namespace at the
        # weighted_sum-input level; collisions would be ambiguous.
        collision = set(emb_ids) & set(signal_ids)
        if collision:
            raise ValueError(
                f"embedding_signal ids collide with complexity_signal ids: "
                f"{sorted(collision)}"
            )
        for sig in emb_signals:
            cs = sig.get("candidates", [])
            if len(cs) < 5:
                print(
                    f"WARN: embedding_signal {sig['id']!r} has only {len(cs)} "
                    "candidates; 5+ recommended for stable scoring",
                    file=sys.stderr,
                )
            if eval_prompts:
                overlap = set(cs) & eval_prompts
                if overlap:
                    raise ValueError(
                        f"embedding_signal {sig['id']!r} contains "
                        f"{len(overlap)} prompts that overlap the eval set; "
                        f"this would contaminate the demo. Example: {next(iter(overlap))!r}"
                    )

    # Structure / context signals — purely declarative (no exemplar
    # overlap risk with the eval set since they're mechanical gates,
    # not semantic similarity).
    for section, required_fields in (
        ("structure_signals", ("id", "feature")),
        ("context_signals", ("id",)),
    ):
        section_signals = ex.get(section, [])
        if not section_signals:
            continue
        seen_ids: set[str] = set()
        for sig in section_signals:
            for field in required_fields:
                if field not in sig:
                    raise ValueError(
                        f"{section}: entry missing required field {field!r}: {sig!r}"
                    )
            sid = sig["id"]
            if sid in seen_ids:
                raise ValueError(f"{section}: duplicate id {sid!r}")
            seen_ids.add(sid)


def _translate_host_for_router_container(url: str) -> str:
    """Swap a host-side localhost URL for the docker-equivalent.

    `.env` carries URLs from the perspective of `make answers`, which
    runs on the host and reaches local vLLM containers via `localhost`.
    The router service, however, runs INSIDE a docker container, where
    `localhost` is the container's own loopback (no vLLM there). The
    docker convention for "the host's network" is `host.docker.internal`
    (resolved via Docker's host-gateway DNS).

    Two forms get translated; everything else passes through:
      http://localhost:8001/v1  →  http://host.docker.internal:8001/v1
      http://127.0.0.1:8001/v1  →  http://host.docker.internal:8001/v1
    """
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in ("localhost", "127.0.0.1"):
        return url
    new_netloc = "host.docker.internal"
    if parsed.port is not None:
        new_netloc = f"{new_netloc}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=new_netloc))


def _apply_backend_env_overrides(be: dict) -> None:
    """Apply `TIER{N}_1_*` env overrides to the router-backends.yaml schema.

    The router only needs ONE endpoint per tier (it doesn't fan out —
    multi-model fan-out happens at `make answers`). Slot 1 is the
    canonical "this is the endpoint the router uses to reach tier N"
    entry; TIER{N}_1_URL / TIER{N}_1_MODEL / TIER{N}_1_API_KEY override
    base_url / model / api_key_env for that tier in router-backends.yaml.

    Bare `TIER{N}_*` env vars (the old single-model form, no longer
    supported) raise with a migration hint.
    """
    backends = be.get("backends") or {}
    legacy_suffixes = ("URL", "MODEL", "API_KEY")
    for tier_id, cfg in backends.items():
        if not (isinstance(tier_id, str) and tier_id.startswith("tier")):
            continue
        suffix = tier_id[len("tier"):]
        if not suffix.isdigit():
            continue
        n = suffix

        # Reject any bare TIER{n}_<suffix> — slots are indexed only.
        for s in legacy_suffixes:
            if os.environ.get(f"TIER{n}_{s}", "").strip():
                raise ValueError(
                    f"TIER{n}_{s} is not supported. Use TIER{n}_1_{s} "
                    f"(slot 1) — env slots are indexed from 1, with no "
                    f"bare/slot-0 form."
                )

        url = os.environ.get(f"TIER{n}_1_URL", "").strip()
        if url:
            cfg["base_url"] = _translate_host_for_router_container(url)

        model = os.environ.get(f"TIER{n}_1_MODEL", "").strip()
        if model:
            cfg["model"] = model

        key = os.environ.get(f"TIER{n}_1_API_KEY", "").strip()
        if key:
            cfg["api_key_env"] = f"TIER{n}_1_API_KEY"


def _validate_backends(be: dict, declared_tier_ids: set[str]) -> None:
    backend_ids = set(be["backends"].keys())
    missing = declared_tier_ids - backend_ids
    if missing:
        raise ValueError(f"backends file is missing entries for tiers: {sorted(missing)}")
    extra = backend_ids - declared_tier_ids
    if extra:
        print(
            f"WARN: backends file has unused entries: {sorted(extra)}",
            file=sys.stderr,
        )


# ─────────────────────────────────────────────────────────────────────────
# Provider / backend emitters (unchanged from the previous design)
# ─────────────────────────────────────────────────────────────────────────

def _is_anthropic(base_url: str) -> bool:
    return "anthropic.com" in base_url.lower()


def _is_https(base_url: str) -> bool:
    return base_url.lower().startswith("https://")


def _apply_api_key(ref: dict, cfg: dict) -> None:
    """Inline the resolved API key value into the backend ref, so the
    router service doesn't need TIER{N}_{i}_API_KEY env vars propagated
    into its docker container (which `vllm-sr serve`'s compose template
    doesn't do by default).

    Resolution order:
      • `cfg.api_key_env` names an env var with a non-empty value →
        write `ref.api_key = <value>` (router reads from config).
      • `cfg.api_key_env` is set but the value is empty/missing → keep
        `ref.api_key_env = <name>` so the router 401s with a clear
        "no key" error rather than silently authing with nothing.
      • No `api_key_env` at all (local HTTP backend) → no-op.
    """
    name = cfg.get("api_key_env")
    if not name:
        return
    value = os.environ.get(name, "").strip()
    if value:
        ref["api_key"] = value
    else:
        ref["api_key_env"] = name


def _emit_backend_ref_oai(cfg: dict) -> dict:
    """HTTP localhost OAI-compatible backend (e.g. vLLM-served local model).

    Used for the `protocol: http` path. Strips any default-port info from
    the URL because urlparse(...).port is None for URLs without an explicit
    port — `host:None` is not what vllm-sr wants.
    """
    u = urlparse(cfg["base_url"])
    host = u.hostname or ""
    port_part = f":{u.port}" if u.port else ""
    endpoint = f"{host}{port_part}{u.path}".rstrip("/")
    ref: dict[str, Any] = {
        "name": "primary",
        "endpoint": endpoint,
        "protocol": "http",
        "weight": 100,
    }
    _apply_api_key(ref, cfg)
    return ref


def _emit_backend_ref_anthropic(cfg: dict) -> dict:
    base = cfg["base_url"].rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    ref: dict[str, Any] = {
        "name": "primary",
        "base_url": base,
        "provider": "anthropic",
        "weight": 100,
    }
    _apply_api_key(ref, cfg)
    return ref


def _emit_backend_ref_openai(cfg: dict) -> dict:
    """HTTPS OpenAI-compatible vendor backend (api.openai.com, OpenRouter,
    Fireworks, etc.). Mirrors the upstream canonical config.yaml shape:
    `provider: openai`, `base_url:` (with scheme), `auth_header` +
    `auth_prefix` for Bearer auth, key resolved inline from env at build.
    """
    base = cfg["base_url"].rstrip("/")
    ref: dict[str, Any] = {
        "name": "primary",
        "base_url": base,
        "provider": "openai",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer",
        "weight": 100,
    }
    _apply_api_key(ref, cfg)
    return ref


def _emit_provider_model(tier_id: str, cfg: dict) -> dict:
    """Choose the right backend ref shape based on the configured base_url.

    The model card's `name:` is the **tier id** (tier1…tier5), not the
    upstream model identifier. The router uses this name in
    `x-vsr-selected-model` and in decision modelRefs. `make route`
    routes through the local OAI mock by default — the mock ACKs any
    model name — so the tier-level abstraction is the right shape:
    routing decisions, headers, and TierLookup all speak tier IDs.

    `make answers` is what actually calls real upstream models, and it
    bypasses the router entirely (uses OAIClient / AnthropicClient
    directly), so the per-vendor model names live in `.env` rather than
    leaking into the router's compiled config.

    Three backend_ref shapes (chosen by base_url):
      • anthropic.com → Anthropic provider (handles OAI→Anthropic shape).
      • Any other HTTPS → generic OpenAI-compatible vendor (api.openai.com,
        OpenRouter, etc.). Uses Bearer auth, `provider: openai`.
      • HTTP (localhost) → the existing local-OAI shape (`protocol: http`).
    """
    base_url = cfg["base_url"]
    if _is_anthropic(base_url):
        api_format = "anthropic"
        ref = _emit_backend_ref_anthropic(cfg)
    elif _is_https(base_url):
        api_format = "openai"
        ref = _emit_backend_ref_openai(cfg)
    else:
        api_format = "openai"
        ref = _emit_backend_ref_oai(cfg)
    return {
        "name": tier_id,
        "provider_model_id": cfg["model"],
        "api_format": api_format,
        "backend_refs": [ref],
    }


def _emit_provider_model_mock(tier_id: str, mock_endpoint: str) -> dict:
    """All tiers point at the local mock; api_format forced to openai.
    The mock ACKs any model name — naming the card after the tier id
    keeps the abstraction clean."""
    return {
        "name": tier_id,
        "provider_model_id": tier_id,
        "api_format": "openai",
        "backend_refs": [
            {
                "name": "primary",
                "endpoint": mock_endpoint,
                "protocol": "http",
                "weight": 100,
            }
        ],
    }


def _emit_model_card(tier: dict) -> dict:
    desc_parts = [p for p in (tier.get("label"), tier.get("role")) if p]
    return {
        "name": tier["id"],
        "modality": "text",
        "description": " — ".join(desc_parts) if desc_parts else tier["id"],
    }


# ─────────────────────────────────────────────────────────────────────────
# Routing emitters (the new shape)
# ─────────────────────────────────────────────────────────────────────────

def _emit_complexity_signal(sig: dict) -> dict:
    """One entry under routing.signals.complexity[]."""
    out: dict[str, Any] = {
        "name": sig["id"],
        "threshold": sig.get("threshold", DEFAULT_SIGNAL_THRESHOLD),
        "hard": {"candidates": list(sig["hard"]["candidates"])},
        "easy": {"candidates": list(sig["easy"]["candidates"])},
    }
    if sig.get("description"):
        out["description"] = sig["description"]
    return out


def _emit_embedding_signal(sig: dict) -> dict:
    """One entry under routing.signals.embeddings[].

    Embedding signals have a single candidates bank (no hard/easy contrast)
    and produce continuous confidence in [0, ~0.7] via cosine similarity.
    aggregation_method=max returns "best match against any candidate".
    """
    out: dict[str, Any] = {
        "name": sig["id"],
        "threshold": sig.get("threshold", DEFAULT_SIGNAL_THRESHOLD),
        "aggregation_method": sig.get("aggregation_method", "max"),
        "candidates": list(sig["candidates"]),
    }
    if sig.get("description"):
        out["description"] = sig["description"]
    return out


def _emit_embedding_input(sig: dict) -> dict:
    """One weighted_sum input for an embedding signal.

    Uses `value_source: confidence` so the continuous similarity score
    feeds the weighted_sum (unlike complexity inputs, which use the
    binary default and contribute their full weight on match). This is
    the upstream "mixed sources" pattern: continuous embeddings +
    binary-qualifier complexity, both fed into one weighted_sum.
    """
    return {
        "type": "embedding",
        "name": sig["id"],
        "weight": float(sig.get("weight", 0.0)),
        "value_source": "confidence",
    }


def _emit_structure_signal(sig: dict) -> dict:
    """One entry under routing.signals.structure[].

    Structure signals measure mechanical properties of the prompt itself:
    word counts, regex matches, ordered token sequences, density of
    constraint markers. They're heuristic and deterministic — cheap to
    compute, carry information orthogonal to embeddings.

    Schema (from upstream config.yaml):
      - name (str), required
      - feature: {type: count|exists|sequence|density, source: {...}}
      - predicate: {gte|gt|lte|lt: <number>}  (optional for exists/sequence)
      - description (str), optional
    """
    out: dict[str, Any] = {
        "name": sig["id"],
        "feature": sig["feature"],
    }
    if "predicate" in sig:
        out["predicate"] = sig["predicate"]
    if sig.get("description"):
        out["description"] = sig["description"]
    return out


def _emit_context_signal(sig: dict) -> dict:
    """One entry under routing.signals.context[].

    Context signals gate on request-level facts the router already knows
    — token counts, modality, etc. Example: long_context fires for
    requests in [min_tokens, max_tokens].
    """
    out: dict[str, Any] = {"name": sig["id"]}
    for field in ("min_tokens", "max_tokens", "modality"):
        if field in sig:
            out[field] = sig[field]
    if sig.get("description"):
        out["description"] = sig["description"]
    return out


def _emit_typed_input(sig: dict, type_name: str) -> dict:
    """Generic weighted_sum input for a non-embedding signal type.

    Used for structure/context inputs — these fire binary
    (match=1.0 / miss=0.0) by default, no value_source needed.
    """
    return {
        "type": type_name,
        "name": sig["id"],
        "weight": float(sig.get("weight", 0.0)),
    }


# Default `:medium` weight factor when the exemplars file doesn't specify
# `medium_weight_factor:`. A `:medium` match contributes this fraction of
# a `:hard` match's weight. Mirrors the canonical ratio in upstream
# config/config.yaml (0.18 medium / 0.36 hard = 0.5). A given signal
# matches at ONE level per query (medium and hard are mutually exclusive
# per `matched_signals`), so per-signal contribution stays ≤ weight and
# total request_difficulty stays bounded in [0, 1] as long as the
# per-signal weights themselves sum to ≤ 1.0.
DEFAULT_MEDIUM_WEIGHT_FACTOR = 0.6


def _emit_difficulty_score(
    signals: list[dict],
    medium_weight_factor: float,
    emb_signals: list[dict] | None = None,
    structure_signals: list[dict] | None = None,
    context_signals: list[dict] | None = None,
) -> dict:
    """`routing.projections.scores.request_difficulty` — weighted_sum
    over the canonical signal-type mix: complexity, embeddings, structure,
    context.

    Schema notes (per upstream vllm-project/semantic-router config.yaml
    and confirmed by inspecting a real eval response):
      • Complexity inputs reference `<signal_id>:hard` or `<signal_id>:medium`,
        not bare `<signal_id>`. The bare form binds to nothing → silent 0.
      • We OMIT `value_source` here. Upstream omits it too, and the docs
        state the default is binary (match=1.0 / miss=0.0). Setting
        `value_source: confidence` instead returns the CONTRASTIVE MARGIN
        (text_hard_score - text_easy_score), typically 0.0-0.05 — far too
        small to clear the band cutoffs. Caused 100% of queries to land
        in tier1_band in our first projections roll-out.
      • For a given query, a signal matches at exactly ONE level
        (observed empirically: `matched_signals.complexity` and
        `unmatched_signals.complexity` are disjoint per level).

    We include both `:medium` (half weight) and `:hard` (full weight) so
    queries that match the hard bank but only weakly still get partial
    credit toward the projected score.
    """
    inputs: list[dict[str, Any]] = []
    # When medium_weight_factor is 0, skip the :medium inputs entirely
    # rather than emit them with weight 0. Cleaner generated config, and
    # documents the design intent: ":medium fires uniformly across queries
    # in our setup, so its contribution is dead weight." See PLAN.md / the
    # routing diagnostic for why this matters.
    emit_medium = medium_weight_factor > 0.0
    for sig in signals:
        weight = float(sig.get("weight", 0.0))
        if emit_medium:
            inputs.append({
                "type": "complexity",
                "name": f"{sig['id']}:medium",
                "weight": weight * medium_weight_factor,
            })
        inputs.append({
            "type": "complexity",
            "name": f"{sig['id']}:hard",
            "weight": weight,
        })
    # Append embedding inputs after complexity ones — they feed continuous
    # confidence into the same weighted_sum (upstream mixed-source pattern).
    for sig in emb_signals or []:
        inputs.append(_emit_embedding_input(sig))
    # Structure / context signals fire binary (match=1.0/miss=0.0).
    # Each contributes its weight on match, zero on miss. No value_source.
    for sig in structure_signals or []:
        inputs.append(_emit_typed_input(sig, "structure"))
    for sig in context_signals or []:
        inputs.append(_emit_typed_input(sig, "context"))
    return {
        "name": "request_difficulty",
        "method": "weighted_sum",
        "inputs": inputs,
    }


def _emit_tier_band_mapping(tier_ids: list[str], cutoffs: list[float]) -> dict:
    """`routing.projections.mappings.tier_band` — threshold_bands turning
    the continuous request_difficulty score into one band per tier."""
    outputs: list[dict[str, Any]] = []
    for i, tier_id in enumerate(tier_ids):
        band: dict[str, Any] = {"name": f"{tier_id}_band"}
        if i > 0:
            band["gt"] = cutoffs[i - 1]
        if i < len(cutoffs):
            band["lte"] = cutoffs[i]
        outputs.append(band)
    return {
        "name": "tier_band",
        "source": "request_difficulty",
        "method": "threshold_bands",
        "outputs": outputs,
    }


def _emit_decisions(
    tier_ids: list[str],
    complexity_signals: list[dict],
    emb_signals: list[dict],
) -> list[dict]:
    """Build the `routing.decisions[]` list, conditional on which signals
    are configured.

    Lane decisions reference specific signal names. If those signals
    aren't present in the exemplars file, the lane can't fire — so we
    omit it rather than emit dead references. This keeps the canonical
    diverse-signal config clean: when the exemplars file uses a single
    `query_difficulty` complexity signal (no `needs_reasoning` etc.),
    the four-signal-era lanes don't emit.

    Band decisions (one per tier) always emit.
    """
    complexity_ids = {s["id"] for s in complexity_signals}
    emb_ids = {s["id"] for s in emb_signals or []}
    decisions: list[dict] = []

    # Lane decisions FIRST, so a quick read of the generated config shows
    # the higher-priority overrides up top. Each is conditional on its
    # referenced signals being defined.
    if {"needs_reasoning", "needs_expertise"} <= complexity_ids:
        decisions.append(_emit_tier5_frontier_lane())
    if {"needs_judgment", "demands_commitment"} <= complexity_ids:
        decisions.append(_emit_tier5_committed_judgment_lane())
    if "frontier_synthesis" in emb_ids:
        decisions.append(_emit_tier5_embedding_frontier_lane("frontier_synthesis"))

    # Band decisions (one per tier) always emit.
    decisions.extend(_emit_decision_for_band(tier_id) for tier_id in tier_ids)
    return decisions


def _emit_decision_for_band(tier_id: str) -> dict:
    """One `routing.decisions[]` entry — fires when its band is active."""
    return {
        "name": f"route_{tier_id}",
        "description": f"Route to {tier_id} when request_difficulty lands in {tier_id}_band.",
        "priority": DEFAULT_DECISION_PRIORITY,
        "rules": {
            "operator": "AND",
            "conditions": [{"type": "projection", "name": f"{tier_id}_band"}],
        },
        "modelRefs": [{"model": tier_id, "use_reasoning": False}],
    }


def _emit_tier5_embedding_frontier_lane(embedding_signal_id: str) -> dict:
    """Override decision: route to tier5 from inside tier4_band when the
    `frontier_synthesis` embedding signal matches.

    Embedding signals fire as `matched` when their cosine similarity
    exceeds the configured threshold (0.55 in our exemplars). For a
    query that looks frontier-coded on at least one of the bank's
    archetype patterns, this lane catches it even without any complexity
    signal reaching :hard.

    Empirically this is the most-likely-to-fire of the three T5 lanes
    with the current exemplar set, because the frontier embedding bank
    is specifically tuned to the kind of prompts the new T5 queries
    look like (long-form, formal, multi-part, commitment-demanding).
    """
    return {
        "name": "route_tier5_embedding_frontier",
        "description": (
            f"Promote tier4_band queries to tier5 when the "
            f"{embedding_signal_id!r} embedding signal matches — "
            "continuous-evidence path to T5."
        ),
        "priority": LANE_DECISION_PRIORITY,
        "rules": {
            "operator": "AND",
            "conditions": [
                {"type": "projection", "name": "tier4_band"},
                {"type": "embedding", "name": embedding_signal_id},
            ],
        },
        "modelRefs": [{"model": "tier5", "use_reasoning": False}],
    }


def _emit_tier5_committed_judgment_lane() -> dict:
    """Override decision: route to tier5 from inside tier4_band when the
    query hits BOTH `needs_judgment:hard` AND `demands_commitment:hard`.

    The canonical "frontier-advice" path. The exemplars-file role for T5
    is "BOTH deeply technical AND demands judgment" — this lane captures
    the *judgment-and-commitment* axis of that definition, complementing
    the technical-and-expertise axis covered by route_tier5_frontier.

    Empirically: if the embedder doesn't produce :hard on either input
    (the common case with the current exemplars), this lane never fires
    and route_tier4 wins on the band alone. Adding it costs nothing and
    provides a structurally clean path when the signal arrives.
    """
    return {
        "name": "route_tier5_committed_judgment",
        "description": (
            "Promote tier4_band queries to tier5 when both judgment AND "
            "commitment hit :hard — the canonical frontier-advice lane."
        ),
        "priority": LANE_DECISION_PRIORITY,
        "rules": {
            "operator": "AND",
            "conditions": [
                {"type": "projection", "name": "tier4_band"},
                {"type": "complexity", "name": "needs_judgment:hard"},
                {"type": "complexity", "name": "demands_commitment:hard"},
            ],
        },
        "modelRefs": [{"model": "tier5", "use_reasoning": False}],
    }


def _emit_tier5_frontier_lane() -> dict:
    """Override decision: route to tier5 from inside tier4_band when the
    query hits BOTH `needs_reasoning:hard` AND `needs_expertise:hard`.

    Why this exists — and why it's a `lane` rather than tighter cutoffs:
    the score+bands axis is the right shape for the continuous T1-T4
    distinction, but distinguishing T4 from T5 is a *lane* distinction
    (canonical T5 per the exemplars file: "needed when a query is BOTH
    deeply technical AND demands judgment"). Boolean composition on top
    of the band, with higher priority than route_tier4, expresses that
    cleanly without trying to extract a fifth tier of resolution from
    the same continuous score.

    Per the projections design doc:
      "decisions such as premium_legal or reasoning_deep combine raw
       domain matches with projection outputs" — same pattern here.

    Empirically: if the embedder doesn't produce :hard for the input,
    this lane never fires — no harm done, route_tier4 (band-only) wins.
    """
    return {
        "name": "route_tier5_frontier",
        "description": (
            "Promote tier4_band queries to tier5 when they look deeply "
            "technical AND demanding (needs_reasoning AND needs_expertise "
            "both at :hard) — the canonical frontier-synthesis lane."
        ),
        "priority": LANE_DECISION_PRIORITY,
        "rules": {
            "operator": "AND",
            "conditions": [
                {"type": "projection", "name": "tier4_band"},
                {"type": "complexity", "name": "needs_reasoning:hard"},
                {"type": "complexity", "name": "needs_expertise:hard"},
            ],
        },
        "modelRefs": [{"model": "tier5", "use_reasoning": False}],
    }


# ─────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────

def build(
    exemplars_path: Path,
    backends_path: Path,
    eval_set_path: Path | None,
    *,
    mock_endpoint: str | None = None,
) -> dict:
    """Read both inputs, validate, emit a vllm-sr v0.3 config dict.

    `mock_endpoint` (e.g. `host.docker.internal:18811/v1`) overrides every
    backend with the local OAI mock.
    """
    ex = yaml.safe_load(exemplars_path.read_text())
    be = yaml.safe_load(backends_path.read_text())
    # Env overrides: TIER{N}_URL / TIER{N}_MODEL / TIER{N}_API_KEY win
    # over YAML when set. Lets .env be the single user-facing place to
    # flip per-tier endpoint config.
    _apply_backend_env_overrides(be)

    eval_prompts: set[str] | None = None
    if eval_set_path:
        # queries.json may be a bare list or wrapped as {"queries": [...]}.
        # Use the canonical loader so we accept both shapes (same rule as
        # config.load_queries).
        from .config import load_queries
        eval_prompts = {q.prompt for q in load_queries(eval_set_path).queries}

    _validate_exemplars(ex, eval_prompts)
    tier_ids = [t["id"] for t in ex["tiers"]]
    _validate_backends(be, set(tier_ids))

    signals = ex["complexity_signals"]
    emb_signals = ex.get("embedding_signals", [])
    structure_signals = ex.get("structure_signals", [])
    context_signals = ex.get("context_signals", [])
    cutoffs = list(ex["tier_cutoffs"])
    medium_weight_factor = float(
        ex.get("medium_weight_factor", DEFAULT_MEDIUM_WEIGHT_FACTOR)
    )
    if not 0.0 <= medium_weight_factor <= 1.0:
        raise ValueError(
            f"medium_weight_factor must be in [0, 1]; got {medium_weight_factor}"
        )

    config: dict[str, Any] = {
        "version": "v0.3",
        "listeners": [
            {
                "name": "http",
                "address": be["listener"]["address"],
                "port": be["listener"]["port"],
                "timeout": be["listener"]["timeout"],
            }
        ],
        "providers": {
            "defaults": {"default_model": be.get("default_tier", tier_ids[0])},
            "models": [
                _emit_provider_model_mock(tier_id, mock_endpoint)
                if mock_endpoint
                else _emit_provider_model(tier_id, cfg)
                for tier_id, cfg in be["backends"].items()
            ],
        },
        "routing": {
            "modelCards": [_emit_model_card(t) for t in ex["tiers"]],
            "signals": {
                "complexity": [_emit_complexity_signal(s) for s in signals],
                **(
                    {"embeddings": [_emit_embedding_signal(s) for s in emb_signals]}
                    if emb_signals
                    else {}
                ),
                **(
                    {"structure": [_emit_structure_signal(s) for s in structure_signals]}
                    if structure_signals
                    else {}
                ),
                **(
                    {"context": [_emit_context_signal(s) for s in context_signals]}
                    if context_signals
                    else {}
                ),
            },
            "projections": {
                "scores": [
                    _emit_difficulty_score(
                        signals,
                        medium_weight_factor,
                        emb_signals,
                        structure_signals=structure_signals,
                        context_signals=context_signals,
                    )
                ],
                "mappings": [_emit_tier_band_mapping(tier_ids, cutoffs)],
            },
            "decisions": _emit_decisions(
                tier_ids, signals, emb_signals
            ),
        },
        # Disable semantic cache: avoids Milvus startup dependency for the
        # demo. Enable the complexity prototype-scoring module explicitly
        # (default may already be on; setting it is belt-and-suspenders).
        "global": {
            "stores": {
                "semantic_cache": {"enabled": False},
            },
            "model_catalog": {
                "modules": {
                    "complexity": {
                        "prototype_scoring": {"enabled": True},
                    },
                },
            },
        },
    }
    return config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exemplars", type=Path, default=Path("config/router-exemplars.yaml"))
    p.add_argument("--backends", type=Path, default=Path("config/router-backends.yaml"))
    p.add_argument("--out", type=Path, default=Path("config/router-config.yaml"))
    p.add_argument(
        "--check-against-eval",
        type=Path,
        help="Path to data/queries.json — refuse to build if any candidate overlaps an eval prompt",
    )
    p.add_argument(
        "--mock-endpoint",
        type=str,
        default=None,
        help=(
            "Route every tier to this host:port[/path] instead of the configured "
            "backends. From inside the router container, use e.g. "
            "host.docker.internal:18811/v1. Pipeline-verification only."
        ),
    )
    args = p.parse_args()

    config = build(
        args.exemplars,
        args.backends,
        args.check_against_eval,
        mock_endpoint=args.mock_endpoint,
    )
    args.out.write_text(yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
