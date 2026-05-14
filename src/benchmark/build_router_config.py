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
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")


# ─────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────

# Per-signal "matched" threshold used for matched_signals reporting and
# composer evaluation. Doesn't gate the projection — that uses raw
# confidence regardless.
DEFAULT_SIGNAL_THRESHOLD = 0.55

# Same priority for every decision: bands are mutually exclusive so only
# one fires regardless of priority.
DEFAULT_DECISION_PRIORITY = 50


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


def _emit_backend_ref_oai(cfg: dict) -> dict:
    u = urlparse(cfg["base_url"])
    endpoint = f"{u.hostname}:{u.port}{u.path}".rstrip("/")
    ref: dict[str, Any] = {
        "name": "primary",
        "endpoint": endpoint,
        "protocol": "http",
        "weight": 100,
    }
    if "api_key_env" in cfg:
        ref["api_key_env"] = cfg["api_key_env"]
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
    if "api_key_env" in cfg:
        ref["api_key_env"] = cfg["api_key_env"]
    return ref


def _emit_provider_model(tier_id: str, cfg: dict) -> dict:
    anthropic = _is_anthropic(cfg["base_url"])
    return {
        "name": tier_id,
        "provider_model_id": cfg["model"],
        "api_format": "anthropic" if anthropic else "openai",
        "backend_refs": [
            _emit_backend_ref_anthropic(cfg) if anthropic else _emit_backend_ref_oai(cfg)
        ],
    }


def _emit_provider_model_mock(tier_id: str, mock_endpoint: str) -> dict:
    """All tiers point at the local mock; api_format forced to openai."""
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


def _emit_difficulty_score(signals: list[dict]) -> dict:
    """`routing.projections.scores.request_difficulty` — weighted_sum of
    each complexity signal's HARD-side confidence.

    Per the upstream canonical config (config/config.yaml in
    vllm-project/semantic-router), complexity-input references in a
    weighted_sum take the form `<signal_id>:hard` (or `:medium`), not
    just `<signal_id>`. With value_source=confidence, the runtime uses
    the matched signal's confidence or 0 when it didn't match — so we
    pull the HARD side, which represents "this query is hard".
    """
    inputs = [
        {
            "type": "complexity",
            "name": f"{sig['id']}:hard",
            "weight": float(sig.get("weight", 0.0)),
            "value_source": "confidence",
        }
        for sig in signals
    ]
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

    `mock_endpoint` (e.g. `host.docker.internal:8811/v1`) overrides every
    backend with the local OAI mock.
    """
    ex = yaml.safe_load(exemplars_path.read_text())
    be = yaml.safe_load(backends_path.read_text())

    eval_prompts: set[str] | None = None
    if eval_set_path:
        eval_data = json.loads(eval_set_path.read_text())
        eval_prompts = {entry["prompt"] for entry in eval_data}

    _validate_exemplars(ex, eval_prompts)
    tier_ids = [t["id"] for t in ex["tiers"]]
    _validate_backends(be, set(tier_ids))

    signals = ex["complexity_signals"]
    cutoffs = list(ex["tier_cutoffs"])

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
            },
            "projections": {
                "scores": [_emit_difficulty_score(signals)],
                "mappings": [_emit_tier_band_mapping(tier_ids, cutoffs)],
            },
            "decisions": [_emit_decision_for_band(tier_id) for tier_id in tier_ids],
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
            "host.docker.internal:8811/v1. Pipeline-verification only."
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
