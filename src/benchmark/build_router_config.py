#!/usr/bin/env python3
"""
Build a vllm-sr v0.3 config from the public training data + backend infra.

Inputs:
  router-exemplars.yaml   — public training-set artifact (exemplars + rules)
  router-backends.yaml    — infrastructure (endpoints, ports, auth)

Output:
  router-config.yaml      — vllm-sr v0.3 config, ready to load

Why the two-file split:
  The exemplars file is the audience-facing "what the router knows" artifact.
  It uses a band-based mental model (low/medium/high per axis) that reads
  cleanly top-to-bottom. v0.3 only supports binary embedding signals
  (matches/doesn't match), so this builder does the band→Boolean translation
  under the hood. Edit the exemplars file; the generated config is just an
  artifact.

Schema mapping (band → v0.3 conditions for one axis):

  [low]              → easy AND NOT hard           (clearly on the easy side)
  [high]             → hard                        (matches a hard exemplar)
  [medium]           → NOT easy AND NOT hard       (in the gap)
  [low, medium]      → NOT hard                    (not on the hard side)
  [medium, high]     → NOT easy                    (not on the easy side)
  [low, high]        → easy OR hard                (clearly classified)
  [low, medium, high]→ (no condition for that axis)

`requires_any_high: [a, b]` → `{operator: OR, conditions: [a_hard, b_hard]}`.

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
# Tunable defaults
# ─────────────────────────────────────────────────────────────────────────
# Starting threshold for embedding similarity matches. Tune after the first
# real make route run. The exemplars file can override per-axis via:
#   axis.threshold_hard, axis.threshold_easy (both floats in [0, 1])
DEFAULT_EMBEDDING_THRESHOLD = 0.5

# Priority spacing for emitted decisions — higher fires first.
# Rules from the exemplars file get priorities 100, 90, 80, ... in order.
PRIORITY_STEP = 10
PRIORITY_BASE = 100


# ─────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────

def _validate_exemplars(ex: dict, eval_prompts: set[str] | None = None) -> None:
    """Sanity-check the exemplars file.

    If `eval_prompts` is provided, refuse to build if any exemplar overlaps
    an eval-set prompt — that would contaminate the demo's eval.
    """
    if "tiers" not in ex or "axes" not in ex or "rules" not in ex:
        raise ValueError("exemplars file missing required sections (tiers/axes/rules)")

    tier_ids = {t["id"] for t in ex["tiers"]}
    if not tier_ids:
        raise ValueError("no tiers defined")

    for axis in ex["axes"]:
        for side in ("hard", "easy"):
            exs = axis["exemplars"].get(side, [])
            if len(exs) < 5:
                print(
                    f"WARN: axis {axis['id']!r} has only {len(exs)} {side} exemplars; "
                    "5+ recommended for stable similarity scoring",
                    file=sys.stderr,
                )
            if eval_prompts:
                overlap = set(exs) & eval_prompts
                if overlap:
                    raise ValueError(
                        f"axis {axis['id']!r} side {side!r} contains "
                        f"{len(overlap)} prompts that overlap the eval set; "
                        f"this would contaminate the demo. Example: {next(iter(overlap))!r}"
                    )

    for rule in ex["rules"]:
        if rule["route_to"] not in tier_ids:
            raise ValueError(
                f"rule {rule['name']!r} routes to unknown tier {rule['route_to']!r}"
            )


def _validate_backends(be: dict, declared_tier_ids: set[str]) -> None:
    """Ensure every declared tier has a matching backend."""
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
# Condition tree builders (small, easily testable atoms)
# ─────────────────────────────────────────────────────────────────────────

def _embed_cond(signal_name: str) -> dict:
    return {"type": "embedding", "name": signal_name}


def _not_(cond: dict) -> dict:
    return {"operator": "NOT", "conditions": [cond]}


def _and_(conds: list[dict]) -> dict:
    return {"operator": "AND", "conditions": conds}


def _or_(conds: list[dict]) -> dict:
    return {"operator": "OR", "conditions": conds}


def _axis_condition(axis_id: str, bands: list[str]) -> dict | None:
    """Translate a band set on one axis to a v0.3 condition tree.

    Returns None if the band set covers all three (no constraint on this axis).
    """
    band_set = set(bands)
    valid = {"low", "medium", "high"}
    unknown = band_set - valid
    if unknown:
        raise ValueError(f"axis {axis_id!r}: unknown bands {sorted(unknown)}")

    hard = _embed_cond(f"{axis_id}_hard")
    easy = _embed_cond(f"{axis_id}_easy")

    if band_set == valid:
        return None
    if band_set == {"low"}:
        return _and_([easy, _not_(hard)])
    if band_set == {"high"}:
        return hard
    if band_set == {"medium"}:
        return _and_([_not_(hard), _not_(easy)])
    if band_set == {"low", "medium"}:
        return _not_(hard)
    if band_set == {"medium", "high"}:
        return _not_(easy)
    if band_set == {"low", "high"}:
        return _or_([easy, hard])
    raise ValueError(f"unrecognized band set for axis {axis_id!r}: {sorted(band_set)}")


def _emit_decision(rule: dict, priority: int) -> dict:
    """Translate one exemplars-format rule to a v0.3 decision."""
    conditions: list[dict] = []

    for axis_id, bands in rule.get("when", {}).items():
        cond = _axis_condition(axis_id, bands)
        if cond is not None:
            conditions.append(cond)

    if "requires_any_high" in rule:
        conditions.append(_or_([
            _embed_cond(f"{axis}_hard")
            for axis in rule["requires_any_high"]
        ]))

    out: dict[str, Any] = {
        "name": rule["name"],
        "priority": priority,
        "rules": {
            "operator": "AND",
            "conditions": conditions,  # empty list = unconditional fallthrough
        },
        "modelRefs": [{"model": rule["route_to"], "use_reasoning": False}],
    }
    if rule.get("description"):
        out["description"] = rule["description"]
    return out


# ─────────────────────────────────────────────────────────────────────────
# Signal emitters
# ─────────────────────────────────────────────────────────────────────────

def _emit_embedding_signal(
    axis_id: str, side: str, candidates: list[str], threshold: float
) -> dict:
    """One v0.3 embedding signal: matches when max similarity to any
    candidate exceeds threshold."""
    return {
        "name": f"{axis_id}_{side}",
        "threshold": threshold,
        "aggregation_method": "max",
        "candidates": candidates,
    }


# ─────────────────────────────────────────────────────────────────────────
# Provider / backend emitters
# ─────────────────────────────────────────────────────────────────────────

def _is_anthropic(base_url: str) -> bool:
    return "anthropic.com" in base_url.lower()


def _emit_backend_ref_oai(cfg: dict) -> dict:
    """An OAI-compatible local endpoint (vLLM, llama.cpp, our mock, etc.)."""
    # The agent-smoke upstream example uses `endpoint: "host:port/v1"` + `protocol: "http"`.
    # Preserve the user-supplied base_url verbatim — we only know it works as configured.
    u = urlparse(cfg["base_url"])
    endpoint = f"{u.hostname}:{u.port}{u.path}".rstrip("/")
    ref = {
        "name": "primary",
        "endpoint": endpoint,
        "protocol": "http",
        "weight": 100,
    }
    if "api_key_env" in cfg:
        ref["api_key_env"] = cfg["api_key_env"]
    return ref


def _emit_backend_ref_anthropic(cfg: dict) -> dict:
    """Anthropic API via vllm-sr's Anthropic adapter."""
    # Strip a trailing /v1 — the adapter handles paths internally.
    base = cfg["base_url"].rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    ref = {
        "name": "primary",
        "base_url": base,
        "provider": "anthropic",
        "weight": 100,
    }
    if "api_key_env" in cfg:
        ref["api_key_env"] = cfg["api_key_env"]
    return ref


def _emit_provider_model(tier_id: str, cfg: dict) -> dict:
    """One entry under v0.3's providers.models[]."""
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
    """Mock-mode override: every tier points at the local OAI mock.

    `mock_endpoint` is in `host:port[/path]` form (e.g. `host.docker.internal:8811/v1`).
    The router forwards via plain OAI — no Anthropic adapter — so the mock
    (stdlib OAI server) can handle all five tiers identically.
    """
    return {
        "name": tier_id,
        "provider_model_id": tier_id,  # mock echoes the model name; tier_id is fine
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
    """One entry under routing.modelCards[]. Description is the audience-
    facing tier label + role from the exemplars file."""
    desc_parts = [p for p in (tier.get("label"), tier.get("role")) if p]
    return {
        "name": tier["id"],
        "modality": "text",
        "description": " — ".join(desc_parts) if desc_parts else tier["id"],
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
    """Read the two input files, validate, emit a vllm-sr v0.3 config dict.

    `mock_endpoint` (e.g. `host.docker.internal:8811/v1`) overrides every
    backend so the router forwards to the local OAI mock. Used for pipeline
    verification before real backends come online.
    """
    ex = yaml.safe_load(exemplars_path.read_text())
    be = yaml.safe_load(backends_path.read_text())

    eval_prompts: set[str] | None = None
    if eval_set_path:
        eval_data = json.loads(eval_set_path.read_text())
        eval_prompts = {entry["prompt"] for entry in eval_data}

    _validate_exemplars(ex, eval_prompts)
    tier_ids = {t["id"] for t in ex["tiers"]}
    _validate_backends(be, tier_ids)

    # Embedding signals: two per axis (hard candidates, easy candidates).
    # The threshold can be overridden per-axis via `axis.threshold_hard` /
    # `axis.threshold_easy` in the exemplars file; otherwise use the default.
    embedding_signals: list[dict] = []
    for axis in ex["axes"]:
        threshold_hard = axis.get("threshold_hard", DEFAULT_EMBEDDING_THRESHOLD)
        threshold_easy = axis.get("threshold_easy", DEFAULT_EMBEDDING_THRESHOLD)
        embedding_signals.append(_emit_embedding_signal(
            axis["id"], "hard", axis["exemplars"]["hard"], threshold_hard,
        ))
        embedding_signals.append(_emit_embedding_signal(
            axis["id"], "easy", axis["exemplars"]["easy"], threshold_easy,
        ))

    # Decisions: priority descends in order from the exemplars file. The last
    # rule (typically a catch-all with empty `when`) ends up with the lowest
    # priority and acts as the fallthrough.
    decisions = [
        _emit_decision(rule, priority=PRIORITY_BASE - PRIORITY_STEP * i)
        for i, rule in enumerate(ex["rules"])
    ]

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
            "defaults": {"default_model": be.get("default_tier", "tier2")},
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
                "embeddings": embedding_signals,
            },
            "decisions": decisions,
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
        help="Path to data/queries.json — refuse to build if any exemplar overlaps an eval prompt",
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
