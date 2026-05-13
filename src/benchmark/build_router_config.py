#!/usr/bin/env python3
"""
Build a vllm-sr config from the public training data + backend infra config.

Inputs:
  router-exemplars.yaml   — public training-set artifact (exemplars + rules)
  router-backends.yaml    — infrastructure (endpoints, ports, auth)

Output:
  router-config.yaml      — vllm-sr v0.3 config, ready to load

The split exists so the exemplars file stays a clean, human-readable
artifact you can show on screen during the demo. Code reads both,
validates, and emits whatever vllm-sr actually wants.

Schema note: vllm-sr's YAML schema is still firming up. If a field name
changes between releases, update _emit_signal() / _emit_rule() / _emit_backend()
below. The public exemplars/backends files should stay stable.

Usage:
  python build_router_config.py \\
      --exemplars router-exemplars.yaml \\
      --backends router-backends.yaml \\
      --out router-config.yaml

  # Then start vllm-sr against the output:
  vllm-sr serve --config router-config.yaml
"""

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")


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
                    "5+ recommended for stable scoring",
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
# Emit — translate public format → vllm-sr v0.3 format
# ─────────────────────────────────────────────────────────────────────────
# These three functions are the only place that touches vllm-sr's actual
# field names. When the schema shifts, change here.

def _emit_backend(tier_id: str, cfg: dict) -> dict:
    """Translate one backend entry to vllm-sr's provider shape."""
    out = {
        "name": tier_id,
        "base_url": cfg["base_url"],
        "model": cfg["model"],
    }
    if "api_key_env" in cfg:
        # vllm-sr reads this env var at startup. Verify field name against
        # your vllm-sr version — some versions use `auth.api_key_env`.
        out["api_key_env"] = cfg["api_key_env"]
    return out


def _emit_signal(axis: dict) -> dict:
    """Translate one axis to a vllm-sr contrastive_embedding signal."""
    return {
        "name": f"complexity_{axis['id']}",
        "type": "contrastive_embedding",
        "description": axis["description"],
        "score_bands": {
            band: [vals["min"], vals["max"]]
            for band, vals in axis["score_bands"].items()
        },
        "exemplars": {
            "hard": axis["exemplars"]["hard"],
            "easy": axis["exemplars"]["easy"],
        },
    }


def _emit_rule(rule: dict, priority: int) -> dict:
    """Translate one decision rule to vllm-sr's decision_rules shape.

    The public format uses `when: {axis: [bands...]}` plus an optional
    `requires_any_high: [axis...]`. We expand to vllm-sr's Boolean
    composition form.
    """
    conditions: list[dict] = []
    for axis_id, allowed_bands in rule.get("when", {}).items():
        if allowed_bands:  # non-empty list
            conditions.append({f"complexity_{axis_id}": {"in": allowed_bands}})

    out: dict = {
        "name": rule["name"],
        "priority": priority,
        "route_to": rule["route_to"],
    }
    if rule.get("description"):
        out["description"] = rule["description"]

    if not conditions and "requires_any_high" not in rule:
        # Fallthrough rule
        out["when"] = {"always": True}
        return out

    when: dict = {"all": conditions} if conditions else {}

    if "requires_any_high" in rule:
        when.setdefault("all", []).append({
            "any": [
                {f"complexity_{axis}": {"in": ["high"]}}
                for axis in rule["requires_any_high"]
            ]
        })

    out["when"] = when
    return out


# ─────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────

def build(exemplars_path: Path, backends_path: Path, eval_set_path: Path | None) -> dict:
    ex = yaml.safe_load(exemplars_path.read_text())
    be = yaml.safe_load(backends_path.read_text())

    eval_prompts: set[str] | None = None
    if eval_set_path:
        import json
        eval_data = json.loads(eval_set_path.read_text())
        eval_prompts = {entry["prompt"] for entry in eval_data}

    _validate_exemplars(ex, eval_prompts)
    tier_ids = {t["id"] for t in ex["tiers"]}
    _validate_backends(be, tier_ids)

    # Build the vllm-sr config
    config = {
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
            "endpoints": [
                _emit_backend(tier_id, cfg)
                for tier_id, cfg in be["backends"].items()
            ],
        },
        "signals": {
            f"complexity_{axis['id']}": _emit_signal(axis)
            for axis in ex["axes"]
        },
        # Built-in MMLU domain signal — included for the signal trace
        # (audience can see domain classification), but not consumed by
        # any rule. Add to rules later if needed.
        "signals_builtin": {
            "domain": {"type": "builtin_mmlu_classifier"},
        },
        "decision_rules": [
            _emit_rule(rule, priority=100 - 10 * i)
            for i, rule in enumerate(ex["rules"])
        ],
        "observability": be.get("observability", {}),
        "dashboard": be.get("dashboard", {}),
    }
    return config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exemplars", type=Path, default=Path("router-exemplars.yaml"))
    p.add_argument("--backends", type=Path, default=Path("router-backends.yaml"))
    p.add_argument("--out", type=Path, default=Path("router-config.yaml"))
    p.add_argument(
        "--check-against-eval",
        type=Path,
        help="Path to all_queries.json — refuse to build if any exemplar overlaps an eval prompt",
    )
    args = p.parse_args()

    config = build(args.exemplars, args.backends, args.check_against_eval)
    args.out.write_text(yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
