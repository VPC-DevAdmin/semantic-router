#!/usr/bin/env python3
"""Build the self-contained dataset the cost-routing demo consumes.

Reads the canonical exports + a pricing config and emits ONE
`demo/data/demo_data.json` that the front-end fetches. Everything the
demo needs to be data-driven — tier structure, model lists, frontier
options, pricing, throughput, per-query answers + costs + judge
verdicts — lives in that file, so dropping in a fresh dataset is a
re-run of this script, no code change.

Inputs:
  data/routed_queries_with_answers.json   (routed answers + tokens + latency)
  data/evaluations.json                    (judge verdicts, 2 evaluators)
  demo/pricing.json                         (per-1M-token USD rates)

Output:
  demo/data/demo_data.json

Token accounting:
  • Routed answers carry REAL token counts (from each model's API
    response, surfaced by `make export`).
  • Frontier/gold answers were never stored with token counts, so we
    estimate their completion tokens here — tiktoken cl100k_base if
    available, else a chars/4 heuristic. The prompt token count is
    shared across both sides (same prompt), taken from the routed
    answer when present.

Usage:
  .venv/bin/python tools/build_demo_data.py
  .venv/bin/python tools/build_demo_data.py --concurrency 8
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROUTED = ROOT / "data" / "routed_queries_with_answers.json"
DEFAULT_EVALS = ROOT / "data" / "evaluations.json"
DEFAULT_PRICING = ROOT / "demo" / "pricing.json"
DEFAULT_OUT = ROOT / "demo" / "data" / "demo_data.json"


# ─────────────────────────────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────────────────────────────

def _make_token_counter():
    """Return (count_fn, method_label). Prefer tiktoken; fall back to a
    char-based heuristic so the tool runs with no extra deps."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return (lambda s: len(enc.encode(s or "")), "tiktoken/cl100k_base")
    except Exception:
        # chars/4 is the conventional rough token estimate for English.
        return (lambda s: max(1, round(len(s or "") / 4)), "chars/4 heuristic")


# ─────────────────────────────────────────────────────────────────────
# Cost
# ─────────────────────────────────────────────────────────────────────

def _cost(prompt_tokens: int, completion_tokens: int, rate: dict | None) -> float:
    """USD cost for one call given a {in_per_1m, out_per_1m} rate."""
    if not rate:
        return 0.0
    return (
        prompt_tokens * rate.get("in_per_1m", 0.0) / 1_000_000
        + completion_tokens * rate.get("out_per_1m", 0.0) / 1_000_000
    )


# ─────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────

def build(
    routed_path: Path,
    evals_path: Path,
    pricing_path: Path,
    *,
    concurrency: int,
) -> dict[str, Any]:
    routed = json.loads(routed_path.read_text())
    evals = json.loads(evals_path.read_text())
    pricing_doc = json.loads(pricing_path.read_text())
    pricing = pricing_doc["models"]

    count_tokens, token_method = _make_token_counter()

    # ── Index evaluations by (query, routed_model, frontier_model, evaluator) ──
    # The front-end re-keys verdicts live when the tier/frontier pickers
    # change, so we hand it a flat lookup table per query.
    evals_by_query: dict[str, dict[str, dict]] = {}
    evaluators: set[str] = set()
    for e in evals:
        evaluators.add(e["evaluator"])
        key = f"{e['routed_model']}|{e['expected_model']}|{e['evaluator']}"
        evals_by_query.setdefault(e["query_id"], {})[key] = {
            "verdict": e["verdict"],
            "rationale": e["rationale"],
            "scores": e["scores"],
        }

    # ── Tier structure + frontier models, derived from the data ──
    tier_models: dict[int, list[dict]] = {}
    frontier_models: dict[str, dict] = {}
    routing_latencies: list[float] = []
    tier_route_counts: dict[int, int] = {}

    queries_out: list[dict] = []
    for q in routed:
        rt = q.get("routed_tier")
        rm = q.get("routing_metadata") or {}
        lat = rm.get("latency_ms")
        if lat is not None:
            routing_latencies.append(lat)
        if rt is not None:
            tier_route_counts[rt] = tier_route_counts.get(rt, 0) + 1

        # Prompt token count (shared both sides). Prefer a real routed
        # answer's prompt_tokens; else tokenize the prompt text.
        prompt_tokens = None
        for ra in q.get("routed_answers", []):
            if ra.get("prompt_tokens"):
                prompt_tokens = ra["prompt_tokens"]
                break
        if prompt_tokens is None:
            prompt_tokens = count_tokens(q.get("prompt", ""))

        # Routed answers, grouped by tier, with real tokens + computed cost.
        routed_by_tier: dict[str, list[dict]] = {}
        for ra in q.get("routed_answers", []):
            model = ra["model"]
            tier_models.setdefault(ra["tier"], [])
            if not any(m["model"] == model for m in tier_models[ra["tier"]]):
                tier_models[ra["tier"]].append(
                    {"provider": ra.get("provider"), "model": model}
                )
            comp = ra.get("completion_tokens")
            if comp is None:
                comp = count_tokens(ra.get("answer") or "")
            p_tok = ra.get("prompt_tokens") or prompt_tokens
            routed_by_tier.setdefault(f"tier{ra['tier']}", []).append({
                "provider": ra.get("provider"),
                "model": model,
                "answer": ra.get("answer"),
                "status": ra.get("status"),
                "prompt_tokens": p_tok,
                "completion_tokens": comp,
                "cost_usd": _cost(p_tok, comp, pricing.get(model)),
            })

        # Frontier answers — completion tokens estimated (not stored).
        frontier_out: list[dict] = []
        for fa in q.get("expected_answers", []):
            model = fa["model"]
            if model not in frontier_models:
                frontier_models[model] = {"provider": fa.get("provider"), "model": model}
            comp = count_tokens(fa.get("answer") or "")
            frontier_out.append({
                "provider": fa.get("provider"),
                "model": model,
                "answer": fa.get("answer"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": comp,
                "completion_tokens_estimated": True,
                "cost_usd": _cost(prompt_tokens, comp, pricing.get(model)),
            })

        queries_out.append({
            "id": q["id"],
            "prompt": q.get("prompt"),
            "expected_min_tier": q.get("expected_min_tier"),
            "routed_tier": rt,
            "routing_latency_ms": lat,
            "prompt_tokens": prompt_tokens,
            "routed_answers": routed_by_tier,
            "frontier_answers": frontier_out,
            "evaluations": evals_by_query.get(q["id"], {}),
        })

    # Stable model ordering within tiers (by provider then model).
    for t in tier_models:
        tier_models[t].sort(key=lambda m: (m["provider"] or "", m["model"]))

    tiers_meta = [
        {"level": lvl, "models": tier_models[lvl]}
        for lvl in sorted(tier_models)
    ]

    # Throughput: concurrency / mean per-query routing latency. This is a
    # floor (the route pass went through the local mock backend, so the
    # latency includes that round-trip, not just the classifier).
    mean_lat_s = (statistics.mean(routing_latencies) / 1000) if routing_latencies else 0
    throughput_qps = round(concurrency / mean_lat_s, 1) if mean_lat_s else 0.0

    def _pct(p: float) -> float:
        if not routing_latencies:
            return 0.0
        s = sorted(routing_latencies)
        return s[min(len(s) - 1, int(len(s) * p))]

    return {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(),
            "token_method": token_method,
            "tiers": tiers_meta,
            "frontier_models": sorted(frontier_models.values(),
                                      key=lambda m: (m["provider"] or "", m["model"])),
            "evaluators": sorted(evaluators),
            "concurrency": concurrency,
            "throughput_qps": throughput_qps,
            "routing_latency_ms": {
                "p50": _pct(0.50), "p90": _pct(0.90), "p99": _pct(0.99),
                "mean": round(statistics.mean(routing_latencies), 1) if routing_latencies else 0,
            },
            "tier_route_counts": {str(k): v for k, v in sorted(tier_route_counts.items())},
            "query_count": len(queries_out),
        },
        "pricing": pricing,
        "queries": queries_out,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routed", type=Path, default=DEFAULT_ROUTED)
    p.add_argument("--evals", type=Path, default=DEFAULT_EVALS)
    p.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--concurrency", type=int, default=8,
        help="Concurrency the route pass ran at (for the throughput stat).",
    )
    args = p.parse_args()

    for path in (args.routed, args.evals, args.pricing):
        if not path.exists():
            print(f"error: {path} does not exist", file=__import__("sys").stderr)
            return 2

    data = build(args.routed, args.evals, args.pricing, concurrency=args.concurrency)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))

    m = data["meta"]
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out}  ({size_mb:.1f} MB)")
    print(f"  queries:       {m['query_count']}")
    print(f"  tiers:         {[t['level'] for t in m['tiers']]}")
    print(f"  models/tier:   {[len(t['models']) for t in m['tiers']]}")
    print(f"  frontier:      {[fm['model'] for fm in m['frontier_models']]}")
    print(f"  evaluators:    {m['evaluators']}")
    print(f"  token method:  {m['token_method']}")
    print(f"  throughput:    {m['throughput_qps']} qps "
          f"(concurrency {m['concurrency']}, mean route {m['routing_latency_ms']['mean']} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
