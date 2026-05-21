"""Tests for the demo-data preprocessor (tools/build_demo_data.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# tools/ isn't a package; load the module by path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import build_demo_data as bdd  # noqa: E402


def _write(tmp_path: Path, routed, evals, pricing) -> tuple[Path, Path, Path]:
    rp = tmp_path / "routed.json"
    ep = tmp_path / "evals.json"
    pp = tmp_path / "pricing.json"
    rp.write_text(json.dumps(routed))
    ep.write_text(json.dumps(evals))
    pp.write_text(json.dumps(pricing))
    return rp, ep, pp


PRICING = {
    "models": {
        "tiny":     {"provider": "Local", "tier": 1, "in_per_1m": 0.0,
                     "out_per_1m": 0.0, "self_hosted": True},
        "mid":      {"provider": "OpenAI", "tier": 2, "in_per_1m": 1.0, "out_per_1m": 2.0},
        "Opus 4.7": {"provider": "Anthropic", "tier": 5, "in_per_1m": 5.0, "out_per_1m": 25.0},
    }
}


def test_cost_math() -> None:
    rate = {"in_per_1m": 1.0, "out_per_1m": 2.0}
    # 1000 prompt @ $1/M + 2000 completion @ $2/M = 0.001 + 0.004 = 0.005
    assert bdd._cost(1000, 2000, rate) == 0.005
    # No rate (self-hosted / unknown) → free.
    assert bdd._cost(1000, 2000, None) == 0.0
    assert bdd._cost(1000, 2000, {}) == 0.0


def test_build_basic_shape(tmp_path) -> None:
    routed = [
        {
            "id": "q1", "prompt": "hi", "expected_min_tier": 1, "routed_tier": 1,
            "routing_metadata": {"latency_ms": 300},
            "routed_answers": [
                {"tier": 1, "provider": "Local", "model": "tiny",
                 "answer": "yo", "status": "success",
                 "prompt_tokens": 10, "completion_tokens": 20},
            ],
            "expected_answers": [
                {"provider": "Anthropic", "model": "Opus 4.7", "answer": "hello there"},
            ],
        },
    ]
    evals = [
        {"query_id": "q1", "routed_model": "tiny", "expected_model": "Opus 4.7",
         "evaluator": "judge-A", "verdict": "Adequate", "rationale": "ok",
         "scores": {"correctness": 4, "completeness": 4, "fitness_for_purpose": 4, "soundness": 4}},
    ]
    rp, ep, pp = _write(tmp_path, routed, evals, PRICING)
    data = bdd.build(rp, ep, pp, concurrency=8)

    # Meta derived from data.
    assert [t["level"] for t in data["meta"]["tiers"]] == [1]
    assert data["meta"]["frontier_models"][0]["model"] == "Opus 4.7"
    assert data["meta"]["evaluators"] == ["judge-A"]
    assert data["meta"]["query_count"] == 1
    # Throughput = concurrency / mean latency (8 / 0.3s ≈ 26.7).
    assert data["meta"]["throughput_qps"] == round(8 / 0.3, 1)

    q = data["queries"][0]
    # Real routed tokens used; self-hosted → free.
    ra = q["routed_answers"]["tier1"][0]
    assert (ra["prompt_tokens"], ra["completion_tokens"]) == (10, 20)
    assert ra["cost_usd"] == 0.0
    # Frontier completion tokens estimated; cost computed from Opus rate.
    fa = q["frontier_answers"][0]
    assert fa["completion_tokens_estimated"] is True
    assert fa["cost_usd"] > 0.0
    # Verdict keyed by routed|frontier|evaluator.
    assert "tiny|Opus 4.7|judge-A" in q["evaluations"]
    assert q["evaluations"]["tiny|Opus 4.7|judge-A"]["verdict"] == "Adequate"


def test_build_estimates_missing_routed_tokens(tmp_path) -> None:
    """If routed answers lack token counts (older export), the
    preprocessor estimates them rather than crashing."""
    routed = [
        {
            "id": "q1", "prompt": "hi", "routed_tier": 2,
            "routing_metadata": {"latency_ms": 400},
            "routed_answers": [
                {"tier": 2, "provider": "OpenAI", "model": "mid",
                 "answer": "a longer answer here", "status": "success"},
            ],
            "expected_answers": [],
        },
    ]
    rp, ep, pp = _write(tmp_path, routed, [], PRICING)
    data = bdd.build(rp, ep, pp, concurrency=4)
    ra = data["queries"][0]["routed_answers"]["tier2"][0]
    assert ra["completion_tokens"] >= 1
    assert ra["cost_usd"] > 0.0  # mid has a non-zero rate


def test_build_handles_t5_query_with_no_routed_answers(tmp_path) -> None:
    """A query routed to the top tier has no routed_answers — the
    preprocessor keeps it (framed downstream as 'right tier')."""
    routed = [
        {
            "id": "q5", "prompt": "hard", "routed_tier": 5,
            "routing_metadata": {"latency_ms": 350},
            "routed_answers": [],
            "expected_answers": [
                {"provider": "Anthropic", "model": "Opus 4.7", "answer": "deep answer"},
            ],
        },
    ]
    rp, ep, pp = _write(tmp_path, routed, [], PRICING)
    data = bdd.build(rp, ep, pp, concurrency=8)
    q = data["queries"][0]
    assert q["routed_tier"] == 5
    assert q["routed_answers"] == {}
    assert len(q["frontier_answers"]) == 1
    assert data["meta"]["tier_route_counts"]["5"] == 1
