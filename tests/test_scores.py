"""Tests for the per-signal scores diagnostic.

Uses the v0.3 projections-design eval response shape — three complexity
signals (`needs_reasoning`, `needs_expertise`, `needs_judgment`) each with
a `threshold: 0.55`, optionally plus the projected `request_difficulty`
score and matched `tier_band`. See the parser docstring in
`src/benchmark/scores.py` for the full schema.
"""
from __future__ import annotations

from benchmark.scores import (
    SIGNAL_THRESHOLD,
    EvalSnapshot,
    SignalScore,
    parse_eval_response,
)

# A trimmed eval response for "What is inflation?" under the projections design.
SAMPLE_RESPONSE: dict = {
    "original_text": "What is inflation?",
    "decision_result": {
        "decision_name": "route_tier2",
    },
    "signal_values": {
        # Main + sub-keys for each signal — parser keeps only the main.
        "embedding:needs_reasoning": 0.436,
        "embedding:needs_reasoning:best": 0.436,
        "embedding:needs_reasoning:support": 0.435,
        "embedding:needs_reasoning:prototype_count": 8,
        "embedding:needs_expertise": 0.576,
        "embedding:needs_expertise:best": 0.581,
        "embedding:needs_expertise:support": 0.564,
        "embedding:needs_expertise:prototype_count": 8,
        "embedding:needs_judgment": 0.476,
        "embedding:needs_judgment:best": 0.479,
        "embedding:needs_judgment:support": 0.470,
        "embedding:needs_judgment:prototype_count": 8,
        # Projected outputs (may or may not be present in real responses).
        "projection:request_difficulty": 0.494,
        "mapping:tier_band": "tier3_band",
    },
}


def test_signal_score_gap_and_above_threshold() -> None:
    s = SignalScore(name="needs_reasoning", score=0.60, threshold=0.55)
    assert abs(s.gap - 0.05) < 1e-6
    assert s.above_threshold is True

    s2 = SignalScore(name="needs_judgment", score=0.40, threshold=0.55)
    assert s2.gap < 0
    assert s2.above_threshold is False


def test_parse_eval_response_extracts_main_scores_only() -> None:
    """Sub-keys (`:best`, `:support`, `:prototype_count`) are not signals."""
    snap = parse_eval_response(SAMPLE_RESPONSE)
    names = {s.name for s in snap.signals}
    assert names == {"needs_reasoning", "needs_expertise", "needs_judgment"}


def test_parse_eval_response_assigns_single_threshold() -> None:
    """Under projections, every complexity signal shares one threshold."""
    snap = parse_eval_response(SAMPLE_RESPONSE)
    for s in snap.signals:
        assert s.threshold == SIGNAL_THRESHOLD


def test_parse_eval_response_surfaces_projection_outputs() -> None:
    """When the response includes the projected score and matched band,
    the parser must surface them so the report can show what actually
    drove the routing decision."""
    snap = parse_eval_response(SAMPLE_RESPONSE)
    assert snap.request_difficulty is not None
    assert abs(snap.request_difficulty - 0.494) < 1e-6
    assert snap.tier_band == "tier3_band"


def test_parse_eval_response_tolerates_missing_projection_outputs() -> None:
    """Older vllm-sr builds may not echo back the projection outputs.
    The parser must still return signal scores in that case."""
    data = {
        "signal_values": {
            "embedding:needs_reasoning": 0.30,
            "embedding:needs_expertise": 0.40,
            "embedding:needs_judgment": 0.50,
        },
    }
    snap = parse_eval_response(data)
    assert len(snap.signals) == 3
    assert snap.request_difficulty is None
    assert snap.tier_band is None


def test_parse_eval_response_handles_missing_signal_values() -> None:
    snap = parse_eval_response({"decision_result": {}})
    assert snap == EvalSnapshot(signals=[], request_difficulty=None, tier_band=None)


def test_parse_eval_response_handles_non_dict() -> None:
    for bad in ("oops", None, []):
        snap = parse_eval_response(bad)
        assert snap.signals == []
        assert snap.request_difficulty is None
        assert snap.tier_band is None
