"""Tests for the per-query routing-trace diagnostic.

Uses the observed real-world vllm-sr v0.3 eval-response shape. The
parser pulls the routing decision trace out of `decision_result`:
`matched_signals.complexity[]` carries entries like `<signal>:<level>`,
`unmatched_signals.complexity[]` carries bare `<signal>` for signals
that didn't match at any level on this query, and
`matched_signals.projection[]` carries the winning band name.
"""
from __future__ import annotations

from benchmark.scores import (
    ComplexityMatch,
    EvalSnapshot,
    parse_eval_response,
)

# A trimmed real eval response captured from `vllm-sr eval --json` on a
# hard-but-not-frontier query: matched at :medium across all three signals,
# projection landed in tier1_band.
SAMPLE_RESPONSE: dict = {
    "original_text": "Here's a Go function that implements a cache ...",
    "decision_result": {
        "decision_name": "route_tier1",
        "used_signals": {"projection": ["tier1_band"]},
        "matched_signals": {
            "complexity": [
                "needs_reasoning:medium",
                "needs_expertise:medium",
                "needs_judgment:medium",
            ],
            "projection": ["tier1_band"],
        },
        "unmatched_signals": {
            "complexity": [
                "needs_reasoning",
                "needs_expertise",
                "needs_judgment",
            ],
            "projection": ["tier2_band", "tier3_band", "tier4_band", "tier5_band"],
        },
    },
}


def test_parse_eval_response_extracts_complexity_levels() -> None:
    snap = parse_eval_response(SAMPLE_RESPONSE)
    by_name = {m.name: m.level for m in snap.matches}
    assert by_name == {
        "needs_reasoning": "medium",
        "needs_expertise": "medium",
        "needs_judgment": "medium",
    }


def test_parse_eval_response_extracts_decision_and_band() -> None:
    snap = parse_eval_response(SAMPLE_RESPONSE)
    assert snap.decision == "route_tier1"
    assert snap.matched_band == "tier1_band"


def test_parse_eval_response_marks_unmatched_as_none() -> None:
    """A signal listed bare in unmatched_signals.complexity matched at no level."""
    data = {
        "decision_result": {
            "decision_name": "route_tier1",
            "matched_signals": {
                "complexity": ["needs_reasoning:hard"],
                "projection": ["tier1_band"],
            },
            "unmatched_signals": {
                "complexity": ["needs_expertise", "needs_judgment"],
            },
        },
    }
    snap = parse_eval_response(data)
    by_name = {m.name: m.level for m in snap.matches}
    assert by_name == {
        "needs_reasoning": "hard",
        "needs_expertise": "none",
        "needs_judgment": "none",
    }


def test_complexity_match_is_match_property() -> None:
    assert ComplexityMatch("x", "hard").is_match is True
    assert ComplexityMatch("x", "medium").is_match is True
    assert ComplexityMatch("x", "easy").is_match is False
    assert ComplexityMatch("x", "none").is_match is False


def test_parse_eval_response_handles_missing_decision_result() -> None:
    snap = parse_eval_response({})
    assert snap == EvalSnapshot(matches=[], decision=None, matched_band=None)


def test_parse_eval_response_handles_non_dict() -> None:
    for bad in ("oops", None, []):
        snap = parse_eval_response(bad)
        assert snap.matches == []
        assert snap.decision is None
        assert snap.matched_band is None
