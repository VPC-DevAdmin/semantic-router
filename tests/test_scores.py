"""Tests for the per-signal scores diagnostic.

Uses the observed real-world vllm-sr v0.3 eval response shape — see the
parser docstring in `src/benchmark/scores.py` for the full schema.
"""
from __future__ import annotations

from benchmark.scores import (
    EASY_THRESHOLD,
    HARD_THRESHOLD,
    SignalScore,
    parse_eval_response,
)

# A trimmed version of the real eval response for "What is inflation?".
SAMPLE_RESPONSE: dict = {
    "original_text": "What is inflation?",
    "decision_result": {
        "decision_name": "light_to_t2",
    },
    "matched_signals": {"embeddings": ["expertise_easy"]},
    "unmatched_signals": {
        "embeddings": [
            "reasoning_hard", "reasoning_easy",
            "expertise_hard",
            "judgment_hard", "judgment_easy",
        ],
    },
    "signal_values": {
        # Main + sub-keys for each signal — parser keeps only the main.
        "embedding:reasoning_hard": 0.436,
        "embedding:reasoning_hard:best": 0.436,
        "embedding:reasoning_hard:support": 0.435,
        "embedding:reasoning_hard:prototype_count": 8,
        "embedding:reasoning_easy": 0.573,
        "embedding:reasoning_easy:best": 0.584,
        "embedding:reasoning_easy:support": 0.543,
        "embedding:reasoning_easy:prototype_count": 8,
        "embedding:expertise_hard": 0.432,
        "embedding:expertise_hard:best": 0.434,
        "embedding:expertise_hard:support": 0.427,
        "embedding:expertise_hard:prototype_count": 8,
        "embedding:expertise_easy": 0.576,
        "embedding:expertise_easy:best": 0.581,
        "embedding:expertise_easy:support": 0.564,
        "embedding:expertise_easy:prototype_count": 8,
        "embedding:judgment_hard": 0.476,
        "embedding:judgment_hard:best": 0.479,
        "embedding:judgment_hard:support": 0.470,
        "embedding:judgment_hard:prototype_count": 8,
        "embedding:judgment_easy": 0.544,
        "embedding:judgment_easy:best": 0.554,
        "embedding:judgment_easy:support": 0.516,
        "embedding:judgment_easy:prototype_count": 8,
    },
}


def test_signal_score_gap_and_above_threshold() -> None:
    s = SignalScore(
        name="reasoning_hard", score=0.43, threshold=0.42, matched_by_router=False
    )
    assert abs(s.gap - 0.01) < 1e-6
    assert s.above_threshold is True

    s2 = SignalScore(
        name="judgment_hard", score=0.39, threshold=0.42, matched_by_router=False
    )
    assert s2.gap < 0
    assert s2.above_threshold is False


def test_parse_eval_response_extracts_main_scores_only() -> None:
    """Sub-keys (`:best`, `:support`, `:prototype_count`) are not signals."""
    signals = parse_eval_response(SAMPLE_RESPONSE)
    names = {s.name for s in signals}
    assert names == {
        "reasoning_hard", "reasoning_easy",
        "expertise_hard", "expertise_easy",
        "judgment_hard", "judgment_easy",
    }


def test_parse_eval_response_assigns_correct_thresholds() -> None:
    signals = {s.name: s for s in parse_eval_response(SAMPLE_RESPONSE)}
    assert signals["reasoning_hard"].threshold == HARD_THRESHOLD
    assert signals["expertise_hard"].threshold == HARD_THRESHOLD
    assert signals["judgment_hard"].threshold == HARD_THRESHOLD
    assert signals["reasoning_easy"].threshold == EASY_THRESHOLD
    assert signals["expertise_easy"].threshold == EASY_THRESHOLD
    assert signals["judgment_easy"].threshold == EASY_THRESHOLD


def test_parse_eval_response_marks_router_matched() -> None:
    """Only expertise_easy is in matched_signals.embeddings in the sample."""
    signals = {s.name: s for s in parse_eval_response(SAMPLE_RESPONSE)}
    assert signals["expertise_easy"].matched_by_router is True
    for name in (
        "reasoning_hard", "reasoning_easy",
        "expertise_hard",
        "judgment_hard", "judgment_easy",
    ):
        assert signals[name].matched_by_router is False, (
            f"{name} should not be in router's matched_signals"
        )


def test_parse_eval_response_captures_above_threshold_discrepancy() -> None:
    """The critical finding: signals can be above threshold but NOT matched
    by the router. The parser must surface both facts so the report can
    diagnose the discrepancy."""
    signals = {s.name: s for s in parse_eval_response(SAMPLE_RESPONSE)}
    # All hard signals above OUR threshold (0.42):
    for name in ("reasoning_hard", "expertise_hard", "judgment_hard"):
        assert signals[name].above_threshold is True
    # ...but none are in router's matched_signals.
    for name in ("reasoning_hard", "expertise_hard", "judgment_hard"):
        assert signals[name].matched_by_router is False


def test_parse_eval_response_handles_missing_signal_values() -> None:
    assert parse_eval_response({"decision_result": {}}) == []


def test_parse_eval_response_handles_non_dict() -> None:
    assert parse_eval_response("oops") == []
    assert parse_eval_response(None) == []
    assert parse_eval_response([]) == []
