"""Tests for the per-signal scores diagnostic.

Light coverage — this is a one-API-call-per-misroute diagnostic that
calls vllm-sr's /api/v1/eval endpoint. We don't have a real router in
CI, so we just verify the response parser handles a few plausible shapes
and the SignalScore gap calculation.
"""
from __future__ import annotations

from benchmark.scores import SignalScore, parse_eval_response


def test_signal_score_gap_positive_when_matched() -> None:
    s = SignalScore(name="reasoning_hard", score=0.51, threshold=0.42, matched=True)
    assert s.gap > 0
    assert abs(s.gap - 0.09) < 1e-6


def test_signal_score_gap_negative_when_missed() -> None:
    s = SignalScore(name="reasoning_hard", score=0.39, threshold=0.42, matched=False)
    assert s.gap < 0


def test_parse_eval_response_shape_with_name_score_threshold() -> None:
    """The 'obvious' shape — list of {name, score, threshold, matched}."""
    data = {
        "decision": "hard_technical_to_t3",
        "signal_confidences": [
            {"name": "reasoning_hard", "score": 0.41, "threshold": 0.42, "matched": False},
            {"name": "expertise_hard", "score": 0.55, "threshold": 0.42, "matched": True},
            {"name": "judgment_hard", "score": 0.30, "threshold": 0.42, "matched": False},
        ],
    }
    signals = parse_eval_response(data)
    names = {s.name for s in signals}
    assert names == {"reasoning_hard", "expertise_hard", "judgment_hard"}
    matched = {s.name: s.matched for s in signals}
    assert matched["expertise_hard"] is True
    assert matched["reasoning_hard"] is False


def test_parse_eval_response_alternative_keys() -> None:
    """Be tolerant to `confidence` / `value` instead of `score`."""
    data = {
        "signals": [
            {"signal": "reasoning_hard", "confidence": 0.45, "threshold": 0.42},
            {"name": "expertise_hard", "value": 0.30, "threshold": 0.42, "matched": False},
        ],
    }
    signals = parse_eval_response(data)
    assert len(signals) == 2
    # Auto-derived matched: signal at 0.45 > 0.42 → matched True.
    by_name = {s.name: s for s in signals}
    assert by_name["reasoning_hard"].matched is True
    assert by_name["reasoning_hard"].score == 0.45
    assert by_name["expertise_hard"].matched is False


def test_parse_eval_response_dedups_repeated_names() -> None:
    """If a signal appears in both `matched` and `used` lists, return once."""
    data = {
        "signals": {
            "matched": [
                {"name": "expertise_hard", "score": 0.55, "threshold": 0.42, "matched": True},
            ],
            "used": [
                {"name": "expertise_hard", "score": 0.55, "threshold": 0.42, "matched": True},
                {"name": "reasoning_hard", "score": 0.40, "threshold": 0.42, "matched": False},
            ],
        },
    }
    signals = parse_eval_response(data)
    assert len(signals) == 2
    assert {s.name for s in signals} == {"expertise_hard", "reasoning_hard"}


def test_parse_eval_response_empty_when_unknown_shape() -> None:
    """Unknown shape → empty list (the CLI will fall back to printing raw)."""
    data = {"foo": "bar", "baz": [1, 2, 3]}
    assert parse_eval_response(data) == []


def test_parse_eval_response_non_dict() -> None:
    """Defensive: handle a non-dict response gracefully."""
    assert parse_eval_response("oops") == []
    assert parse_eval_response(None) == []
    assert parse_eval_response([]) == []
