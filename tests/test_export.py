"""demo.json export tests (multi-model shape)."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from benchmark.db import Evaluation, GoldAnswer, Pass1Result, TierAnswer, session_scope
from benchmark.export import export_demo_json
from benchmark.runs import create_run

from ._helpers import bootstrap_db, make_models_yaml, make_router_yaml

QUERIES = [
    {
        "id": "q1", "prompt": "easy",
        "expected_answers": [
            {"answer": "Paris.", "model": "Opus", "provider": "Anthropic"},
        ],
        "expected_min_tier": 1, "specializations": ["general"],
    },
    {
        "id": "q2", "prompt": "harder",
        "expected_answers": [
            {"answer": "Long proof.", "model": "Opus", "provider": "Anthropic"},
        ],
        "expected_min_tier": 3, "specializations": ["coding"],
    },
]


def _setup(tmp_path: Path) -> tuple[Path, int]:
    db = bootstrap_db(tmp_path, QUERIES)  # also seeds upstream GoldAnswer rows
    rid = create_run(
        db,
        router_config_path=make_router_yaml(tmp_path),
        models_config_path=make_models_yaml(tmp_path),
    )
    return db, rid


def _add_pass1(db: Path, rid: int, qid: str, *, tier: int) -> None:
    with session_scope(db) as s:
        s.add(Pass1Result(
            run_id=rid, query_id=qid,
            router_selected_model=f"tier{tier}",
            router_selected_tier=tier,
            router_selected_specs=["general"],
            meets_minimum_tier=1,
            matches_specialization=1,
            latency_ms=20,
            raw_routing_metadata={"category": "general"},
            status="success",
            attempted_at=datetime.now(UTC),
        ))


def _add_tier_answer(
    db: Path, rid: int, qid: str, tier_level: int, model_id: str, response,
    *, provider: str | None = None, slot: int = 0, status: str = "success",
) -> None:
    with session_scope(db) as s:
        s.add(TierAnswer(
            run_id=rid, query_id=qid,
            tier_level=tier_level, model_id=model_id, model_slot=slot,
            provider=provider, tier_name=f"tier{tier_level}",
            response_text=response,
            prompt_tokens=5, completion_tokens=10, latency_ms=10,
            status=status,
            attempted_at=datetime.now(UTC),
        ))


def test_export_basic_multimodel_shape(tmp_path: Path) -> None:
    db, rid = _setup(tmp_path)
    _add_pass1(db, rid, "q1", tier=3)
    # Two models in the routed tier (3).
    _add_tier_answer(db, rid, "q1", 3, "gpt-5-mini", "Paris (OpenAI)",
                     provider="OpenAI", slot=0)
    _add_tier_answer(db, rid, "q1", 3, "gemini-flash", "Paris (Google)",
                     provider="Google", slot=1)

    out = tmp_path / "demo.json"
    summary = export_demo_json(db, rid, out)
    assert summary.queries_exported == 2

    q1 = next(e for e in json.loads(out.read_text()) if e["id"] == "q1")
    assert q1["routed_tier"] == 3
    assert q1["routing_metadata"]["selected_model"] == "tier3"
    # Per-query routing time is surfaced (set by _add_pass1).
    assert q1["routing_metadata"]["latency_ms"] == 20

    # Expected answers: the single gold declared in queries.json.
    assert q1["expected_answers"] == [
        {"provider": "Anthropic", "model": "Opus", "answer": "Paris."},
    ]

    # Both routed-tier models present, ordered by slot.
    assert q1["routed_answers"] == [
        {"tier": 3, "provider": "OpenAI", "model": "gpt-5-mini",
         "answer": "Paris (OpenAI)", "status": "success", "latency_ms": 10},
        {"tier": 3, "provider": "Google", "model": "gemini-flash",
         "answer": "Paris (Google)", "status": "success", "latency_ms": 10},
    ]

    # `all_tier_answers` no longer emitted — `make answers` only calls
    # the routed tier, so it was always a subset of `routed_answers`.


def test_export_missing_routing_is_null(tmp_path: Path) -> None:
    db, rid = _setup(tmp_path)
    _add_tier_answer(db, rid, "q1", 2, "m2", "ans", provider="X")

    out = tmp_path / "demo.json"
    export_demo_json(db, rid, out)
    q1 = next(e for e in json.loads(out.read_text()) if e["id"] == "q1")

    assert q1["routed_tier"] is None
    assert q1["routing_metadata"] is None
    assert q1["routed_answers"] == []
    # Declared gold still present even with no routing.
    assert q1["expected_answers"][0]["answer"] == "Paris."


def test_export_routed_tier_with_error_model(tmp_path: Path) -> None:
    """A routed-tier model that errored appears in routed_answers with
    `status='error'` and `answer=null` so downstream consumers can see
    the failure rather than silently missing it."""
    db, rid = _setup(tmp_path)
    _add_pass1(db, rid, "q1", tier=3)
    _add_tier_answer(db, rid, "q1", 3, "ok-model", "good", provider="A", slot=0)
    _add_tier_answer(db, rid, "q1", 3, "bad-model", None,
                     provider="B", slot=1, status="error")

    out = tmp_path / "demo.json"
    export_demo_json(db, rid, out)
    q1 = next(e for e in json.loads(out.read_text()) if e["id"] == "q1")

    statuses = {r["model"]: r["status"] for r in q1["routed_answers"]}
    assert statuses == {"ok-model": "success", "bad-model": "error"}
    # The errored model's row has status='error' and answer=None.
    bad = next(r for r in q1["routed_answers"] if r["model"] == "bad-model")
    assert bad["answer"] is None
    ok = next(r for r in q1["routed_answers"] if r["model"] == "ok-model")
    assert ok["answer"] == "good"


def test_export_per_provider_expected_answers(tmp_path: Path) -> None:
    db, rid = _setup(tmp_path)
    # Add a second provider's gold alongside the file-declared "Opus" one
    # (e.g. as would be added by `make update-gold` or import-answers).
    with session_scope(db) as s:
        s.add(GoldAnswer(
            query_id="q1", model_id="gpt-5", provider="OpenAI",
            answer="Paris, the capital of France.",
            generated_at=datetime.now(UTC),
        ))

    out = tmp_path / "demo.json"
    export_demo_json(db, rid, out)
    q1 = next(e for e in json.loads(out.read_text()) if e["id"] == "q1")
    # Sorted alphabetically by model_id: "Opus" (capital O = 79 in ASCII)
    # comes before "gpt-5" (lowercase g = 103).
    assert q1["expected_answers"] == [
        {"provider": "Anthropic", "model": "Opus", "answer": "Paris."},
        {"provider": "OpenAI",    "model": "gpt-5",
         "answer": "Paris, the capital of France."},
    ]


def test_export_no_gold_means_empty(tmp_path: Path) -> None:
    db, rid = _setup(tmp_path)
    with session_scope(db) as s:
        s.execute(
            select(GoldAnswer).where(GoldAnswer.query_id == "q1")
        ).scalar_one()  # exists
        for g in s.execute(
            select(GoldAnswer).where(GoldAnswer.query_id == "q1")
        ).scalars().all():
            s.delete(g)

    out = tmp_path / "demo.json"
    export_demo_json(db, rid, out)
    q1 = next(e for e in json.loads(out.read_text()) if e["id"] == "q1")
    assert q1["expected_answers"] == []


def test_export_summary_counts(tmp_path: Path) -> None:
    db, rid = _setup(tmp_path)
    _add_pass1(db, rid, "q1", tier=1)
    _add_tier_answer(db, rid, "q1", 1, "m1a", "a", slot=0)
    _add_tier_answer(db, rid, "q1", 1, "m1b", "b", slot=1)

    out = tmp_path / "demo.json"
    summary = export_demo_json(db, rid, out)
    assert summary.queries_exported == 2
    assert summary.with_routed_tier == 1          # only q1
    assert summary.with_routed_answer == 1        # q1's tier-1 models answered
    assert summary.with_expected == 2             # both have upstream gold
    # Per-tier-per-model counts: 1 successful answer each for the two
    # tier-1 models that ran on q1. q2 had no routed answers.
    assert summary.routed_answers_per_model == {
        (1, "m1a"): 1,
        (1, "m1b"): 1,
    }
    # q2 has no pass1 row (not routed) — doesn't count as top-tier.
    assert summary.top_tier_routed == 0


# ─────────────────────────────────────────────────────────────────────
# Sibling evaluations.json export
# ─────────────────────────────────────────────────────────────────────

def _add_evaluation(
    db: Path, rid: int, qid: str, *, tier: int, routed: str, gold: str,
    evaluator: str, verdict: str = "Adequate",
    correctness: int = 4, completeness: int = 4, fitness: int = 4,
    soundness: int = 4,
    routed_provider: str | None = None, gold_provider: str | None = None,
) -> None:
    with session_scope(db) as s:
        s.add(Evaluation(
            run_id=rid, query_id=qid,
            routed_tier=tier, routed_model=routed,
            gold_model_id=gold, evaluator=evaluator,
            routed_provider=routed_provider, gold_provider=gold_provider,
            verdict=verdict, rationale="ok",
            correctness=correctness, completeness=completeness,
            fitness_for_purpose=fitness, soundness=soundness,
            status="success", latency_ms=10,
            evaluated_at=datetime.now(UTC),
        ))


def test_export_emits_evaluations_when_rows_exist(tmp_path: Path) -> None:
    db, rid = _setup(tmp_path)
    _add_pass1(db, rid, "q1", tier=3)
    _add_tier_answer(db, rid, "q1", 3, "gpt-5-mini", "Paris (OpenAI)", provider="OpenAI")
    _add_evaluation(
        db, rid, "q1", tier=3, routed="gpt-5-mini", gold="Opus",
        evaluator="claude-sonnet-4-6", verdict="Adequate",
        routed_provider="OpenAI", gold_provider="Anthropic",
    )
    out = tmp_path / "demo.json"
    summary = export_demo_json(db, rid, out)

    assert summary.evaluations_path == tmp_path / "evaluations.json"
    assert summary.evaluations_written == 1
    assert summary.evaluations_by_evaluator == {"claude-sonnet-4-6": 1}

    evs = json.loads((tmp_path / "evaluations.json").read_text())
    assert len(evs) == 1
    e = evs[0]
    # Evaluator suffix in eval_id (the new format).
    assert "--claude-sonnet-4-6" in e["eval_id"]
    # The expected/routed labels mirror the input.
    assert e["query_id"] == "q1"
    assert e["routed_model"] == "gpt-5-mini"
    assert e["expected_model"] == "Opus"
    assert e["evaluator"] == "claude-sonnet-4-6"
    assert e["verdict"] == "Adequate"
    # Four-dimension scores.
    assert set(e["scores"].keys()) == {
        "correctness", "completeness", "fitness_for_purpose", "soundness",
    }


def test_export_skips_evaluations_when_no_rows(tmp_path: Path) -> None:
    """If no Evaluation rows exist for the run, no file is written
    (clean signal — downstream readers see absence vs. an empty list)."""
    db, rid = _setup(tmp_path)
    _add_pass1(db, rid, "q1", tier=1)
    _add_tier_answer(db, rid, "q1", 1, "m1", "ans")
    out = tmp_path / "demo.json"
    summary = export_demo_json(db, rid, out)

    assert summary.evaluations_path is None
    assert summary.evaluations_written == 0
    assert not (tmp_path / "evaluations.json").exists()


def test_export_evaluations_ordered_stably(tmp_path: Path) -> None:
    """The sibling export sorts evaluation entries by (query, tier,
    routed_model, gold_model, evaluator) so reruns produce byte-stable
    files — important when the export is committed."""
    db, rid = _setup(tmp_path)
    _add_pass1(db, rid, "q1", tier=2)
    _add_tier_answer(db, rid, "q1", 2, "m1", "a", slot=0)
    _add_evaluation(db, rid, "q1", tier=2, routed="m1", gold="Opus",
                     evaluator="judge-Z")
    _add_evaluation(db, rid, "q1", tier=2, routed="m1", gold="Opus",
                     evaluator="judge-A")
    out = tmp_path / "demo.json"
    export_demo_json(db, rid, out)
    evs = json.loads((tmp_path / "evaluations.json").read_text())
    # Sort key has evaluator last → judge-A before judge-Z.
    assert [e["evaluator"] for e in evs] == ["judge-A", "judge-Z"]
