"""demo.json export tests."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from benchmark.db import Pass1Result, Query, TierAnswer, session_scope
from benchmark.export import export_demo_json
from benchmark.runs import create_run

from ._helpers import bootstrap_db, make_models_yaml, make_router_yaml

QUERIES = [
    {
        "id": "q1", "prompt": "easy",
        "expected_answer": "Paris.",
        "expected_min_tier": 1, "specializations": ["general"],
    },
    {
        "id": "q2", "prompt": "harder",
        "expected_answer": "Long proof.",
        "expected_min_tier": 3, "specializations": ["coding"],
    },
]


def _setup(tmp_path: Path) -> tuple[Path, int]:
    db = bootstrap_db(tmp_path, QUERIES)
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
    db: Path, rid: int, qid: str, tier_level: int, response: str, *, status: str = "success"
) -> None:
    with session_scope(db) as s:
        s.add(TierAnswer(
            run_id=rid, query_id=qid,
            tier_level=tier_level,
            tier_name=f"tier{tier_level}",
            response_text=response,
            prompt_tokens=5, completion_tokens=10, latency_ms=10,
            status=status,
            attempted_at=datetime.now(UTC),
        ))


def test_export_basic_shape(tmp_path: Path) -> None:
    db, rid = _setup(tmp_path)
    _add_pass1(db, rid, "q1", tier=1)
    _add_tier_answer(db, rid, "q1", 1, "Paris (tier1)")
    _add_tier_answer(db, rid, "q1", 5, "Paris, capital of France. (tier5)")

    out = tmp_path / "demo.json"
    summary = export_demo_json(db, rid, out)
    assert summary.queries_exported == 2
    assert out.exists()

    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert len(data) == 2

    q1 = next(e for e in data if e["id"] == "q1")
    assert q1["prompt"] == "easy"
    assert q1["expected_min_tier"] == 1
    assert q1["specializations"] == ["general"]
    assert q1["routed_tier"] == 1
    assert q1["routing_metadata"]["selected_model"] == "tier1"

    # Gold + routed responses present.
    assert q1["responses"]["gold"]["answer"] == "Paris."
    assert q1["responses"]["gold"]["tier"] == 5  # default gold tier
    assert q1["responses"]["routed"]["answer"] == "Paris (tier1)"
    assert q1["responses"]["routed"]["tier"] == 1

    # all_tier_answers includes both tiers we populated.
    assert q1["all_tier_answers"] == {
        "tier1": "Paris (tier1)",
        "tier5": "Paris, capital of France. (tier5)",
    }


def test_export_missing_routing_is_null(tmp_path: Path) -> None:
    """No pass1 row → routed_tier null, routed response null, gold still present."""
    db, rid = _setup(tmp_path)
    _add_tier_answer(db, rid, "q1", 2, "ans")

    out = tmp_path / "demo.json"
    export_demo_json(db, rid, out)
    q1 = next(e for e in json.loads(out.read_text()) if e["id"] == "q1")

    assert q1["routed_tier"] is None
    assert q1["routing_metadata"] is None
    assert q1["responses"]["routed"] is None
    assert q1["responses"]["gold"]["answer"] == "Paris."


def test_export_routed_tier_without_matching_answer(tmp_path: Path) -> None:
    """Router picked tier 3 but tier 3 hasn't been answered → routed response null."""
    db, rid = _setup(tmp_path)
    _add_pass1(db, rid, "q1", tier=3)
    _add_tier_answer(db, rid, "q1", 1, "tier1 ans")  # tier 3 missing

    out = tmp_path / "demo.json"
    export_demo_json(db, rid, out)
    q1 = next(e for e in json.loads(out.read_text()) if e["id"] == "q1")

    assert q1["routed_tier"] == 3
    assert q1["responses"]["routed"] is None
    assert q1["all_tier_answers"] == {"tier1": "tier1 ans"}


def test_export_failed_tier_excluded_from_all_tier_answers(tmp_path: Path) -> None:
    db, rid = _setup(tmp_path)
    _add_tier_answer(db, rid, "q1", 1, "good", status="success")
    _add_tier_answer(db, rid, "q1", 2, None, status="error")  # no response_text

    out = tmp_path / "demo.json"
    export_demo_json(db, rid, out)
    q1 = next(e for e in json.loads(out.read_text()) if e["id"] == "q1")

    assert q1["all_tier_answers"] == {"tier1": "good"}


def test_export_no_gold_means_null(tmp_path: Path) -> None:
    db, rid = _setup(tmp_path)
    with session_scope(db) as s:
        q = s.execute(select(Query).where(Query.query_id == "q1")).scalar_one()
        q.gold_answer = None

    out = tmp_path / "demo.json"
    export_demo_json(db, rid, out)
    q1 = next(e for e in json.loads(out.read_text()) if e["id"] == "q1")
    assert q1["responses"]["gold"] is None


def test_export_summary_counts(tmp_path: Path) -> None:
    db, rid = _setup(tmp_path)
    _add_pass1(db, rid, "q1", tier=1)
    _add_tier_answer(db, rid, "q1", 1, "a")
    _add_tier_answer(db, rid, "q1", 2, "b")
    _add_tier_answer(db, rid, "q2", 3, "c")
    # q2 has no pass1, just one tier answer.

    out = tmp_path / "demo.json"
    summary = export_demo_json(db, rid, out)
    assert summary.queries_exported == 2
    assert summary.with_routed_tier == 1  # only q1
    assert summary.with_routed_answer == 1  # q1 was routed to tier 1, which we have
    assert summary.tiers_per_query == {2: 1, 1: 1}  # q1 has 2 tier answers, q2 has 1
