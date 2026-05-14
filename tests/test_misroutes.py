"""Tests for the misroutes diagnostic."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from benchmark.db import Pass1Result, Query, session_scope
from benchmark.misroutes import list_misroutes, render_misroutes
from benchmark.runs import create_run

from ._helpers import bootstrap_db, make_models_yaml, make_router_yaml

QUERIES = [
    {"id": "q1", "prompt": "trivial", "expected_min_tier": 1, "specializations": ["general"]},
    {
        "id": "q2", "prompt": "hard reasoning task",
        "expected_min_tier": 4, "specializations": ["reasoning"],
    },
    {
        "id": "q3", "prompt": "judgment-heavy advice",
        "expected_min_tier": 4, "specializations": ["general"],
    },
    {
        "id": "q4", "prompt": "frontier synthesis",
        "expected_min_tier": 5, "specializations": ["reasoning", "creative_writing"],
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


def _add_pass1(
    db: Path, rid: int, qid: str, *,
    routed_tier: int, meets_min: int,
    category: str = "other",
) -> None:
    with session_scope(db) as s:
        s.add(Pass1Result(
            run_id=rid, query_id=qid,
            router_selected_model=f"tier{routed_tier}",
            router_selected_tier=routed_tier,
            router_selected_specs=["general"],
            meets_minimum_tier=meets_min,
            matches_specialization=1,
            latency_ms=20,
            raw_routing_metadata={"category": category, "reasoning": "off"},
            status="success",
            attempted_at=datetime.now(UTC),
        ))


def test_list_misroutes_only_includes_under_routes(tmp_path: Path) -> None:
    db, rid = _setup(tmp_path)
    # q1: expected 1, routed to 1, met. NOT a misroute.
    _add_pass1(db, rid, "q1", routed_tier=1, meets_min=1)
    # q2: expected 4, routed to 2, MISROUTE.
    _add_pass1(db, rid, "q2", routed_tier=2, meets_min=0, category="business")
    # q3: expected 4, routed to 5, met (over-route, not flagged).
    _add_pass1(db, rid, "q3", routed_tier=5, meets_min=1)
    # q4: expected 5, routed to 3, MISROUTE.
    _add_pass1(db, rid, "q4", routed_tier=3, meets_min=0, category="other")

    misroutes = list_misroutes(db, run_id=rid)
    assert [m.query_id for m in misroutes] == ["q4", "q2"]  # higher expected_min_tier first
    assert misroutes[0].routed_tier == 3
    assert misroutes[1].category == "business"


def test_list_misroutes_empty_when_no_runs(tmp_path: Path) -> None:
    db, _rid = _setup(tmp_path)
    # No misroutes recorded → empty list.
    assert list_misroutes(db, run_id=None) == [] or list_misroutes(db, run_id=999) == []


def test_render_misroutes_handles_empty() -> None:
    out = render_misroutes([])
    assert "No misroutes" in out


def test_render_misroutes_contains_breakdown(tmp_path: Path) -> None:
    db, rid = _setup(tmp_path)
    _add_pass1(db, rid, "q2", routed_tier=2, meets_min=0, category="business")
    _add_pass1(db, rid, "q3", routed_tier=2, meets_min=0, category="business")
    _add_pass1(db, rid, "q4", routed_tier=3, meets_min=0, category="other")

    out = render_misroutes(list_misroutes(db, run_id=rid))
    assert "3 misroute" in out
    assert "T4 expected:" in out
    assert "T5 expected:" in out
    assert "routed→T2:" in out
    assert "business" in out
    assert "other" in out


def test_list_misroutes_truncates_prompt_in_render(tmp_path: Path) -> None:
    """Long prompts get textwrap.shortened in the rendered output."""
    db, rid = _setup(tmp_path)
    # Add a long-prompt query.
    long_prompt = "very " * 200 + "long prompt"
    with session_scope(db) as s:
        s.add(Query(
            query_id="qlong",
            prompt=long_prompt,
            prompt_hash="x",
            expected_min_tier=5,
            specializations=["general"],
        ))
    _add_pass1(db, rid, "qlong", routed_tier=1, meets_min=0)

    out = render_misroutes(list_misroutes(db, run_id=rid))
    # The rendered output should not contain the full long prompt verbatim.
    assert long_prompt not in out
    assert "qlong" in out
