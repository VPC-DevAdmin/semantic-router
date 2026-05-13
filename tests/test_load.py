"""Loader semantics: insert, no-op, metadata update, gold-from-expected_answer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import select

from benchmark.db import Query, init_db, session_scope
from benchmark.load import GOLD_SOURCE_MARKER, load_into_db

QUERIES_V1 = [
    {
        "id": "t001",
        "prompt": "What is 2+2?",
        "expected_answer": "4.",
        "expected_min_tier": 1,
        "specializations": ["general"],
    },
    {
        "id": "t002",
        "prompt": "Refactor x",
        "expected_answer": "use list comprehension",
        "expected_min_tier": 2,
        "specializations": ["coding"],
    },
]


def _setup(tmp_path: Path, queries: list[dict]) -> tuple[Path, Path]:
    db = tmp_path / "test.db"
    qp = tmp_path / "queries.json"
    qp.write_text(json.dumps(queries))
    init_db(db)
    return db, qp


def test_initial_insert_populates_gold(tmp_path: Path) -> None:
    db, qp = _setup(tmp_path, QUERIES_V1)
    r = load_into_db(qp, db)
    assert (r.inserted, r.updated, r.unchanged) == (2, 0, 0)

    with session_scope(db) as s:
        t001 = s.execute(select(Query).where(Query.query_id == "t001")).scalar_one()
        assert t001.gold_answer == "4."
        assert t001.gold_model == GOLD_SOURCE_MARKER
        assert t001.gold_generated_at is not None


def test_reload_is_noop(tmp_path: Path) -> None:
    db, qp = _setup(tmp_path, QUERIES_V1)
    load_into_db(qp, db)
    r2 = load_into_db(qp, db)
    assert (r2.inserted, r2.updated, r2.unchanged) == (0, 0, 2)


def test_metadata_change_updates(tmp_path: Path) -> None:
    db, qp = _setup(tmp_path, QUERIES_V1)
    load_into_db(qp, db)
    v2 = [dict(q) for q in QUERIES_V1]
    v2[0]["notes"] = "now with notes"
    qp.write_text(json.dumps(v2))
    r = load_into_db(qp, db)
    assert (r.inserted, r.updated, r.unchanged) == (0, 1, 1)
    with session_scope(db) as s:
        q = s.execute(select(Query).where(Query.query_id == "t001")).scalar_one()
        assert q.notes == "now with notes"
        assert q.gold_answer == "4."  # gold preserved


def test_expected_answer_change_updates_gold(tmp_path: Path) -> None:
    db, qp = _setup(tmp_path, QUERIES_V1)
    load_into_db(qp, db)
    v2 = [dict(q) for q in QUERIES_V1]
    v2[0]["expected_answer"] = "Four. (Updated upstream.)"
    qp.write_text(json.dumps(v2))
    r = load_into_db(qp, db)
    assert r.updated == 1
    with session_scope(db) as s:
        q = s.execute(select(Query).where(Query.query_id == "t001")).scalar_one()
        assert q.gold_answer == "Four. (Updated upstream.)"


def test_prompt_change_refreshes_gold(tmp_path: Path) -> None:
    db, qp = _setup(tmp_path, QUERIES_V1)
    load_into_db(qp, db)
    v2 = [dict(q) for q in QUERIES_V1]
    v2[0]["prompt"] = "What is 2+2? (rephrased)"
    v2[0]["expected_answer"] = "Still 4."
    qp.write_text(json.dumps(v2))
    load_into_db(qp, db)
    with session_scope(db) as s:
        q = s.execute(select(Query).where(Query.query_id == "t001")).scalar_one()
        assert "rephrased" in q.prompt
        assert q.gold_answer == "Still 4."


def test_duplicate_id_raises(tmp_path: Path) -> None:
    queries = [
        {"id": "dup", "prompt": "a", "expected_min_tier": 1, "specializations": ["general"]},
        {"id": "dup", "prompt": "b", "expected_min_tier": 1, "specializations": ["general"]},
    ]
    db, qp = _setup(tmp_path, queries)
    with pytest.raises(ValueError, match="duplicate query id"):
        load_into_db(qp, db)


def test_unknown_specialization_rejected(tmp_path: Path) -> None:
    queries = [
        {
            "id": "x", "prompt": "p", "expected_min_tier": 1,
            "specializations": ["not_a_real_spec"],
        }
    ]
    db, qp = _setup(tmp_path, queries)
    with pytest.raises(Exception, match="unknown specializations"):
        load_into_db(qp, db)
