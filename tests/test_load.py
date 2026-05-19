"""Loader semantics: insert, no-op, metadata update, gold-from-expected_answer."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from benchmark.db import GoldAnswer, Query, init_db, session_scope
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


def test_initial_insert_seeds_upstream_gold_answer(tmp_path: Path) -> None:
    db, qp = _setup(tmp_path, QUERIES_V1)
    load_into_db(qp, db)
    with session_scope(db) as s:
        g = s.execute(
            select(GoldAnswer)
            .where(GoldAnswer.query_id == "t001")
            .where(GoldAnswer.model_id == "upstream")
        ).scalar_one()
        assert g.answer == "4."
        assert g.source == "upstream"
        assert g.provider is None


def test_expected_answer_change_updates_upstream_gold_row(tmp_path: Path) -> None:
    db, qp = _setup(tmp_path, QUERIES_V1)
    load_into_db(qp, db)
    v2 = [dict(q) for q in QUERIES_V1]
    v2[0]["expected_answer"] = "Four. (Updated upstream.)"
    qp.write_text(json.dumps(v2))
    load_into_db(qp, db)
    with session_scope(db) as s:
        g = s.execute(
            select(GoldAnswer)
            .where(GoldAnswer.query_id == "t001")
            .where(GoldAnswer.model_id == "upstream")
        ).scalar_one()
        assert g.answer == "Four. (Updated upstream.)"


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


# ─── Multiple expected answers per query ───────────────────────────────────

def test_multiple_expected_answers_seed_one_gold_row_each(tmp_path: Path) -> None:
    queries = [{
        "id": "m001", "prompt": "Capital of France?",
        "expected_min_tier": 1, "specializations": ["general"],
        "expected_answers": [
            {"answer": "Paris.",
             "source": "upstream", "provider": None, "model": "upstream"},
            {"answer": "Paris, the capital of France.",
             "source": "human", "provider": None, "model": "expert-review"},
            {"answer": "Paris (vendor).",
             "source": "vendor-export", "provider": "Anthropic",
             "model": "claude-opus-4-7"},
        ],
    }]
    db, qp = _setup(tmp_path, queries)
    r = load_into_db(qp, db)
    assert r.inserted == 1

    with session_scope(db) as s:
        rows = s.execute(
            select(GoldAnswer).where(GoldAnswer.query_id == "m001")
        ).scalars().all()
    by_mid = {r.model_id: r for r in rows}
    assert set(by_mid) == {"upstream", "expert-review", "claude-opus-4-7"}
    assert by_mid["upstream"].source == "upstream"
    assert by_mid["expert-review"].source == "human"
    assert by_mid["claude-opus-4-7"].provider == "Anthropic"
    assert by_mid["claude-opus-4-7"].answer == "Paris (vendor)."


def test_model_defaults_to_source_when_omitted(tmp_path: Path) -> None:
    queries = [{
        "id": "m002", "prompt": "p", "expected_min_tier": 1,
        "specializations": ["general"],
        "expected_answers": [
            {"answer": "A", "source": "human"},          # model → "human"
            {"answer": "B", "source": "vendor-export"},  # model → "vendor-export"
        ],
    }]
    db, qp = _setup(tmp_path, queries)
    load_into_db(qp, db)
    with session_scope(db) as s:
        mids = {
            r.model_id for r in s.execute(
                select(GoldAnswer).where(GoldAnswer.query_id == "m002")
            ).scalars()
        }
    assert mids == {"human", "vendor-export"}


def test_legacy_and_multi_coexist(tmp_path: Path) -> None:
    """expected_answer (legacy single) + expected_answers (list) → one
    upstream row plus the listed rows."""
    queries = [{
        "id": "m003", "prompt": "p", "expected_min_tier": 1,
        "specializations": ["general"],
        "expected_answer": "Upstream gold.",
        "expected_answers": [
            {"answer": "Anthropic gold.", "source": "update-gold",
             "provider": "Anthropic", "model": "claude-opus-4-7"},
        ],
    }]
    db, qp = _setup(tmp_path, queries)
    load_into_db(qp, db)
    with session_scope(db) as s:
        rows = s.execute(
            select(GoldAnswer).where(GoldAnswer.query_id == "m003")
        ).scalars().all()
        by_mid = {r.model_id: r for r in rows}
        # Query.gold_answer mirrors the upstream entry (back-compat).
        q = s.execute(select(Query).where(Query.query_id == "m003")).scalar_one()
    assert set(by_mid) == {"upstream", "claude-opus-4-7"}
    assert by_mid["upstream"].answer == "Upstream gold."
    assert q.gold_answer == "Upstream gold."
    assert q.gold_model == GOLD_SOURCE_MARKER


def test_duplicate_gold_model_id_rejected(tmp_path: Path) -> None:
    queries = [{
        "id": "m004", "prompt": "p", "expected_min_tier": 1,
        "specializations": ["general"],
        "expected_answers": [
            {"answer": "A", "source": "human", "model": "dup"},
            {"answer": "B", "source": "vendor-export", "model": "dup"},
        ],
    }]
    db, qp = _setup(tmp_path, queries)
    with pytest.raises(Exception, match="duplicate gold model id"):
        load_into_db(qp, db)


def test_reload_multi_is_noop(tmp_path: Path) -> None:
    queries = [{
        "id": "m005", "prompt": "p", "expected_min_tier": 1,
        "specializations": ["general"],
        "expected_answers": [
            {"answer": "A", "source": "human", "model": "h"},
            {"answer": "B", "source": "vendor-export", "model": "v",
             "provider": "Anthropic"},
        ],
    }]
    db, qp = _setup(tmp_path, queries)
    load_into_db(qp, db)
    r2 = load_into_db(qp, db)
    assert (r2.inserted, r2.updated, r2.unchanged) == (0, 0, 1)


def test_reload_edits_secondary_gold_triggers_update(tmp_path: Path) -> None:
    queries = [{
        "id": "m006", "prompt": "p", "expected_min_tier": 1,
        "specializations": ["general"],
        "expected_answers": [
            {"answer": "A v1", "source": "human", "model": "h"},
        ],
    }]
    db, qp = _setup(tmp_path, queries)
    load_into_db(qp, db)
    queries[0]["expected_answers"][0]["answer"] = "A v2 (edited)"
    qp.write_text(json.dumps(queries))
    r2 = load_into_db(qp, db)
    assert r2.updated == 1
    with session_scope(db) as s:
        row = s.execute(
            select(GoldAnswer).where(GoldAnswer.query_id == "m006")
            .where(GoldAnswer.model_id == "h")
        ).scalar_one()
    assert row.answer == "A v2 (edited)"


def test_load_does_not_clobber_update_gold_rows(tmp_path: Path) -> None:
    """A row from update-gold / import-answers (model_id outside the
    file) must NOT be deleted by re-loading queries.json."""
    queries = [{
        "id": "m007", "prompt": "p", "expected_min_tier": 1,
        "specializations": ["general"],
        "expected_answer": "Upstream.",
    }]
    db, qp = _setup(tmp_path, queries)
    load_into_db(qp, db)
    # Simulate an update-gold row landing alongside.
    with session_scope(db) as s:
        s.add(GoldAnswer(
            query_id="m007", model_id="gpt-5", provider="OpenAI",
            answer="OpenAI gold.", source="update-gold",
            generated_at=datetime.now(UTC),
        ))
    # Re-load (no-op for the file content).
    load_into_db(qp, db)
    with session_scope(db) as s:
        mids = {
            r.model_id for r in s.execute(
                select(GoldAnswer).where(GoldAnswer.query_id == "m007")
            ).scalars()
        }
    assert mids == {"upstream", "gpt-5"}  # update-gold row preserved


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
