"""Loader semantics: insert / no-op / metadata update / multi-gold sync."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from benchmark.db import GoldAnswer, Query, init_db, session_scope
from benchmark.load import load_into_db

QUERIES_V1 = [
    {
        "id": "t001",
        "prompt": "What is 2+2?",
        "expected_answers": [
            {"answer": "4.", "model": "Opus", "provider": "Anthropic"},
        ],
        "expected_min_tier": 1,
        "specializations": ["general"],
    },
    {
        "id": "t002",
        "prompt": "Refactor x",
        "expected_answers": [
            {"answer": "use list comprehension", "model": "Opus",
             "provider": "Anthropic"},
        ],
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


def test_initial_insert_seeds_gold(tmp_path: Path) -> None:
    db, qp = _setup(tmp_path, QUERIES_V1)
    r = load_into_db(qp, db)
    assert (r.inserted, r.updated, r.unchanged) == (2, 0, 0)

    with session_scope(db) as s:
        g = s.execute(
            select(GoldAnswer)
            .where(GoldAnswer.query_id == "t001")
            .where(GoldAnswer.model_id == "Opus")
        ).scalar_one()
        assert g.answer == "4."
        assert g.provider == "Anthropic"
        assert g.generated_at is not None


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
        # Gold preserved.
        g = s.execute(
            select(GoldAnswer).where(GoldAnswer.query_id == "t001")
            .where(GoldAnswer.model_id == "Opus")
        ).scalar_one()
        assert g.answer == "4."


def test_gold_edit_triggers_update(tmp_path: Path) -> None:
    db, qp = _setup(tmp_path, QUERIES_V1)
    load_into_db(qp, db)
    v2 = json.loads(json.dumps(QUERIES_V1))
    v2[0]["expected_answers"][0]["answer"] = "Four. (Updated upstream.)"
    qp.write_text(json.dumps(v2))
    r = load_into_db(qp, db)
    assert r.updated == 1
    with session_scope(db) as s:
        g = s.execute(
            select(GoldAnswer).where(GoldAnswer.query_id == "t001")
            .where(GoldAnswer.model_id == "Opus")
        ).scalar_one()
        assert g.answer == "Four. (Updated upstream.)"


def test_prompt_change_refreshes_gold(tmp_path: Path) -> None:
    db, qp = _setup(tmp_path, QUERIES_V1)
    load_into_db(qp, db)
    v2 = json.loads(json.dumps(QUERIES_V1))
    v2[0]["prompt"] = "What is 2+2? (rephrased)"
    v2[0]["expected_answers"][0]["answer"] = "Still 4."
    qp.write_text(json.dumps(v2))
    load_into_db(qp, db)
    with session_scope(db) as s:
        q = s.execute(select(Query).where(Query.query_id == "t001")).scalar_one()
        g = s.execute(
            select(GoldAnswer).where(GoldAnswer.query_id == "t001")
            .where(GoldAnswer.model_id == "Opus")
        ).scalar_one()
    assert "rephrased" in q.prompt
    assert g.answer == "Still 4."


def test_duplicate_id_raises(tmp_path: Path) -> None:
    queries = [
        {"id": "dup", "prompt": "a", "expected_min_tier": 1,
         "specializations": ["general"], "expected_answers": []},
        {"id": "dup", "prompt": "b", "expected_min_tier": 1,
         "specializations": ["general"], "expected_answers": []},
    ]
    db, qp = _setup(tmp_path, queries)
    with pytest.raises(ValueError, match="duplicate query id"):
        load_into_db(qp, db)


# ─── Multiple expected answers per query ───────────────────────────────────

def test_multi_provider_golds_seed_one_row_per_model(tmp_path: Path) -> None:
    queries = [{
        "id": "m001",
        "prompt": "What is 17 + 26?",
        "expected_answers": [
            {"answer": "43.", "model": "Opus 4.7", "provider": "Anthropic"},
            {"answer": "17 + 26 = 43.", "model": "GPT-5.5",
             "provider": "OpenAI"},
        ],
        "expected_min_tier": 1,
        "specializations": ["general"],
        "domain_tags": ["arithmetic"],
        "notes": "Single-step arithmetic with no ambiguity.",
    }]
    db, qp = _setup(tmp_path, queries)
    assert load_into_db(qp, db).inserted == 1

    with session_scope(db) as s:
        rows = s.execute(
            select(GoldAnswer).where(GoldAnswer.query_id == "m001")
        ).scalars().all()
    by_mid = {r.model_id: r for r in rows}
    assert set(by_mid) == {"Opus 4.7", "GPT-5.5"}
    assert by_mid["Opus 4.7"].provider == "Anthropic"
    assert by_mid["Opus 4.7"].answer == "43."
    assert by_mid["GPT-5.5"].provider == "OpenAI"
    assert by_mid["GPT-5.5"].answer == "17 + 26 = 43."


def test_single_gold_must_still_be_a_list(tmp_path: Path) -> None:
    """One gold = a one-entry expected_answers list — no scalar shortcut."""
    queries = [{
        "id": "m002", "prompt": "p", "expected_min_tier": 1,
        "specializations": ["general"],
        "expected_answers": [
            {"answer": "only one", "model": "Opus", "provider": "Anthropic"},
        ],
    }]
    db, qp = _setup(tmp_path, queries)
    load_into_db(qp, db)
    with session_scope(db) as s:
        rows = s.execute(
            select(GoldAnswer).where(GoldAnswer.query_id == "m002")
        ).scalars().all()
    assert len(rows) == 1 and rows[0].model_id == "Opus"


def test_legacy_expected_answer_field_rejected(tmp_path: Path) -> None:
    """The old scalar `expected_answer` field is not part of the schema."""
    queries = [{
        "id": "m003", "prompt": "p", "expected_min_tier": 1,
        "specializations": ["general"],
        "expected_answer": "I'm a stale legacy field.",
    }]
    db, qp = _setup(tmp_path, queries)
    with pytest.raises(Exception, match="[Ee]xtra|expected_answer|forbid"):
        load_into_db(qp, db)


def test_model_required_on_each_entry(tmp_path: Path) -> None:
    queries = [{
        "id": "m004", "prompt": "p", "expected_min_tier": 1,
        "specializations": ["general"],
        "expected_answers": [{"answer": "A"}],  # missing model
    }]
    db, qp = _setup(tmp_path, queries)
    with pytest.raises(Exception, match="model"):
        load_into_db(qp, db)


def test_duplicate_model_id_within_query_rejected(tmp_path: Path) -> None:
    queries = [{
        "id": "m005", "prompt": "p", "expected_min_tier": 1,
        "specializations": ["general"],
        "expected_answers": [
            {"answer": "A", "model": "dup"},
            {"answer": "B", "model": "dup"},
        ],
    }]
    db, qp = _setup(tmp_path, queries)
    with pytest.raises(Exception, match="duplicate gold model id"):
        load_into_db(qp, db)


def test_reload_does_not_clobber_update_gold_rows(tmp_path: Path) -> None:
    """A gold row with a model_id NOT declared in the file (e.g. from
    `make update-gold` or `make import-answers`) must survive reload."""
    queries = [{
        "id": "m006", "prompt": "p", "expected_min_tier": 1,
        "specializations": ["general"],
        "expected_answers": [
            {"answer": "Opus answer.", "model": "Opus", "provider": "Anthropic"},
        ],
    }]
    db, qp = _setup(tmp_path, queries)
    load_into_db(qp, db)
    with session_scope(db) as s:
        s.add(GoldAnswer(
            query_id="m006", model_id="gpt-5", provider="OpenAI",
            answer="OpenAI answer.",
            generated_at=datetime.now(UTC),
        ))
    load_into_db(qp, db)  # no-op on the file's content
    with session_scope(db) as s:
        mids = {
            r.model_id for r in s.execute(
                select(GoldAnswer).where(GoldAnswer.query_id == "m006")
            ).scalars()
        }
    assert mids == {"Opus", "gpt-5"}


def test_query_specializations_are_free_form(tmp_path: Path) -> None:
    """Specializations are downstream metadata (sort/review/matches metric),
    not routing inputs — so the loader accepts any non-empty list of
    strings rather than enforcing a whitelist. Verbatim labels survive
    into the DB."""
    queries = [{
        "id": "x", "prompt": "p", "expected_min_tier": 1,
        "specializations": ["code", "creative", "anything-goes"],
        "expected_answers": [],
    }]
    db, qp = _setup(tmp_path, queries)
    r = load_into_db(qp, db)
    assert r.inserted == 1
    with session_scope(db) as s:
        q = s.execute(select(Query).where(Query.query_id == "x")).scalar_one()
    assert q.specializations == ["code", "creative", "anything-goes"]


def test_empty_specializations_still_rejected(tmp_path: Path) -> None:
    """An empty list is still an error — every query must have at least one."""
    queries = [{
        "id": "x", "prompt": "p", "expected_min_tier": 1,
        "specializations": [], "expected_answers": [],
    }]
    db, qp = _setup(tmp_path, queries)
    with pytest.raises(Exception, match="at least one specialization"):
        load_into_db(qp, db)
