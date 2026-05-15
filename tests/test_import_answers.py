"""Tests for the import-answers markdown parser and DB upsert."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from benchmark.db import TierAnswer, session_scope
from benchmark.import_answers import import_answers_file, parse_answers_markdown
from benchmark.runs import create_run

from ._helpers import bootstrap_db, make_models, make_models_yaml, make_router_yaml

QUERIES = [
    {"id": "q00046", "prompt": "career decision",
     "expected_min_tier": 4, "specializations": ["general"]},
    {"id": "q00068", "prompt": "engineering deadlines",
     "expected_min_tier": 4, "specializations": ["general"]},
    {"id": "q00070", "prompt": "atomic weapons",
     "expected_min_tier": 4, "specializations": ["general"]},
    {"id": "q00026", "prompt": "what is inflation",
     "expected_min_tier": 3, "specializations": ["general"]},
]


def _bootstrap(tmp_path: Path) -> tuple[Path, int]:
    db = bootstrap_db(tmp_path, QUERIES)
    rid = create_run(
        db,
        router_config_path=make_router_yaml(tmp_path),
        models_config_path=make_models_yaml(tmp_path),
    )
    return db, rid


# ─── Parser ─────────────────────────────────────────────────────────────────

def test_parse_handles_em_dash_separator() -> None:
    md = """\
## q00046 — Staff Role at Startup vs. Big Tech

The financial gap here is large.

## q00068 — PM vs. Engineering Team

Reframe the question.
"""
    parsed = parse_answers_markdown(md)
    assert parsed == [
        ("q00046", "The financial gap here is large."),
        ("q00068", "Reframe the question."),
    ]


def test_parse_handles_colon_separator() -> None:
    md = """\
## q00025: Hash Tables & O(1) Lookup
Hash spreads keys across buckets.

## q00026: Inflation
Too much money chasing goods.
"""
    parsed = parse_answers_markdown(md)
    assert parsed == [
        ("q00025", "Hash spreads keys across buckets."),
        ("q00026", "Too much money chasing goods."),
    ]


def test_parse_strips_horizontal_rule_separators() -> None:
    md = """\
## q00046 — Title

Body of q00046.

---

## q00068 — Other Title

Body of q00068.
---
"""
    parsed = parse_answers_markdown(md)
    qids = [p[0] for p in parsed]
    bodies = [p[1] for p in parsed]
    assert qids == ["q00046", "q00068"]
    assert "---" not in bodies[0]
    assert "---" not in bodies[1]
    assert "Body of q00046." in bodies[0]
    assert "Body of q00068." in bodies[1]


def test_parse_preserves_multiparagraph_body() -> None:
    md = """\
## q00046 — Title

First paragraph.

Second paragraph with **bold** and `code`.

```python
print("hello")
```

Third paragraph.

## q00068 — Next
Short body.
"""
    parsed = parse_answers_markdown(md)
    assert parsed[0][0] == "q00046"
    body = parsed[0][1]
    assert "First paragraph." in body
    assert "Second paragraph" in body
    assert "```python" in body
    assert "Third paragraph." in body


def test_parse_no_sections_returns_empty() -> None:
    assert parse_answers_markdown("# Just a regular heading\n\nNo qNNNNN sections here.") == []


def test_parse_ignores_non_qid_headings() -> None:
    md = """\
## Introduction

Some prose.

## q00046 — Title

Body.

## Conclusion

More prose.
"""
    parsed = parse_answers_markdown(md)
    assert len(parsed) == 1
    assert parsed[0][0] == "q00046"


# ─── DB upsert ──────────────────────────────────────────────────────────────

def test_import_inserts_new_rows(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    md_path = tmp_path / "answers.md"
    md_path.write_text(
        "## q00046 — Career advice\n\nLong-form answer for q00046.\n\n"
        "## q00068 — Eng team\n\nAnswer for q00068.\n"
    )
    models = make_models([1, 2, 3, 4, 5])
    result = import_answers_file(db, rid, tier_level=4, file_path=md_path, models=models)
    assert result.parsed == 2
    assert result.inserted == 2
    assert result.updated == 0

    with session_scope(db) as s:
        rows = s.execute(
            select(TierAnswer).where(TierAnswer.run_id == rid)
        ).scalars().all()
        by_qid = {r.query_id: r for r in rows}
    assert set(by_qid.keys()) == {"q00046", "q00068"}
    assert by_qid["q00046"].tier_level == 4
    assert by_qid["q00046"].tier_name == "tier4"
    assert by_qid["q00046"].response_text == "Long-form answer for q00046."
    assert by_qid["q00046"].status == "success"


def test_import_updates_existing_rows(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    # Pre-seed a row at tier 4 with old/error content.
    from datetime import UTC, datetime
    with session_scope(db) as s:
        s.add(TierAnswer(
            run_id=rid, query_id="q00046", tier_level=4, tier_name="tier4",
            response_text="OLD ANSWER", status="error", error_msg="prior failure",
            attempted_at=datetime.now(UTC),
        ))

    md_path = tmp_path / "answers.md"
    md_path.write_text("## q00046 — Title\n\nFresh imported answer.\n")
    models = make_models([1, 2, 3, 4, 5])

    result = import_answers_file(db, rid, tier_level=4, file_path=md_path, models=models)
    assert result.inserted == 0
    assert result.updated == 1

    with session_scope(db) as s:
        row = s.execute(
            select(TierAnswer)
            .where(TierAnswer.run_id == rid)
            .where(TierAnswer.query_id == "q00046")
        ).scalar_one()
    assert row.response_text == "Fresh imported answer."
    assert row.status == "success"
    assert row.error_msg is None


def test_import_skips_unknown_query_ids(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    md_path = tmp_path / "answers.md"
    md_path.write_text(
        "## q00046 — Real\n\nReal answer.\n\n"
        "## q99999 — Unknown\n\nThis qid doesn't exist in the DB.\n"
    )
    models = make_models([1, 2, 3, 4, 5])
    result = import_answers_file(db, rid, tier_level=4, file_path=md_path, models=models)
    assert result.inserted == 1
    assert result.skipped_unknown == ["q99999"]


def test_import_skips_empty_bodies(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    md_path = tmp_path / "answers.md"
    md_path.write_text(
        "## q00046 — With body\n\nReal answer.\n\n"
        "## q00068 — Empty section\n\n"
    )
    models = make_models([1, 2, 3, 4, 5])
    result = import_answers_file(db, rid, tier_level=4, file_path=md_path, models=models)
    assert result.inserted == 1
    assert result.skipped_empty == ["q00068"]


def test_import_idempotent_re_run(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    md_path = tmp_path / "answers.md"
    md_path.write_text("## q00046 — Title\n\nAnswer v1.\n")
    models = make_models([1, 2, 3, 4, 5])

    r1 = import_answers_file(db, rid, tier_level=4, file_path=md_path, models=models)
    assert r1.inserted == 1

    # Re-run with updated content — should update, not insert duplicate.
    md_path.write_text("## q00046 — Title\n\nAnswer v2 (revised).\n")
    r2 = import_answers_file(db, rid, tier_level=4, file_path=md_path, models=models)
    assert r2.inserted == 0
    assert r2.updated == 1

    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.run_id == rid)).scalars().all()
    assert len(rows) == 1
    assert rows[0].response_text == "Answer v2 (revised)."
