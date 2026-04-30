"""Smoke tests for seed semantics: insert, no-op, metadata update, gold invalidation."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from benchmark.db import Query, init_db, session_scope
from benchmark.seed import seed_from_yaml

QUERIES_V1 = """
- id: t001
  prompt: "What is 2+2?"
  expected_min_tier: 1
  specializations: [general]
- id: t002
  prompt: "Refactor x"
  expected_min_tier: 2
  specializations: [code]
"""

QUERIES_V2_METADATA_ONLY = """
- id: t001
  prompt: "What is 2+2?"
  expected_min_tier: 1
  specializations: [general]
  notes: now with notes
- id: t002
  prompt: "Refactor x"
  expected_min_tier: 2
  specializations: [code]
"""

QUERIES_V3_PROMPT_CHANGED = """
- id: t001
  prompt: "What is 2+2? (revised wording)"
  expected_min_tier: 1
  specializations: [general]
  notes: now with notes
- id: t002
  prompt: "Refactor x"
  expected_min_tier: 2
  specializations: [code]
"""


def _setup(tmp_path: Path, yaml_text: str) -> tuple[Path, Path]:
    db_path = tmp_path / "test.db"
    yaml_path = tmp_path / "queries.yaml"
    yaml_path.write_text(yaml_text)
    init_db(db_path)
    return db_path, yaml_path


def test_initial_insert_then_noop(tmp_path: Path) -> None:
    db_path, yaml_path = _setup(tmp_path, QUERIES_V1)

    r1 = seed_from_yaml(yaml_path, db_path)
    assert (r1.inserted, r1.updated, r1.unchanged) == (2, 0, 0)

    r2 = seed_from_yaml(yaml_path, db_path)
    assert (r2.inserted, r2.updated, r2.unchanged) == (0, 0, 2)


def test_metadata_only_update_preserves_gold(tmp_path: Path) -> None:
    db_path, yaml_path = _setup(tmp_path, QUERIES_V1)
    seed_from_yaml(yaml_path, db_path)

    # Simulate gold answer existing.
    with session_scope(db_path) as session:
        q = session.execute(select(Query).where(Query.query_id == "t001")).scalar_one()
        q.gold_answer = "4"
        q.gold_model = "tier5-frontier"
        q.gold_generated_at = datetime.now(UTC)

    yaml_path.write_text(QUERIES_V2_METADATA_ONLY)
    report = seed_from_yaml(yaml_path, db_path)

    assert (report.inserted, report.updated, report.unchanged) == (0, 1, 1)
    assert report.gold_invalidated == 0

    with session_scope(db_path) as session:
        q = session.execute(select(Query).where(Query.query_id == "t001")).scalar_one()
        assert q.notes == "now with notes"
        assert q.gold_answer == "4"  # preserved


def test_prompt_change_invalidates_gold(tmp_path: Path) -> None:
    db_path, yaml_path = _setup(tmp_path, QUERIES_V2_METADATA_ONLY)
    seed_from_yaml(yaml_path, db_path)

    with session_scope(db_path) as session:
        q = session.execute(select(Query).where(Query.query_id == "t001")).scalar_one()
        q.gold_answer = "4"
        q.gold_model = "tier5-frontier"
        q.gold_generated_at = datetime.now(UTC)

    yaml_path.write_text(QUERIES_V3_PROMPT_CHANGED)
    report = seed_from_yaml(yaml_path, db_path)

    assert report.gold_invalidated == 1
    assert report.invalidated_ids == ["t001"]

    with session_scope(db_path) as session:
        q = session.execute(select(Query).where(Query.query_id == "t001")).scalar_one()
        assert q.gold_answer is None
        assert q.gold_model is None
        assert q.gold_generated_at is None


def test_duplicate_id_in_yaml_raises(tmp_path: Path) -> None:
    yaml_text = """
- id: dup
  prompt: "a"
  expected_min_tier: 1
  specializations: [general]
- id: dup
  prompt: "b"
  expected_min_tier: 1
  specializations: [general]
"""
    db_path, yaml_path = _setup(tmp_path, yaml_text)
    with pytest.raises(ValueError, match="duplicate query id"):
        seed_from_yaml(yaml_path, db_path)
