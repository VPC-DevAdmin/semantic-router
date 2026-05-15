"""Tests for `update-gold` — regenerating gold via the top-tier model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from benchmark import update_gold as ug
from benchmark.db import Query, session_scope
from benchmark.tiers import ChatResult

from ._helpers import bootstrap_db, make_models

QUERIES = [
    {"id": "q1", "prompt": "what is 2+2", "expected_min_tier": 1,
     "specializations": ["general"], "expected_answer": "old gold 1"},
    {"id": "q2", "prompt": "explain entropy", "expected_min_tier": 3,
     "specializations": ["general"], "expected_answer": "old gold 2"},
]


@dataclass
class _FakeTopClient:
    fail_on: str | None = None
    empty_on: str | None = None

    async def chat(self, prompt: str, *, max_tokens=None, **_: Any) -> ChatResult:
        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError("top tier boom")
        content = "" if (self.empty_on and self.empty_on in prompt) \
            else f"FRESH GOLD :: {prompt}"
        return ChatResult(
            content=content, model="tier5",
            prompt_tokens=3, completion_tokens=7, latency_ms=42, raw={},
        )


@pytest.mark.asyncio
async def test_update_gold_overwrites_named_queries(tmp_path: Path, monkeypatch) -> None:
    db = bootstrap_db(tmp_path, QUERIES)
    monkeypatch.setattr(ug, "client_from_tier", lambda tier: _FakeTopClient())

    result = await ug.update_gold_answers(
        db, query_ids=["q1"], models=make_models([1, 2, 3, 4, 5])
    )
    assert result.updated == ["q1"]
    assert result.errors == []

    with session_scope(db) as s:
        q1 = s.execute(select(Query).where(Query.query_id == "q1")).scalar_one()
        q2 = s.execute(select(Query).where(Query.query_id == "q2")).scalar_one()
    assert q1.gold_answer == "FRESH GOLD :: what is 2+2"
    assert q1.gold_model.startswith(ug.REGEN_GOLD_MARKER)
    assert q1.gold_generated_at is not None
    # q2 untouched (wasn't requested).
    assert q2.gold_answer == "old gold 2"


@pytest.mark.asyncio
async def test_update_gold_uses_top_tier_model(tmp_path: Path, monkeypatch) -> None:
    """The client must be built from the MAX-level tier, not tier 1."""
    db = bootstrap_db(tmp_path, QUERIES)
    captured = {}

    def _capture(tier):
        captured["level"] = tier.level
        return _FakeTopClient()

    monkeypatch.setattr(ug, "client_from_tier", _capture)
    await ug.update_gold_answers(
        db, query_ids=["q1"], models=make_models([1, 2, 3, 4, 5])
    )
    assert captured["level"] == 5


@pytest.mark.asyncio
async def test_update_gold_reports_unknown_qids(tmp_path: Path, monkeypatch) -> None:
    db = bootstrap_db(tmp_path, QUERIES)
    monkeypatch.setattr(ug, "client_from_tier", lambda tier: _FakeTopClient())
    result = await ug.update_gold_answers(
        db, query_ids=["q1", "q99999"], models=make_models([1, 2, 5])
    )
    assert result.updated == ["q1"]
    assert result.skipped_unknown == ["q99999"]


@pytest.mark.asyncio
async def test_update_gold_errors_keep_old_gold(tmp_path: Path, monkeypatch) -> None:
    db = bootstrap_db(tmp_path, QUERIES)
    monkeypatch.setattr(
        ug, "client_from_tier", lambda tier: _FakeTopClient(fail_on="entropy")
    )
    result = await ug.update_gold_answers(
        db, query_ids=["q1", "q2"], models=make_models([1, 5])
    )
    assert result.updated == ["q1"]
    assert len(result.errors) == 1
    assert result.errors[0][0] == "q2"

    with session_scope(db) as s:
        q2 = s.execute(select(Query).where(Query.query_id == "q2")).scalar_one()
    # Failed regen must not clobber the prior gold.
    assert q2.gold_answer == "old gold 2"


@pytest.mark.asyncio
async def test_update_gold_empty_response_is_error(tmp_path: Path, monkeypatch) -> None:
    db = bootstrap_db(tmp_path, QUERIES)
    monkeypatch.setattr(
        ug, "client_from_tier", lambda tier: _FakeTopClient(empty_on="2+2")
    )
    result = await ug.update_gold_answers(
        db, query_ids=["q1"], models=make_models([1, 5])
    )
    assert result.updated == []
    assert len(result.errors) == 1
    assert "empty response" in result.errors[0][1]

    with session_scope(db) as s:
        q1 = s.execute(select(Query).where(Query.query_id == "q1")).scalar_one()
    assert q1.gold_answer == "old gold 1"  # unchanged
