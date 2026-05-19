"""Tests for `update-gold` — regenerating per-provider gold via the top tier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from benchmark import update_gold as ug
from benchmark.db import GoldAnswer, Query, session_scope
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


def _patch(monkeypatch, client: _FakeTopClient) -> None:
    monkeypatch.setattr(ug, "client_from_model", lambda m: client)


@pytest.mark.asyncio
async def test_update_gold_writes_per_model_gold(tmp_path: Path, monkeypatch) -> None:
    db = bootstrap_db(tmp_path, QUERIES)
    _patch(monkeypatch, _FakeTopClient())

    result = await ug.update_gold_answers(
        db, query_ids=["q1"], models=make_models([1, 2, 3, 4, 5])
    )
    # make_models gives the top tier (5) a single model "tier5".
    assert result.updated == [("q1", "tier5")]
    assert result.errors == []

    with session_scope(db) as s:
        g = s.execute(
            select(GoldAnswer)
            .where(GoldAnswer.query_id == "q1")
            .where(GoldAnswer.model_id == "tier5")
        ).scalar_one()
        assert g.answer == "FRESH GOLD :: what is 2+2"
        assert g.source == ug.REGEN_GOLD_SOURCE
        # The upstream gold row is still there, untouched.
        ups = s.execute(
            select(GoldAnswer)
            .where(GoldAnswer.query_id == "q1")
            .where(GoldAnswer.model_id == "upstream")
        ).scalar_one()
        assert ups.answer == "old gold 1"
        # slot-0 model also refreshed the back-compat Query.gold_answer.
        q1 = s.execute(select(Query).where(Query.query_id == "q1")).scalar_one()
        assert q1.gold_answer == "FRESH GOLD :: what is 2+2"
        assert q1.gold_model.startswith(ug.REGEN_GOLD_MARKER)
        # q2 untouched.
        q2 = s.execute(select(Query).where(Query.query_id == "q2")).scalar_one()
        assert q2.gold_answer == "old gold 2"


@pytest.mark.asyncio
async def test_update_gold_uses_top_tier(tmp_path: Path, monkeypatch) -> None:
    """Models passed to client_from_model must be the MAX-level tier's."""
    db = bootstrap_db(tmp_path, QUERIES)
    seen: list[str] = []

    def _cap(m):
        seen.append(m.served_model_name)
        return _FakeTopClient()

    monkeypatch.setattr(ug, "client_from_model", _cap)
    await ug.update_gold_answers(
        db, query_ids=["q1"], models=make_models([1, 2, 3, 4, 5])
    )
    assert seen == ["tier5"]  # not tier1


@pytest.mark.asyncio
async def test_update_gold_reports_unknown_qids(tmp_path: Path, monkeypatch) -> None:
    db = bootstrap_db(tmp_path, QUERIES)
    _patch(monkeypatch, _FakeTopClient())
    result = await ug.update_gold_answers(
        db, query_ids=["q1", "q99999"], models=make_models([1, 2, 5])
    )
    assert result.updated == [("q1", "tier5")]
    assert result.skipped_unknown == ["q99999"]


@pytest.mark.asyncio
async def test_update_gold_errors_keep_old_gold(tmp_path: Path, monkeypatch) -> None:
    db = bootstrap_db(tmp_path, QUERIES)
    _patch(monkeypatch, _FakeTopClient(fail_on="entropy"))
    result = await ug.update_gold_answers(
        db, query_ids=["q1", "q2"], models=make_models([1, 5])
    )
    assert result.updated == [("q1", "tier5")]
    assert len(result.errors) == 1
    assert result.errors[0][0] == "q2:tier5"

    with session_scope(db) as s:
        # Failed regen must not create/clobber gold for q2.
        rows = s.execute(
            select(GoldAnswer).where(GoldAnswer.query_id == "q2")
        ).scalars().all()
        assert {r.model_id for r in rows} == {"upstream"}
        assert rows[0].answer == "old gold 2"


@pytest.mark.asyncio
async def test_update_gold_empty_response_is_error(tmp_path: Path, monkeypatch) -> None:
    db = bootstrap_db(tmp_path, QUERIES)
    _patch(monkeypatch, _FakeTopClient(empty_on="2+2"))
    result = await ug.update_gold_answers(
        db, query_ids=["q1"], models=make_models([1, 5])
    )
    assert result.updated == []
    assert len(result.errors) == 1
    assert "empty response" in result.errors[0][1]

    with session_scope(db) as s:
        rows = s.execute(
            select(GoldAnswer).where(GoldAnswer.query_id == "q1")
        ).scalars().all()
        assert {r.model_id for r in rows} == {"upstream"}  # no update-gold row
