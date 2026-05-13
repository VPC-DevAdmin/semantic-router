"""Per-tier answer collection tests with a fake OAIClient per tier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from benchmark.answers import run_answers
from benchmark.config import ModelsConfig, TierConfig
from benchmark.db import TierAnswer, session_scope
from benchmark.runs import create_run, seed_pending_tiers
from benchmark.tiers import ChatResult

from ._helpers import bootstrap_db, make_models_yaml, make_router_yaml


def _models(levels: list[int]) -> ModelsConfig:
    return ModelsConfig(
        tiers=[
            TierConfig(
                name=f"tier{lvl}",
                level=lvl,
                endpoint=f"http://localhost:880{lvl}/v1",
                model_id=f"tier{lvl}",
                api_key_env=None,
                specializations=["general"],
            )
            for lvl in levels
        ]
    )


QUERIES = [
    {"id": "q1", "prompt": "easy", "expected_min_tier": 1, "specializations": ["general"]},
    {
        "id": "q2", "prompt": "harder", "expected_min_tier": 3,
        "specializations": ["coding"],
    },
    {
        "id": "qtts", "prompt": "say it", "expected_min_tier": 2,
        "specializations": ["tts"],
    },
]


def _bootstrap(tmp_path: Path) -> tuple[Path, int]:
    db = bootstrap_db(tmp_path, QUERIES)
    rid = create_run(
        db,
        router_config_path=make_router_yaml(tmp_path),
        models_config_path=make_models_yaml(tmp_path),
    )
    return db, rid


@dataclass
class _FakeClient:
    """Echoes a deterministic per-tier response so we can assert routing."""

    tier_level: int
    fail_on_query: str | None = None

    async def chat(self, prompt: str, *, attachments=None, max_tokens=None, **_: Any) -> ChatResult:
        if self.fail_on_query and self.fail_on_query in prompt:
            raise RuntimeError(f"tier{self.tier_level} unhappy with {prompt!r}")
        return ChatResult(
            content=f"[tier{self.tier_level}] {prompt}",
            model=f"tier{self.tier_level}",
            prompt_tokens=5,
            completion_tokens=10,
            latency_ms=10 * self.tier_level,
            raw={},
        )


def _clients(levels: list[int], fail_on_query: str | None = None) -> dict[int, _FakeClient]:
    return {lvl: _FakeClient(tier_level=lvl, fail_on_query=fail_on_query) for lvl in levels}


# ---- seed_pending_tiers ----

def test_seed_pending_tiers_one_row_per_query_per_tier(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    models = _models([1, 2, 3])
    seeded = seed_pending_tiers(db, rid, models)
    # 2 non-TTS queries × 3 tiers = 6 rows. qtts is excluded.
    assert seeded == 6
    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.run_id == rid)).scalars().all()
        assert len(rows) == 6
        assert {(r.query_id, r.tier_level) for r in rows} == {
            ("q1", 1), ("q1", 2), ("q1", 3),
            ("q2", 1), ("q2", 2), ("q2", 3),
        }
        assert all(r.status == "pending" for r in rows)
        # tier_name is populated from models.yaml model_id.
        assert {r.tier_name for r in rows} == {"tier1", "tier2", "tier3"}


def test_seed_pending_tiers_idempotent(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    models = _models([1, 2])
    assert seed_pending_tiers(db, rid, models) == 4  # 2 queries × 2 tiers
    assert seed_pending_tiers(db, rid, models) == 0  # no new rows


def test_seed_pending_tiers_only_filter(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    models = _models([1, 2])
    assert seed_pending_tiers(db, rid, models, only=["q1"]) == 2


def test_seed_pending_tiers_skips_tts_only(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    models = _models([1, 2, 3])
    seed_pending_tiers(db, rid, models)
    with session_scope(db) as s:
        qids = {
            r.query_id
            for r in s.execute(select(TierAnswer).where(TierAnswer.run_id == rid)).scalars()
        }
        assert "qtts" not in qids


# ---- run_answers ----

@pytest.mark.asyncio
async def test_run_answers_persists_each_tier(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    models = _models([1, 2, 3])
    seed_pending_tiers(db, rid, models)
    report = await run_answers(
        db, rid,
        models=models,
        clients_by_level=_clients([1, 2, 3]),
    )
    assert report.attempted == 6
    assert report.succeeded == 6
    assert report.errors == 0

    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.run_id == rid)).scalars().all()
        assert all(r.status == "success" for r in rows)
        # Each row's response_text identifies the tier that produced it.
        for r in rows:
            expected_prompt = "easy" if r.query_id == "q1" else "harder"
            assert r.response_text == f"[tier{r.tier_level}] {expected_prompt}"
            assert r.latency_ms == 10 * r.tier_level


@pytest.mark.asyncio
async def test_run_answers_error_per_tier_is_isolated(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    models = _models([1, 2])
    seed_pending_tiers(db, rid, models)
    clients = _clients([1, 2], fail_on_query="harder")
    report = await run_answers(db, rid, models=models, clients_by_level=clients)
    # q1 succeeds at both tiers; q2 fails at both tiers.
    assert report.succeeded == 2
    assert report.errors == 2
    qids = {qid for qid, _, _ in report.error_rows}
    assert qids == {"q2"}


@pytest.mark.asyncio
async def test_run_answers_resume_skips_success(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    models = _models([1, 2])
    seed_pending_tiers(db, rid, models)

    # First run: tier 2 errors for both queries.
    failing = {1: _FakeClient(tier_level=1), 2: _FakeClient(tier_level=2, fail_on_query="")}
    failing[2].fail_on_query = "easy"  # only q1 fails at tier 2
    r1 = await run_answers(db, rid, models=models, clients_by_level=failing)
    assert r1.errors == 1

    # Resume with all-happy clients — should only re-run the 1 errored row.
    r2 = await run_answers(db, rid, models=models, clients_by_level=_clients([1, 2]))
    assert r2.attempted == 1
    assert r2.succeeded == 1
    assert r2.errors == 0


@pytest.mark.asyncio
async def test_run_answers_unknown_tier_recorded_as_error(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    # Seed expects tiers 1+2 (so 4 rows), but the clients map only has tier 1.
    seed_pending_tiers(db, rid, _models([1, 2]))
    clients = {1: _FakeClient(tier_level=1)}  # tier 2 missing
    report = await run_answers(
        db, rid,
        models=_models([1, 2]),
        clients_by_level=clients,
    )
    assert report.succeeded == 2  # both queries succeed at tier 1
    assert report.errors == 2  # both fail at tier 2 with "no tier with level=2"
    assert all("level=2" in msg for _, _, msg in report.error_rows)
