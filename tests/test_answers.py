"""Per-tier answer collection tests with a fake OAIClient per tier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from benchmark.answers import run_answers
from benchmark.db import Pass1Result, TierAnswer, session_scope
from benchmark.runs import create_run, seed_pending_answers
from benchmark.tiers import ChatResult

from ._helpers import bootstrap_db, make_models, make_models_yaml, make_router_yaml

_models = make_models


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


def _record_pass1(db: Path, rid: int, query_id: str, tier_level: int) -> None:
    """Insert a successful pass1_results row so seed_pending_answers picks it up."""
    from datetime import UTC, datetime

    with session_scope(db) as s:
        s.add(
            Pass1Result(
                run_id=rid,
                query_id=query_id,
                router_selected_model=f"tier{tier_level}",
                router_selected_tier=tier_level,
                router_selected_specs=["general"],
                meets_minimum_tier=1,
                matches_specialization=1,
                latency_ms=10,
                raw_routing_metadata={"category": None, "reasoning": "off"},
                status="success",
                attempted_at=datetime.now(UTC),
            )
        )


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


# ---- seed_pending_answers ----

def test_seed_one_row_per_routed_query(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", tier_level=1)
    _record_pass1(db, rid, "q2", tier_level=3)
    models = _models([1, 2, 3])
    result = seed_pending_answers(db, rid, models)
    # 2 routed queries → 2 rows. qtts is excluded (TTS-only) and has no pass1 row.
    assert result.seeded == 2
    assert result.replaced == 0
    assert result.kept == 0
    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.run_id == rid)).scalars().all()
        assert {(r.query_id, r.tier_level) for r in rows} == {("q1", 1), ("q2", 3)}
        assert all(r.status == "pending" for r in rows)
        assert {r.tier_name for r in rows} == {"tier1", "tier3"}


def test_seed_skips_unrouted_queries(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", tier_level=1)
    # q2 has no pass1 row → not seeded.
    assert seed_pending_answers(db, rid, _models([1, 2, 3])).seeded == 1


def test_seed_idempotent_when_tier_unchanged(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", tier_level=1)
    models = _models([1, 2])
    r1 = seed_pending_answers(db, rid, models)
    assert r1.seeded == 1
    r2 = seed_pending_answers(db, rid, models)
    # Re-seeded, nothing changed: row was at tier 1, still at tier 1.
    assert r2.seeded == 0
    assert r2.replaced == 0
    assert r2.kept == 1


def test_seed_skips_tts_only(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "qtts", tier_level=2)
    assert seed_pending_answers(db, rid, _models([1, 2, 3])).seeded == 0


def test_seed_replaces_stale_tier(tmp_path: Path) -> None:
    """After `make route` re-runs and picks a different tier for a query,
    the next `make answers` should detect the mismatch and re-seed."""
    db, rid = _bootstrap(tmp_path)
    models = _models([1, 2, 3])

    # First round: router picked tier 1.
    _record_pass1(db, rid, "q1", tier_level=1)
    r1 = seed_pending_answers(db, rid, models)
    assert r1.seeded == 1

    # Router re-runs and now picks tier 3 for the same query. Update the
    # pass1 row to reflect that.
    with session_scope(db) as s:
        p1 = s.execute(
            select(Pass1Result).where(Pass1Result.query_id == "q1")
        ).scalar_one()
        p1.router_selected_tier = 3

    # Next seed call should DELETE the stale tier_answer (tier_level=1) and
    # insert a fresh pending one at tier_level=3.
    r2 = seed_pending_answers(db, rid, models)
    assert r2.replaced == 1
    assert r2.seeded == 0
    assert r2.kept == 0

    with session_scope(db) as s:
        rows = s.execute(
            select(TierAnswer).where(TierAnswer.query_id == "q1")
        ).scalars().all()
        # Exactly one row, at the new tier.
        assert len(rows) == 1
        assert rows[0].tier_level == 3
        assert rows[0].status == "pending"


# ---- run_answers ----

@pytest.mark.asyncio
async def test_run_answers_persists_routed_tier(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", tier_level=1)
    _record_pass1(db, rid, "q2", tier_level=3)
    models = _models([1, 2, 3])
    seed_pending_answers(db, rid, models)
    report = await run_answers(
        db, rid,
        models=models,
        clients_by_level=_clients([1, 2, 3]),
    )
    assert report.attempted == 2
    assert report.succeeded == 2
    assert report.errors == 0

    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.run_id == rid)).scalars().all()
        assert all(r.status == "success" for r in rows)
        for r in rows:
            expected_prompt = "easy" if r.query_id == "q1" else "harder"
            assert r.response_text == f"[tier{r.tier_level}] {expected_prompt}"


@pytest.mark.asyncio
async def test_run_answers_errors_dont_fail_the_pass(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", tier_level=1)
    _record_pass1(db, rid, "q2", tier_level=2)
    models = _models([1, 2])
    seed_pending_answers(db, rid, models)
    clients = _clients([1, 2], fail_on_query="harder")  # q2 (prompt "harder") fails at tier 2
    report = await run_answers(db, rid, models=models, clients_by_level=clients)
    assert report.succeeded == 1
    assert report.errors == 1
    qids = {qid for qid, _, _ in report.error_rows}
    assert qids == {"q2"}


@pytest.mark.asyncio
async def test_run_answers_resume_only_retries_errors(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", tier_level=1)
    _record_pass1(db, rid, "q2", tier_level=2)
    models = _models([1, 2])
    seed_pending_answers(db, rid, models)

    # First run: q2 errors.
    failing = _clients([1, 2], fail_on_query="harder")
    r1 = await run_answers(db, rid, models=models, clients_by_level=failing)
    assert r1.errors == 1

    # Resume with all-happy clients — should only re-run the 1 errored row.
    r2 = await run_answers(db, rid, models=models, clients_by_level=_clients([1, 2]))
    assert r2.attempted == 1
    assert r2.succeeded == 1
    assert r2.errors == 0


def test_build_clients_for_mock_uses_mock_url() -> None:
    """`_build_clients_for_mock` constructs one OAIClient per tier, all pointing
    at the mock URL, regardless of what each tier's configured endpoint is."""
    from benchmark.answers import _build_clients_for_mock

    models = _models([1, 2, 3])
    mock = "http://localhost:8811/v1"
    clients = _build_clients_for_mock(models, mock)

    assert set(clients.keys()) == {1, 2, 3}
    for level, client in clients.items():
        # OAIClient.endpoint strips trailing /. Check the start matches.
        assert client.endpoint == mock.rstrip("/"), (
            f"tier {level} client endpoint {client.endpoint!r} != mock {mock!r}"
        )


@pytest.mark.asyncio
async def test_run_answers_unknown_tier_recorded_as_error(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    # Router picked tier 9, which has no entry in our 2-tier model config.
    _record_pass1(db, rid, "q1", tier_level=9)
    seed_pending_answers(db, rid, _models([1, 2]))  # still seeds the row
    report = await run_answers(
        db, rid,
        models=_models([1, 2]),
        clients_by_level={1: _FakeClient(tier_level=1), 2: _FakeClient(tier_level=2)},
    )
    assert report.errors == 1
    assert all("level=9" in msg for _, _, msg in report.error_rows)
