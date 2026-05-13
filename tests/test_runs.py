"""Run lifecycle and pending-row seeding tests."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from benchmark.db import Pass1Result, Pass2Result, Run, session_scope
from benchmark.runs import (
    clean_results,
    create_run,
    latest_active_run,
    mark_finished,
    seed_pending,
)

from ._helpers import bootstrap_db, make_models_yaml, make_router_yaml

QUERIES = [
    {"id": "r001", "prompt": "trivial", "expected_min_tier": 1, "specializations": ["general"]},
    {"id": "r002", "prompt": "code task", "expected_min_tier": 2, "specializations": ["coding"]},
    {"id": "r003", "prompt": "tts task", "expected_min_tier": 3, "specializations": ["tts"]},
]


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    db = bootstrap_db(tmp_path, QUERIES)
    return db, make_router_yaml(tmp_path), make_models_yaml(tmp_path)


def test_create_and_finish(tmp_path: Path) -> None:
    db, r, m = _setup(tmp_path)
    rid = create_run(db, router_config_path=r, models_config_path=m)
    assert latest_active_run(db) == rid
    mark_finished(db, rid, status="done")
    assert latest_active_run(db) is None
    with session_scope(db) as s:
        run = s.execute(select(Run).where(Run.run_id == rid)).scalar_one()
        assert run.status == "done"
        assert run.finished_at is not None
        assert run.router_config_hash and run.models_config_hash


def test_seed_pending_idempotent(tmp_path: Path) -> None:
    db, r, m = _setup(tmp_path)
    rid = create_run(db, router_config_path=r, models_config_path=m)
    assert seed_pending(db, rid) == (3, 3)
    assert seed_pending(db, rid) == (0, 0)
    with session_scope(db) as s:
        p1 = len(
            s.execute(select(Pass1Result).where(Pass1Result.run_id == rid)).scalars().all()
        )
        p2 = len(
            s.execute(select(Pass2Result).where(Pass2Result.run_id == rid)).scalars().all()
        )
        assert p1 == 3 and p2 == 3


def test_seed_pending_only_filter(tmp_path: Path) -> None:
    db, r, m = _setup(tmp_path)
    rid = create_run(db, router_config_path=r, models_config_path=m)
    assert seed_pending(db, rid, only=["r001", "r002"]) == (2, 2)


def test_seed_pending_skips_tts_from_pass2(tmp_path: Path) -> None:
    db, r, m = _setup(tmp_path)
    rid = create_run(db, router_config_path=r, models_config_path=m)
    p1, p2 = seed_pending(db, rid, skip_query_ids={"r003"})
    assert p1 == 3 and p2 == 2
    with session_scope(db) as s:
        rows = s.execute(
            select(Pass2Result).where(Pass2Result.run_id == rid)
        ).scalars()
        assert {row.query_id for row in rows} == {"r001", "r002"}


def test_clean_results_preserves_queries(tmp_path: Path) -> None:
    db, r, m = _setup(tmp_path)
    rid = create_run(db, router_config_path=r, models_config_path=m)
    seed_pending(db, rid)
    deleted = clean_results(db)
    assert deleted["pass1_results"] == 3
    assert deleted["pass2_results"] == 3
    assert deleted["runs"] == 1
    with session_scope(db) as s:
        from benchmark.db import Query
        assert len(s.execute(select(Query)).scalars().all()) == 3
