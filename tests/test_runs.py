"""Run lifecycle and pending-row seeding tests."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from benchmark.db import Pass1Result, Pass2Result, Run, init_db, session_scope
from benchmark.runs import (
    clean_results,
    create_run,
    latest_active_run,
    mark_finished,
    seed_pending,
)
from benchmark.seed import seed_from_yaml

QUERIES = """
- id: r001
  prompt: "trivial"
  expected_min_tier: 1
  specializations: [general]
- id: r002
  prompt: "code task"
  expected_min_tier: 2
  specializations: [code]
- id: r003
  prompt: "tts task"
  expected_min_tier: 3
  specializations: [tts]
"""


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    db_path = tmp_path / "t.db"
    yaml_path = tmp_path / "queries.yaml"
    yaml_path.write_text(QUERIES)
    init_db(db_path)
    seed_from_yaml(yaml_path, db_path)

    router_yaml = tmp_path / "router.yaml"
    router_yaml.write_text("placeholder: true\n")
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        "tiers:\n"
        "  - name: t\n"
        "    level: 1\n"
        "    endpoint: x\n"
        "    model_id: m\n"
        "    specializations: [general]\n"
    )
    return db_path, router_yaml, models_yaml


def test_create_and_finish(tmp_path: Path) -> None:
    db_path, r_yaml, m_yaml = _setup(tmp_path)
    rid = create_run(db_path, router_config_path=r_yaml, models_config_path=m_yaml)
    assert latest_active_run(db_path) == rid
    mark_finished(db_path, rid, status="done")
    assert latest_active_run(db_path) is None
    with session_scope(db_path) as s:
        run = s.execute(select(Run).where(Run.run_id == rid)).scalar_one()
        assert run.status == "done"
        assert run.finished_at is not None
        assert run.router_config_hash and run.models_config_hash


def test_seed_pending_idempotent(tmp_path: Path) -> None:
    db_path, r_yaml, m_yaml = _setup(tmp_path)
    rid = create_run(db_path, router_config_path=r_yaml, models_config_path=m_yaml)

    p1, p2 = seed_pending(db_path, rid)
    assert p1 == 3
    assert p2 == 3

    # Second invocation should not seed any more.
    p1_again, p2_again = seed_pending(db_path, rid)
    assert (p1_again, p2_again) == (0, 0)

    with session_scope(db_path) as s:
        p1_count = len(
            s.execute(select(Pass1Result).where(Pass1Result.run_id == rid)).scalars().all()
        )
        p2_count = len(
            s.execute(select(Pass2Result).where(Pass2Result.run_id == rid)).scalars().all()
        )
        assert p1_count == 3
        assert p2_count == 3


def test_seed_pending_only_filter(tmp_path: Path) -> None:
    db_path, r_yaml, m_yaml = _setup(tmp_path)
    rid = create_run(db_path, router_config_path=r_yaml, models_config_path=m_yaml)
    p1, p2 = seed_pending(db_path, rid, only=["r001", "r002"])
    assert p1 == 2
    assert p2 == 2


def test_seed_pending_skips_tts_from_pass2(tmp_path: Path) -> None:
    db_path, r_yaml, m_yaml = _setup(tmp_path)
    rid = create_run(db_path, router_config_path=r_yaml, models_config_path=m_yaml)
    p1, p2 = seed_pending(db_path, rid, skip_query_ids={"r003"})
    assert p1 == 3   # pass1 still gets all
    assert p2 == 2   # pass2 skips tts

    with session_scope(db_path) as s:
        rows = s.execute(
            select(Pass2Result).where(Pass2Result.run_id == rid)
        ).scalars()
        p2_qids = {r.query_id for r in rows}
        assert p2_qids == {"r001", "r002"}


def test_clean_results_preserves_queries(tmp_path: Path) -> None:
    db_path, r_yaml, m_yaml = _setup(tmp_path)
    rid = create_run(db_path, router_config_path=r_yaml, models_config_path=m_yaml)
    seed_pending(db_path, rid)

    deleted = clean_results(db_path)
    assert deleted["pass1_results"] == 3
    assert deleted["pass2_results"] == 3
    assert deleted["runs"] == 1

    with session_scope(db_path) as s:
        # Queries still there.
        from benchmark.db import Query
        assert len(s.execute(select(Query)).scalars().all()) == 3
        # Runs gone.
        assert s.execute(select(Run)).scalars().all() == []
