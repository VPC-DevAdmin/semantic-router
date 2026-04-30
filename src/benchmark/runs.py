"""Run lifecycle and pending-row seeding.

Each invocation of `make run` creates a new row in `runs` with the current
config hashes. Per-pass result rows are seeded with `status='pending'` so the
worker can pick them up via a simple status filter — that's what makes resume
work without any extra coordination.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from .config import hash_file
from .db import Pass1Result, Pass2Result, Query, Run, session_scope


def create_run(
    db_path: Path,
    *,
    router_config_path: Path,
    models_config_path: Path,
    notes: str | None = None,
) -> int:
    with session_scope(db_path) as session:
        run = Run(
            started_at=datetime.now(UTC),
            router_config_hash=hash_file(router_config_path),
            models_config_hash=hash_file(models_config_path),
            notes=notes,
            status="running",
        )
        session.add(run)
        session.flush()  # populate run_id
        return run.run_id


def mark_finished(db_path: Path, run_id: int, *, status: str = "done") -> None:
    with session_scope(db_path) as session:
        run = session.execute(select(Run).where(Run.run_id == run_id)).scalar_one()
        run.finished_at = datetime.now(UTC)
        run.status = status


def latest_active_run(db_path: Path) -> int | None:
    with session_scope(db_path) as session:
        row = session.execute(
            select(Run).where(Run.status == "running").order_by(Run.run_id.desc())
        ).scalars().first()
        return None if row is None else row.run_id


def get_run_status(db_path: Path, run_id: int) -> str:
    with session_scope(db_path) as session:
        run = session.execute(select(Run).where(Run.run_id == run_id)).scalar_one()
        return run.status


def seed_pending(
    db_path: Path,
    run_id: int,
    *,
    only: list[str] | None = None,
    skip_query_ids: set[str] | None = None,
) -> tuple[int, int]:
    """Seed pending pass1 + pass2 rows for every query (or a filtered subset).

    Returns (pass1_seeded, pass2_seeded). Idempotent — if rows already exist
    for this (run_id, query_id), they are left alone.

    `skip_query_ids` lets pass2 callers exclude TTS-only queries (those have
    no text gold and so no pass2). Pass 1 still runs against all queries.
    """
    skip_query_ids = skip_query_ids or set()
    pass1_seeded = 0
    pass2_seeded = 0
    now = datetime.now(UTC)

    with session_scope(db_path) as session:
        stmt = select(Query.query_id)
        if only:
            stmt = stmt.where(Query.query_id.in_(only))
        all_qids = [r[0] for r in session.execute(stmt).all()]

        existing_p1 = {
            r[0]
            for r in session.execute(
                select(Pass1Result.query_id).where(Pass1Result.run_id == run_id)
            ).all()
        }
        existing_p2 = {
            r[0]
            for r in session.execute(
                select(Pass2Result.query_id).where(Pass2Result.run_id == run_id)
            ).all()
        }

        for qid in all_qids:
            if qid not in existing_p1:
                session.add(
                    Pass1Result(
                        run_id=run_id,
                        query_id=qid,
                        status="pending",
                        attempted_at=now,
                    )
                )
                pass1_seeded += 1
            if qid not in existing_p2 and qid not in skip_query_ids:
                session.add(
                    Pass2Result(
                        run_id=run_id,
                        query_id=qid,
                        status="pending",
                        attempted_at=now,
                    )
                )
                pass2_seeded += 1

    return pass1_seeded, pass2_seeded


def clean_results(db_path: Path) -> dict[str, int]:
    """Wipe runs, results, scores. Preserves queries (with gold)."""
    with session_scope(db_path) as session:
        from .db import Pass1Result, Pass2Result, Run, Score

        deleted = {
            "scores": session.query(Score).delete(),
            "pass2_results": session.query(Pass2Result).delete(),
            "pass1_results": session.query(Pass1Result).delete(),
            "runs": session.query(Run).delete(),
        }
    return deleted
