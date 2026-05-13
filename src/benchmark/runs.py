"""Run lifecycle and pending-row seeding.

Each invocation of a benchmark pass creates (or reuses) a row in `runs` with
the current config hashes. Per-pass result rows are seeded with
`status='pending'` so the worker can pick them up via a simple status
filter — that's what makes resume work without any extra coordination.

Today we seed only `pass1_results`. When `make answers` lands it will add
a per-tier table (`tier_answers` per PLAN.md §10) with its own seeding
logic.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from .config import ModelsConfig, hash_file
from .db import Pass1Result, Query, Run, TierAnswer, session_scope


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
) -> int:
    """Seed pending `pass1_results` rows for every query (or a filtered subset).

    Returns the number of rows seeded. Idempotent — if a row already exists
    for `(run_id, query_id)`, it is left alone.
    """
    seeded = 0
    now = datetime.now(UTC)

    with session_scope(db_path) as session:
        stmt = select(Query.query_id)
        if only:
            stmt = stmt.where(Query.query_id.in_(only))
        all_qids = [r[0] for r in session.execute(stmt).all()]

        existing = {
            r[0]
            for r in session.execute(
                select(Pass1Result.query_id).where(Pass1Result.run_id == run_id)
            ).all()
        }

        for qid in all_qids:
            if qid in existing:
                continue
            session.add(
                Pass1Result(
                    run_id=run_id,
                    query_id=qid,
                    status="pending",
                    attempted_at=now,
                )
            )
            seeded += 1

    return seeded


def seed_pending_tiers(
    db_path: Path,
    run_id: int,
    models: ModelsConfig,
    *,
    only: list[str] | None = None,
) -> int:
    """Seed pending `tier_answers` rows: one per (query, tier_level).

    Returns the number of rows seeded. Idempotent — rows that already exist
    for `(run_id, query_id, tier_level)` are left alone. Skips TTS-only
    queries (text gold doesn't apply to audio output).

    Tiers come from `models.yaml`. A tier with `level=N` and `model_id=X`
    contributes one row per query with `tier_level=N` and `tier_name=X`.
    """
    seeded = 0
    now = datetime.now(UTC)

    # Build the canonical tier set: (level, model_id) pairs from models.yaml.
    # If two tiers share the same level, the later one wins — matches the
    # ModelsConfig.by_name semantics.
    tier_pairs: dict[int, str] = {}
    for tier in models.tiers:
        tier_pairs[tier.level] = tier.model_id

    with session_scope(db_path) as session:
        stmt = select(Query.query_id, Query.specializations)
        if only:
            stmt = stmt.where(Query.query_id.in_(only))
        rows = session.execute(stmt).all()
        # Exclude TTS-only queries from per-tier text answer collection.
        target_qids = [
            qid
            for qid, specs in rows
            if not (specs and all(s == "tts" for s in (specs or [])))
        ]

        existing = {
            (r[0], r[1])
            for r in session.execute(
                select(TierAnswer.query_id, TierAnswer.tier_level)
                .where(TierAnswer.run_id == run_id)
            ).all()
        }

        for qid in target_qids:
            for level, name in tier_pairs.items():
                if (qid, level) in existing:
                    continue
                session.add(
                    TierAnswer(
                        run_id=run_id,
                        query_id=qid,
                        tier_level=level,
                        tier_name=name,
                        status="pending",
                        attempted_at=now,
                    )
                )
                seeded += 1

    return seeded


def clean_results(db_path: Path) -> dict[str, int]:
    """Wipe runs and per-pass results. Preserves queries (with gold)."""
    with session_scope(db_path) as session:
        deleted = {
            "tier_answers": session.query(TierAnswer).delete(),
            "pass1_results": session.query(Pass1Result).delete(),
            "runs": session.query(Run).delete(),
        }
    return deleted
