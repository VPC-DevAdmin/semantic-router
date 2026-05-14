"""Run lifecycle and pending-row seeding.

Each invocation of a benchmark pass creates (or reuses) a row in `runs` with
the current config hashes. Per-pass result rows are seeded with
`status='pending'` so the worker can pick them up via a simple status
filter — that's what makes resume work without any extra coordination.

`make answers` semantics (post-refactor): exactly one TierAnswer row per
query, with `tier_level` set to the router's pick (from pass1_results).
Queries with no successful pass1 row are skipped at seeding time and will
be picked up on the next make answers after make route succeeds for them.
"""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class AnswerSeedResult:
    """Outcome of `seed_pending_answers`."""

    seeded: int = 0       # new tier_answers rows inserted
    replaced: int = 0     # stale rows deleted and re-seeded (tier mismatch with pass1)
    kept: int = 0         # rows left alone (already at the correct tier)

    def __int__(self) -> int:
        # Backward-compat: callers that expected just an int count still work.
        return self.seeded + self.replaced

    def __str__(self) -> str:
        parts = [f"seeded={self.seeded}"]
        if self.replaced:
            parts.append(f"replaced={self.replaced} (stale routing)")
        if self.kept:
            parts.append(f"kept={self.kept}")
        return ", ".join(parts)


def seed_pending_answers(
    db_path: Path,
    run_id: int,
    models: ModelsConfig,
    *,
    only: list[str] | None = None,
) -> AnswerSeedResult:
    """Seed one pending `tier_answers` row per query, using the routed tier.

    The routed tier is read from `pass1_results.router_selected_tier`. Queries
    without a successful pass1 row (or with a null routed tier) are skipped;
    they'll be picked up by a subsequent seed call after `make route` records
    their decision. TTS-only queries are excluded.

    Behaviour against existing rows:
      - Row exists at the same tier_level as current pass1 → kept (worker will
        retry if status is 'pending'/'error', leave if 'success').
      - Row exists at a DIFFERENT tier_level → stale: delete it and seed a
        fresh pending row at the correct level. This keeps tier_answers in
        sync with the latest routing decisions after a `make route` re-run.
      - No row exists → seed a new pending row.

    Returns an `AnswerSeedResult` with the per-bucket counts.
    """
    result = AnswerSeedResult()
    now = datetime.now(UTC)

    # tier_level → router_alias (used as tier_name when seeding).
    name_by_level: dict[int, str] = {t.level: t.router_alias for t in models.tiers}

    with session_scope(db_path) as session:
        stmt = select(Query.query_id, Query.specializations)
        if only:
            stmt = stmt.where(Query.query_id.in_(only))
        rows = session.execute(stmt).all()
        # Exclude TTS-only queries from text-answer collection.
        target_qids = {
            qid
            for qid, specs in rows
            if not (specs and all(s == "tts" for s in (specs or [])))
        }

        routed_by_qid: dict[str, int] = {}
        for qid, lvl in session.execute(
            select(Pass1Result.query_id, Pass1Result.router_selected_tier)
            .where(Pass1Result.run_id == run_id)
            .where(Pass1Result.status == "success")
        ).all():
            if qid in target_qids and lvl is not None:
                routed_by_qid[qid] = lvl

        # Existing tier_answer rows for this run, indexed by query_id.
        existing_by_qid: dict[str, TierAnswer] = {}
        for ta in session.execute(
            select(TierAnswer).where(TierAnswer.run_id == run_id)
        ).scalars():
            existing_by_qid[ta.query_id] = ta

        for qid, level in routed_by_qid.items():
            existing = existing_by_qid.get(qid)
            if existing is not None and existing.tier_level == level:
                # Already at the right tier; the answers worker handles it.
                result.kept += 1
                continue
            if existing is not None:
                # Stale: pass1 has since picked a different tier for this query.
                session.delete(existing)
                result.replaced += 1
            else:
                result.seeded += 1
            name = name_by_level.get(level, f"tier{level}")
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

    return result


def reset_pass1(db_path: Path, run_id: int) -> int:
    """Delete all pass1_results rows for `run_id`. Returns the count deleted.

    Used by `make route RUN_NEW=true` before re-seeding.
    """
    with session_scope(db_path) as session:
        deleted = (
            session.query(Pass1Result).filter(Pass1Result.run_id == run_id).delete()
        )
    return deleted


def reset_answers(db_path: Path, run_id: int) -> int:
    """Delete all tier_answers rows for `run_id`. Returns the count deleted.

    Used by `make answers RUN_NEW=true` before re-seeding.
    """
    with session_scope(db_path) as session:
        deleted = (
            session.query(TierAnswer).filter(TierAnswer.run_id == run_id).delete()
        )
    return deleted


def clean_results(db_path: Path) -> dict[str, int]:
    """Wipe runs and per-pass results. Preserves queries (with gold)."""
    with session_scope(db_path) as session:
        deleted = {
            "tier_answers": session.query(TierAnswer).delete(),
            "pass1_results": session.query(Pass1Result).delete(),
            "runs": session.query(Run).delete(),
        }
    return deleted
