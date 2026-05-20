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
    """Outcome of `seed_pending_answers` (counts are per ROW = per model)."""

    seeded: int = 0    # new pending tier_answers rows inserted (needs a call)
    replaced: int = 0  # stale rows deleted (wrong tier, or a model no longer
                       # configured for the routed tier)
    kept: int = 0      # rows left alone (already at the right tier+model)
    skipped_top_tier: int = 0  # queries routed to the top tier — NOT called
                               # by make answers (the top tier IS the gold;
                               # its per-provider answers come from
                               # update-gold / upstream, never make answers)

    def __int__(self) -> int:
        # Backward-compat: callers that expected just an int count still work.
        return self.seeded + self.replaced

    def __str__(self) -> str:
        parts = [f"seeded={self.seeded}"]
        if self.replaced:
            parts.append(f"replaced={self.replaced} (stale tier/model)")
        if self.kept:
            parts.append(f"kept={self.kept}")
        if self.skipped_top_tier:
            parts.append(f"skipped_top_tier={self.skipped_top_tier}")
        return ", ".join(parts)


def seed_pending_answers(
    db_path: Path,
    run_id: int,
    models: ModelsConfig,
    *,
    only: list[str] | None = None,
) -> AnswerSeedResult:
    """Seed `tier_answers` rows for every (query, routed tier, model).

    The routed tier is read from `pass1_results.router_selected_tier`.
    The router picks ONE tier; we then call EVERY model that tier fronts
    (Anthropic / OpenAI / Google …), so a routed query gets one pending
    row per model in its tier — that's how the demo shows "your answer on
    OpenAI vs Anthropic". Queries without a successful pass1 row (or a
    null routed tier) are skipped until `make route` records a decision.
    TTS-only queries are excluded.

    TOP-TIER SHORTCUT: the top tier IS the gold/reference tier — every
    comparison in the demo is "routed answer vs. the top-tier expected
    answer", never the top tier against itself. So a query the router
    sent to the top tier needs NO model calls here: its per-provider
    answers are the gold, produced separately by `make update-gold`
    (which calls every top-tier model) and the `expected_answers[]`
    declared in queries.json (seeded into gold_answers at `make load`).
    Such queries are skipped (and any stale top-tier rows for them are
    deleted).

    Row reconciliation for non-top tiers (per row = per model):
      - A row already at the right (tier_level, model_id) → kept.
      - A row at a different tier_level, or for a model no longer
        configured for the routed tier → stale: deleted (replaced).
      - A wanted (tier_level, model_id) with no row → seeded (pending).
    """
    result = AnswerSeedResult()
    now = datetime.now(UTC)
    top_tier_level = max((t.level for t in models.tiers), default=0)

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

        # Existing tier_answer rows for this run, grouped by query_id.
        existing_by_qid: dict[str, list[TierAnswer]] = {}
        for ta in session.execute(
            select(TierAnswer).where(TierAnswer.run_id == run_id)
        ).scalars():
            existing_by_qid.setdefault(ta.query_id, []).append(ta)

        for qid, level in routed_by_qid.items():
            if top_tier_level and level == top_tier_level:
                # Routed to the gold tier → make answers does NOT call it.
                # Drop any stale rows that may exist for this query.
                for ta in existing_by_qid.get(qid, []):
                    session.delete(ta)
                    result.replaced += 1
                result.skipped_top_tier += 1
                continue
            try:
                tier = models.by_level(level)
            except KeyError:
                # No tier config for this level — can't seed model rows.
                continue
            wanted = {m.served_model_name: m for m in tier.resolved_models()}
            present = existing_by_qid.get(qid, [])

            # Keep rows that match a wanted (level, model); delete the rest.
            keep_keys: set[str] = set()
            for ta in present:
                if ta.tier_level == level and ta.model_id in wanted:
                    result.kept += 1
                    keep_keys.add(ta.model_id)
                else:
                    session.delete(ta)  # wrong tier, or model dropped
                    result.replaced += 1

            for mid, m in wanted.items():
                if mid in keep_keys:
                    continue
                session.add(
                    TierAnswer(
                        run_id=run_id,
                        query_id=qid,
                        tier_level=level,
                        model_id=mid,
                        model_slot=m.slot,
                        provider=m.provider,
                        tier_name=tier.router_alias,
                        status="pending",
                        attempted_at=now,
                    )
                )
                result.seeded += 1

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


def reset_answers(db_path: Path, run_id: int, *, tier_level: int | None = None) -> int:
    """Delete tier_answers rows for `run_id`. Returns the count deleted.

    Used by `make answers RUN_NEW=true` before re-seeding. With
    `tier_level` set (e.g. `make answers TIER=1 RUN_NEW=true`), only
    that tier's rows are deleted — leaves other tiers' state intact so
    a partial re-run of one tier doesn't trash everything.
    """
    with session_scope(db_path) as session:
        q = session.query(TierAnswer).filter(TierAnswer.run_id == run_id)
        if tier_level is not None:
            q = q.filter(TierAnswer.tier_level == tier_level)
        deleted = q.delete()
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
