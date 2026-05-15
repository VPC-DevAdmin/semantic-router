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

    seeded: int = 0       # new tier_answers rows inserted (pending — needs LLM)
    replaced: int = 0     # stale rows deleted and re-seeded (tier mismatch with pass1)
    kept: int = 0         # rows left alone (already at the correct tier)
    gold_filled: int = 0  # top-tier rows filled from the gold answer (no LLM call)

    def __int__(self) -> int:
        # Backward-compat: callers that expected just an int count still work.
        return self.seeded + self.replaced

    def __str__(self) -> str:
        parts = [f"seeded={self.seeded}"]
        if self.replaced:
            parts.append(f"replaced={self.replaced} (stale routing)")
        if self.gold_filled:
            parts.append(f"gold_filled={self.gold_filled} (top tier == gold)")
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
    """Seed one `tier_answers` row per query, using the routed tier.

    The routed tier is read from `pass1_results.router_selected_tier`. Queries
    without a successful pass1 row (or with a null routed tier) are skipped;
    they'll be picked up by a subsequent seed call after `make route` records
    their decision. TTS-only queries are excluded.

    TOP-TIER SHORTCUT: the query set's `expected_answer` (→ `Query.gold_answer`)
    is an upstream Opus-grade reference. The top tier in our routing IS
    Claude Opus. So for a query routed to the top tier, calling the LLM
    just regenerates an Opus answer that's redundant with the gold we
    already have — wasted API spend. Such rows are filled directly from
    `gold_answer` with status='success', so `make answers` skips them.
    Queries with no/empty gold fall back to a normal pending row. (To
    *regenerate* the gold itself, use the `update-gold` command, which
    calls the top tier and overwrites `Query.gold_answer`.)

    Behaviour against existing rows:
      - Row exists at the same tier_level as current pass1 → kept (but a
        pending top-tier row is upgraded to gold-success).
      - Row exists at a DIFFERENT tier_level → stale: delete it and seed a
        fresh row at the correct level.
      - No row exists → seed a new row (pending, or gold-success for top tier).

    Returns an `AnswerSeedResult` with the per-bucket counts.
    """
    result = AnswerSeedResult()
    now = datetime.now(UTC)

    # tier_level → router_alias (used as tier_name when seeding).
    name_by_level: dict[int, str] = {t.level: t.router_alias for t in models.tiers}
    top_tier_level = max((t.level for t in models.tiers), default=0)

    with session_scope(db_path) as session:
        stmt = select(Query.query_id, Query.specializations, Query.gold_answer)
        if only:
            stmt = stmt.where(Query.query_id.in_(only))
        rows = session.execute(stmt).all()
        # Exclude TTS-only queries from text-answer collection.
        target_qids = {
            qid
            for qid, specs, _gold in rows
            if not (specs and all(s == "tts" for s in (specs or [])))
        }
        gold_by_qid: dict[str, str | None] = {
            qid: gold for qid, _specs, gold in rows
        }

        def _is_gold_top_tier(qid: str, level: int) -> bool:
            return level == top_tier_level and bool(gold_by_qid.get(qid))

        def _new_row(qid: str, level: int) -> TierAnswer:
            """Build a row WITHOUT touching counters (caller counts)."""
            name = name_by_level.get(level, f"tier{level}")
            if _is_gold_top_tier(qid, level):
                return TierAnswer(
                    run_id=run_id,
                    query_id=qid,
                    tier_level=level,
                    tier_name=name,
                    response_text=gold_by_qid[qid],
                    status="success",
                    attempted_at=now,
                )
            return TierAnswer(
                run_id=run_id,
                query_id=qid,
                tier_level=level,
                tier_name=name,
                status="pending",
                attempted_at=now,
            )

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
            is_gold = _is_gold_top_tier(qid, level)

            if existing is not None and existing.tier_level == level:
                # Right tier already. Upgrade a not-yet-successful top-tier
                # row to gold-success so the worker doesn't call the LLM.
                if existing.status != "success" and is_gold:
                    existing.response_text = gold_by_qid[qid]
                    existing.status = "success"
                    existing.error_msg = None
                    existing.attempted_at = now
                    result.gold_filled += 1
                else:
                    result.kept += 1
                continue

            if existing is not None:
                # Stale: pass1 has since picked a different tier for this query.
                session.delete(existing)
                # Count by what the REPLACEMENT is: a gold-filled row didn't
                # need an LLM; a pending row did. Either way the stale row
                # was fixed — but gold_filled is the more salient fact.
                if is_gold:
                    result.gold_filled += 1
                else:
                    result.replaced += 1
            elif is_gold:
                result.gold_filled += 1
            else:
                result.seeded += 1

            session.add(_new_row(qid, level))

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
