"""Per-tier answer collection — backs `make answers`.

For each pending `tier_answers` row, look up the matching tier in
`models.yaml`, build an `OAIClient` against that tier's endpoint, and call
chat completions. This bypasses the router entirely — the router was
already asked (in `make route`) which tier it would pick; here we collect
EVERY tier's response so the export step can show "what would tier X have
said for this query."

Resumable: workers select rows where `status IN ('pending', 'error')`.
Per-row session commits make killing the process mid-run safe.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from .config import Attachment, ModelsConfig
from .db import Query, TierAnswer, session_scope
from .tiers import OAIClient, client_from_tier


@dataclass
class AnswersReport:
    attempted: int = 0
    succeeded: int = 0
    errors: int = 0
    error_rows: list[tuple[str, int, str]] = field(default_factory=list)  # (qid, tier_level, msg)

    def __str__(self) -> str:
        lines = [
            f"  attempted: {self.attempted}",
            f"  succeeded: {self.succeeded}",
            f"  errors:    {self.errors}",
        ]
        for qid, level, msg in self.error_rows[:10]:
            lines.append(f"    [tier {level}] {qid}: {msg}")
        if len(self.error_rows) > 10:
            lines.append(f"    ... and {len(self.error_rows) - 10} more")
        return "\n".join(lines)


def _build_clients_by_level(models: ModelsConfig) -> dict[int, OAIClient]:
    """One OAIClient per tier level. Reused across all queries for that tier."""
    out: dict[int, OAIClient] = {}
    for tier in models.tiers:
        # Later definitions for the same level win (matches by_name semantics).
        out[tier.level] = client_from_tier(tier)
    return out


async def run_answers(
    db_path: Path,
    run_id: int,
    *,
    models: ModelsConfig,
    concurrency: int = 8,
    max_tokens: int = 2048,
    clients_by_level: dict[int, OAIClient] | None = None,
) -> AnswersReport:
    """Process pending tier_answers rows. `clients_by_level` is injectable for tests."""
    if clients_by_level is None:
        clients_by_level = _build_clients_by_level(models)

    report = AnswersReport()

    # Snapshot the pending rows. We don't hold a DB session across the
    # async fan-out; each worker opens its own short-lived session.
    with session_scope(db_path) as session:
        rows = session.execute(
            select(TierAnswer, Query)
            .join(Query, TierAnswer.query_id == Query.query_id)
            .where(TierAnswer.run_id == run_id)
            .where(TierAnswer.status.in_(["pending", "error"]))
        ).all()
        snapshot = [
            {
                "query_id": ta.query_id,
                "tier_level": ta.tier_level,
                "prompt": q.prompt,
                "attachments": list(q.attachments or []),
            }
            for (ta, q) in rows
        ]

    sem = asyncio.Semaphore(concurrency)

    async def run_one(snap: dict) -> None:
        async with sem:
            qid = snap["query_id"]
            level = snap["tier_level"]
            attempted_at = datetime.now(UTC)
            client = clients_by_level.get(level)
            if client is None:
                report.errors += 1
                report.error_rows.append(
                    (qid, level, f"no tier with level={level} in models.yaml")
                )
                with session_scope(db_path) as session:
                    row = session.execute(
                        select(TierAnswer)
                        .where(TierAnswer.run_id == run_id)
                        .where(TierAnswer.query_id == qid)
                        .where(TierAnswer.tier_level == level)
                    ).scalar_one()
                    row.status = "error"
                    row.error_msg = f"no tier with level={level}"
                    row.attempted_at = attempted_at
                report.attempted += 1
                return

            try:
                attachments = [Attachment.model_validate(a) for a in snap["attachments"]]
                result = await client.chat(
                    snap["prompt"],
                    attachments=attachments,
                    max_tokens=max_tokens,
                )
                with session_scope(db_path) as session:
                    row = session.execute(
                        select(TierAnswer)
                        .where(TierAnswer.run_id == run_id)
                        .where(TierAnswer.query_id == qid)
                        .where(TierAnswer.tier_level == level)
                    ).scalar_one()
                    row.response_text = result.content
                    row.prompt_tokens = result.prompt_tokens
                    row.completion_tokens = result.completion_tokens
                    row.latency_ms = result.latency_ms
                    row.status = "success"
                    row.error_msg = None
                    row.attempted_at = attempted_at
                report.succeeded += 1
            except Exception as e:  # noqa: BLE001
                report.errors += 1
                report.error_rows.append((qid, level, f"{type(e).__name__}: {e}"))
                with session_scope(db_path) as session:
                    row = session.execute(
                        select(TierAnswer)
                        .where(TierAnswer.run_id == run_id)
                        .where(TierAnswer.query_id == qid)
                        .where(TierAnswer.tier_level == level)
                    ).scalar_one()
                    row.status = "error"
                    row.error_msg = f"{type(e).__name__}: {e}"
                    row.attempted_at = attempted_at
            finally:
                report.attempted += 1

    await asyncio.gather(*(run_one(s) for s in snapshot))
    return report
