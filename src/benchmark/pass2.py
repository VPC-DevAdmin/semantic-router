"""Pass 2 — Response quality.

For each pending pass2_result row, send the query to the router and persist
the full generation, token counts, and latency. Scoring is a separate phase
(M5).

TTS-only queries are skipped at seed time (no text gold means no text score).
This module just processes whatever rows the seeder created.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from .config import Attachment
from .db import Pass2Result, Query, session_scope
from .router_client import RouterClient


@dataclass
class Pass2Report:
    attempted: int = 0
    succeeded: int = 0
    errors: int = 0
    error_ids: list[tuple[str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"  attempted: {self.attempted}",
            f"  succeeded: {self.succeeded}",
            f"  errors:    {self.errors}",
        ]
        for qid, msg in self.error_ids[:10]:
            lines.append(f"    {qid}: {msg}")
        if len(self.error_ids) > 10:
            lines.append(f"    ... and {len(self.error_ids) - 10} more")
        return "\n".join(lines)


async def run_pass2(
    db_path: Path,
    run_id: int,
    *,
    router_client: RouterClient,
    concurrency: int = 8,
    max_tokens: int | None = None,
) -> Pass2Report:
    report = Pass2Report()

    with session_scope(db_path) as session:
        rows = session.execute(
            select(Pass2Result, Query)
            .join(Query, Pass2Result.query_id == Query.query_id)
            .where(Pass2Result.run_id == run_id)
            .where(Pass2Result.status.in_(["pending", "error"]))
        ).all()
        snapshot = [
            {
                "query_id": p.query_id,
                "prompt": q.prompt,
                "attachments": list(q.attachments or []),
            }
            for (p, q) in rows
        ]

    sem = asyncio.Semaphore(concurrency)

    async def run_one(snap: dict) -> None:
        async with sem:
            qid = snap["query_id"]
            attempted_at = datetime.now(UTC)
            try:
                attachments = [Attachment.model_validate(a) for a in snap["attachments"]]
                result = await router_client.chat(
                    snap["prompt"],
                    attachments=attachments,
                    max_tokens=max_tokens,
                )
                with session_scope(db_path) as session:
                    row = session.execute(
                        select(Pass2Result)
                        .where(Pass2Result.run_id == run_id)
                        .where(Pass2Result.query_id == qid)
                    ).scalar_one()
                    row.router_selected_model = result.decision.selected_model
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
                report.error_ids.append((qid, f"{type(e).__name__}: {e}"))
                with session_scope(db_path) as session:
                    row = session.execute(
                        select(Pass2Result)
                        .where(Pass2Result.run_id == run_id)
                        .where(Pass2Result.query_id == qid)
                    ).scalar_one()
                    row.status = "error"
                    row.error_msg = f"{type(e).__name__}: {e}"
                    row.attempted_at = attempted_at
            finally:
                report.attempted += 1

    await asyncio.gather(*(run_one(s) for s in snapshot))
    return report
