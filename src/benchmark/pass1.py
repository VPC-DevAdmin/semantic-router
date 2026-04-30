"""Pass 1 — Routing accuracy.

For each query in the active run that has a pending or errored pass1_result,
send it to the router with `max_tokens=1` (we only care about the routing
decision, not the generation), then compute:

  - meets_minimum_tier   = router_selected_tier >= query.expected_min_tier
  - matches_specialization = expected_specs ⊆ router_selected_specs

Rows are committed individually so killing the process mid-run is safe; the
next invocation picks up where it stopped.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from .config import Attachment
from .db import Pass1Result, Query, session_scope
from .router_client import RouterClient


@dataclass
class Pass1Report:
    attempted: int = 0
    succeeded: int = 0
    errors: int = 0
    meets_min_tier: int = 0
    matches_spec: int = 0
    unknown_tier: int = 0  # router selected a model not in models.yaml
    error_ids: list[tuple[str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"  attempted:       {self.attempted}",
            f"  succeeded:       {self.succeeded}",
            f"  errors:          {self.errors}",
            f"  meets min tier:  {self.meets_min_tier}/{self.succeeded}",
            f"  matches spec:    {self.matches_spec}/{self.succeeded}",
            f"  unknown tier:    {self.unknown_tier}",
        ]
        for qid, msg in self.error_ids[:10]:
            lines.append(f"    {qid}: {msg}")
        if len(self.error_ids) > 10:
            lines.append(f"    ... and {len(self.error_ids) - 10} more")
        return "\n".join(lines)


def _meets_min_tier(selected_tier: int | None, expected_min_tier: int) -> int | None:
    if selected_tier is None:
        return None
    return 1 if selected_tier >= expected_min_tier else 0


def _matches_spec(
    selected_specs: list[str] | None, expected_specs: list[str]
) -> int | None:
    if selected_specs is None:
        return None
    sel = set(selected_specs)
    return 1 if all(s in sel for s in expected_specs) else 0


async def run_pass1(
    db_path: Path,
    run_id: int,
    *,
    router_client: RouterClient,
    concurrency: int = 8,
    max_tokens: int = 1,
) -> Pass1Report:
    report = Pass1Report()

    # Load pending/error rows + their query metadata. Detach from session
    # before the async fan-out so each worker manages its own session.
    with session_scope(db_path) as session:
        rows = session.execute(
            select(Pass1Result, Query)
            .join(Query, Pass1Result.query_id == Query.query_id)
            .where(Pass1Result.run_id == run_id)
            .where(Pass1Result.status.in_(["pending", "error"]))
        ).all()
        snapshot = [
            {
                "query_id": p.query_id,
                "prompt": q.prompt,
                "attachments": list(q.attachments or []),
                "expected_min_tier": q.expected_min_tier,
                "expected_specs": list(q.specializations or []),
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
                d = result.decision
                meets = _meets_min_tier(d.selected_tier, snap["expected_min_tier"])
                matches = _matches_spec(d.selected_specs, snap["expected_specs"])

                with session_scope(db_path) as session:
                    row = session.execute(
                        select(Pass1Result)
                        .where(Pass1Result.run_id == run_id)
                        .where(Pass1Result.query_id == qid)
                    ).scalar_one()
                    row.router_selected_model = d.selected_model
                    row.router_selected_tier = d.selected_tier
                    row.router_selected_specs = d.selected_specs
                    row.meets_minimum_tier = meets
                    row.matches_specialization = matches
                    row.latency_ms = result.latency_ms
                    row.raw_routing_metadata = {
                        "category": d.category,
                        "reasoning": d.reasoning,
                        "cache_hit": d.cache_hit,
                        "headers": result.raw_headers,
                    }
                    row.status = "success"
                    row.error_msg = None
                    row.attempted_at = attempted_at

                report.succeeded += 1
                if meets == 1:
                    report.meets_min_tier += 1
                if matches == 1:
                    report.matches_spec += 1
                if d.selected_tier is None:
                    report.unknown_tier += 1
            except Exception as e:  # noqa: BLE001
                report.errors += 1
                report.error_ids.append((qid, f"{type(e).__name__}: {e}"))
                with session_scope(db_path) as session:
                    row = session.execute(
                        select(Pass1Result)
                        .where(Pass1Result.run_id == run_id)
                        .where(Pass1Result.query_id == qid)
                    ).scalar_one()
                    row.status = "error"
                    row.error_msg = f"{type(e).__name__}: {e}"
                    row.attempted_at = attempted_at
            finally:
                report.attempted += 1

    await asyncio.gather(*(run_one(s) for s in snapshot))
    return report
