"""Routed-tier answer collection — backs `make answers`.

For each pending `tier_answers` row (one per query, with `tier_level` set
to the router's pick from pass1_results), build an `OAIClient` against that
tier's endpoint and call chat completions. This bypasses the router itself —
the router already chose; here we go direct.

Error policy (per user design): an unreachable or erroring upstream marks
the row as `status='error'` with `error_msg` populated. The pass keeps
going; nothing fails. Re-running `make answers` automatically retries
errored rows on the next attempt.

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


def _extra_body_by_level(models: ModelsConfig) -> dict[int, dict]:
    """Per-tier `extra_body` merged into each chat request.

    Read from `backend.extra_body` in the tier YAML (BackendSpec has
    extra='allow', so arbitrary keys survive model_dump). Used to pass
    provider-specific knobs the OpenAI schema doesn't model — e.g.
    Qwen3's `chat_template_kwargs: {enable_thinking: false}` to stop the
    model from spending the whole token budget on a hidden <think> chain
    (the root cause of the T1 ReadTimeouts on CPU).
    """
    out: dict[int, dict] = {}
    for tier in models.tiers:
        extra = tier.backend.model_dump().get("extra_body")
        if isinstance(extra, dict) and extra:
            out[tier.level] = extra
    return out


def _max_tokens_by_level(models: ModelsConfig) -> dict[int, int]:
    """Per-tier generation cap from `tier.max_tokens` (set via TIER{N}_MAX_TOKENS).

    Tiers without an explicit per-tier cap are absent from the map; the
    worker falls back to the global `--max-tokens` for those. Lets a slow
    local tier be given a bigger budget to finish a complete answer while
    vendor tiers keep the default.
    """
    out: dict[int, int] = {}
    for tier in models.tiers:
        if tier.max_tokens is not None:
            out[tier.level] = tier.max_tokens
    return out


def _build_clients_for_mock(models: ModelsConfig, mock_endpoint: str) -> dict[int, OAIClient]:
    """All tiers point at one mock endpoint — used by `make answers MOCK=true`.

    Preserves each tier's served_model_name + timeout_s; just overrides the URL
    and drops API auth. The mock doesn't enforce auth.
    """
    out: dict[int, OAIClient] = {}
    for tier in models.tiers:
        out[tier.level] = OAIClient(
            endpoint=mock_endpoint,
            model_id=tier.served_model_name,
            api_key=None,
            timeout_s=float(tier.timeout_s),
        )
    return out


async def run_answers(
    db_path: Path,
    run_id: int,
    *,
    models: ModelsConfig,
    concurrency: int = 8,
    max_tokens: int = 2048,
    clients_by_level: dict[int, OAIClient] | None = None,
    mock_endpoint: str | None = None,
    tier_level: int | None = None,
) -> AnswersReport:
    """Process pending tier_answers rows.

    `clients_by_level` is injectable for tests. `mock_endpoint` (e.g.
    `http://localhost:8811/v1`) overrides every tier's endpoint to point at
    the local mock — used for pipeline verification before real backends
    come online.

    `tier_level` (e.g. `1`) restricts the worker to a single tier. Other
    tiers' pending/error rows are left untouched. Useful when you want
    to exercise just-wired backends without re-hitting expensive vendor
    APIs for the tiers that already succeeded.
    """
    if clients_by_level is None:
        if mock_endpoint:
            clients_by_level = _build_clients_for_mock(models, mock_endpoint)
        else:
            clients_by_level = _build_clients_by_level(models)

    # Per-tier extra request-body knobs (e.g. disable Qwen3 thinking).
    # Skipped when hitting the mock (it ignores unknown body fields, but
    # there's no reason to send them).
    extra_by_level = {} if mock_endpoint else _extra_body_by_level(models)

    # Per-tier generation caps (TIER{N}_MAX_TOKENS). Absent → global default.
    # Skipped for the mock, which ignores the budget anyway.
    max_tokens_by_level = {} if mock_endpoint else _max_tokens_by_level(models)

    report = AnswersReport()

    # Snapshot the pending rows. We don't hold a DB session across the
    # async fan-out; each worker opens its own short-lived session.
    with session_scope(db_path) as session:
        q = (
            select(TierAnswer, Query)
            .join(Query, TierAnswer.query_id == Query.query_id)
            .where(TierAnswer.run_id == run_id)
            .where(TierAnswer.status.in_(["pending", "error"]))
        )
        if tier_level is not None:
            q = q.where(TierAnswer.tier_level == tier_level)
        rows = session.execute(q).all()
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
                    max_tokens=max_tokens_by_level.get(level, max_tokens),
                    extra=extra_by_level.get(level) or None,
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
