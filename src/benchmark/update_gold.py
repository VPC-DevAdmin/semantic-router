"""Regenerate gold answers by calling the top-tier model.

Backs `make update-gold`. The query set's `expected_answer` is the
gold reference; the top tier in our routing IS the gold-grade model
(Claude Opus). When you want a *fresh* gold for some queries — because
the upstream answer is stale, wrong, or missing — this calls the top
tier and overwrites `Query.gold_answer` (plus provenance fields).

This is intentionally a SEPARATE command from `make answers`:
  • `make answers` consumes gold (top-tier rows are filled from it).
  • `make update-gold` PRODUCES gold (calls Opus, writes it back).

It's destructive (overwrites the existing gold), so the CLI requires
explicit query IDs or an explicit --all, and the marker written to
`gold_model` records that this gold came from a live top-tier call,
distinguishing it from the upstream-import marker in load.py.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from .config import ModelsConfig, TierConfig
from .db import Query, session_scope
from .tiers import client_from_tier

# Written to Query.gold_model so a regenerated gold is distinguishable
# from the upstream-import marker (load.GOLD_SOURCE_MARKER).
REGEN_GOLD_MARKER = "regenerated via update-gold (top tier)"


@dataclass
class UpdateGoldResult:
    updated: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (qid, msg)
    skipped_unknown: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"  updated:  {len(self.updated)}",
            f"  errors:   {len(self.errors)}",
        ]
        for qid, msg in self.errors[:10]:
            lines.append(f"     {qid}: {msg}")
        if len(self.errors) > 10:
            lines.append(f"     ... and {len(self.errors) - 10} more")
        if self.skipped_unknown:
            lines.append(
                f"  skipped (unknown qid): {len(self.skipped_unknown)} → "
                f"{', '.join(self.skipped_unknown[:5])}"
                f"{'...' if len(self.skipped_unknown) > 5 else ''}"
            )
        return "\n".join(lines)


def _top_tier(models: ModelsConfig) -> TierConfig:
    if not models.tiers:
        raise ValueError("no tiers configured")
    return max(models.tiers, key=lambda t: t.level)


async def update_gold_answers(
    db_path: Path,
    *,
    query_ids: list[str],
    models: ModelsConfig,
    max_tokens: int = 2048,
    concurrency: int = 4,
) -> UpdateGoldResult:
    """Call the top-tier model for each query and overwrite its gold.

    `query_ids` is explicit (no implicit "all") — the caller decides the
    scope. Unknown ids are reported, not silently dropped. The prompt
    sent is the query's own `prompt`; the response replaces
    `Query.gold_answer` and stamps `gold_model` / `gold_generated_at`.
    """
    tier = _top_tier(models)
    client = client_from_tier(tier)
    result = UpdateGoldResult()

    # Snapshot (query_id, prompt) for the requested ids.
    with session_scope(db_path) as session:
        rows = session.execute(
            select(Query.query_id, Query.prompt).where(
                Query.query_id.in_(query_ids)
            )
        ).all()
    found = {qid: prompt for qid, prompt in rows}
    for qid in query_ids:
        if qid not in found:
            result.skipped_unknown.append(qid)

    sem = asyncio.Semaphore(concurrency)
    now = datetime.now(UTC)

    async def _one(qid: str, prompt: str) -> None:
        async with sem:
            try:
                chat = await client.chat(prompt, max_tokens=max_tokens)
            except Exception as e:  # noqa: BLE001 — surface, keep going
                result.errors.append((qid, f"{type(e).__name__}: {e}"))
                return
            text = (chat.content or "").strip()
            if not text:
                result.errors.append((qid, "empty response from top tier"))
                return
            with session_scope(db_path) as session:
                q = session.execute(
                    select(Query).where(Query.query_id == qid)
                ).scalar_one()
                q.gold_answer = text
                q.gold_model = f"{REGEN_GOLD_MARKER}: {tier.served_model_name}"
                q.gold_generated_at = now
            result.updated.append(qid)

    await asyncio.gather(*(_one(qid, found[qid]) for qid in found))
    return result
