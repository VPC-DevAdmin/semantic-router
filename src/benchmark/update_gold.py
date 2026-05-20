"""Regenerate per-provider gold answers by calling the top-tier models.

Backs `make update-gold`. The top tier can be fronted by several models
(Anthropic Opus, OpenAI GPT-5, Google Gemini Pro…). This calls EVERY
top-tier model for each requested query and upserts one `gold_answers`
row per (query, model) — so the demo's `expected_answers[]` shows a
gold per provider.

Separate from `make answers`:
  • `make answers` collects the routed tier's answers (tier_answers).
  • `make update-gold` PRODUCES gold (calls the top tier, writes
    gold_answers, and mirrors the slot-0 model into Query.gold_answer
    for back-compat).

Destructive (overwrites existing per-model gold), so the CLI requires
explicit scope.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from .config import ModelsConfig, TierConfig
from .db import GoldAnswer, Query, session_scope
from .tiers import client_from_model

# Written to Query.gold_model (back-compat single-gold field) so a
# regenerated gold is distinguishable from the upstream-import marker.
REGEN_GOLD_MARKER = "regenerated via update-gold (top tier)"


@dataclass
class UpdateGoldResult:
    updated: list[tuple[str, str]] = field(default_factory=list)  # (qid, model_id)
    errors: list[tuple[str, str]] = field(default_factory=list)   # (qid:model, msg)
    skipped_unknown: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        n_q = len({qid for qid, _ in self.updated})
        lines = [
            f"  updated:  {len(self.updated)} gold row(s) across {n_q} query(ies)",
            f"  errors:   {len(self.errors)}",
        ]
        for tag, msg in self.errors[:10]:
            lines.append(f"     {tag}: {msg}")
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


def _upsert_gold(session, qid: str, model_id: str, provider, answer: str, now) -> None:
    row = session.execute(
        select(GoldAnswer)
        .where(GoldAnswer.query_id == qid)
        .where(GoldAnswer.model_id == model_id)
    ).scalar_one_or_none()
    if row is None:
        session.add(
            GoldAnswer(
                query_id=qid,
                model_id=model_id,
                provider=provider,
                answer=answer,
                generated_at=now,
            )
        )
    else:
        row.provider = provider
        row.answer = answer
        row.generated_at = now


async def update_gold_answers(
    db_path: Path,
    *,
    query_ids: list[str],
    models: ModelsConfig,
    max_tokens: int = 2048,
    concurrency: int = 4,
) -> UpdateGoldResult:
    """Call every top-tier model for each query; upsert per-model gold.

    `query_ids` is explicit (no implicit "all" here — the caller decides
    scope). Unknown ids are reported, not silently dropped. Each
    top-tier model's response becomes a `gold_answers` row; the slot-0
    model also refreshes `Query.gold_answer` for back-compat.
    """
    tier = _top_tier(models)
    tier_models = tier.resolved_models()
    result = UpdateGoldResult()

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

    async def _one(qid: str, prompt: str, m) -> None:
        async with sem:
            client = client_from_model(m)
            tag = f"{qid}:{m.served_model_name}"
            try:
                chat = await client.chat(prompt, max_tokens=max_tokens)
            except Exception as e:  # noqa: BLE001 — surface, keep going
                result.errors.append((tag, f"{type(e).__name__}: {e}"))
                return
            text = (chat.content or "").strip()
            if not text:
                result.errors.append((tag, "empty response from top tier"))
                return
            with session_scope(db_path) as session:
                _upsert_gold(
                    session, qid, m.served_model_name, m.provider, text, now
                )
                if m.slot == 0:
                    q = session.execute(
                        select(Query).where(Query.query_id == qid)
                    ).scalar_one()
                    q.gold_answer = text
                    q.gold_model = f"{REGEN_GOLD_MARKER}: {m.served_model_name}"
                    q.gold_generated_at = now
            result.updated.append((qid, m.served_model_name))

    await asyncio.gather(
        *(_one(qid, prompt, m) for qid, prompt in found.items() for m in tier_models)
    )
    return result
