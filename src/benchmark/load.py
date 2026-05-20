"""Load `data/queries.json` into the SQLite DB.

Every query carries `expected_answers: [{answer, model, provider?}, …]`
— a list, even if there's only one gold. `model` is the per-query
unique key. The JSON is the source of truth for the prompt and its
gold(s); there is no separate `gold` generation step at load.

Semantics:
  - INSERT new queries; upsert every declared gold into the
    gold_answers table (PK (query_id, model_id)).
  - UPDATE metadata + golds when the query already exists.
  - Prompt change → prompt_hash change; golds re-synced too.
  - Reloading is idempotent (unchanged queries are a no-op).
  - load NEVER deletes gold_answers rows — rows produced by
    `make update-gold` and `make import-answers` (with their own
    model_ids) are preserved across reloads. If you remove an
    `expected_answers` entry and re-load, the old row lingers until
    you drop it explicitly.
  - Queries removed from queries.json are NOT deleted (the DB is
    canonical).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from .config import ExpectedAnswer, QuerySet, hash_prompt, load_queries
from .db import GoldAnswer, Query, session_scope


def _existing_golds(session, qid: str) -> dict[str, GoldAnswer]:
    return {
        r.model_id: r
        for r in session.execute(
            select(GoldAnswer).where(GoldAnswer.query_id == qid)
        ).scalars()
    }


def _golds_changed(session, qid: str, golds: list[ExpectedAnswer]) -> bool:
    """True if any declared gold differs from what's in the DB.

    Only looks at the model_ids the file declares. Rows with other
    model_ids (update-gold / import-answers) are intentionally ignored.
    """
    rows = _existing_golds(session, qid)
    for g in golds:
        r = rows.get(g.model_id)
        if r is None or r.answer != g.answer or r.provider != g.provider:
            return True
    return False


def _sync_query_golds(
    session, qid: str, golds: list[ExpectedAnswer], now
) -> None:
    """Upsert every declared gold into gold_answers (PK (qid, model_id))."""
    rows = _existing_golds(session, qid)
    for g in golds:
        r = rows.get(g.model_id)
        if r is None:
            session.add(
                GoldAnswer(
                    query_id=qid,
                    model_id=g.model_id,
                    provider=g.provider,
                    answer=g.answer,
                    generated_at=now,
                )
            )
        else:
            r.provider = g.provider
            r.answer = g.answer
            r.generated_at = now


@dataclass
class LoadReport:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0

    def __str__(self) -> str:
        return (
            f"  inserted:  {self.inserted}\n"
            f"  updated:   {self.updated}\n"
            f"  unchanged: {self.unchanged}"
        )


def load_into_db(queries_json: Path, db_path: Path) -> LoadReport:
    qs: QuerySet = load_queries(queries_json)
    report = LoadReport()
    seen_ids: set[str] = set()
    now = datetime.now(UTC)

    with session_scope(db_path) as session:
        for spec in qs.queries:
            if spec.id in seen_ids:
                raise ValueError(f"duplicate query id in queries.json: {spec.id}")
            seen_ids.add(spec.id)

            new_hash = hash_prompt(spec.prompt, spec.attachments)
            attachments_payload = [a.model_dump() for a in spec.attachments] or None
            golds = spec.expected_answers
            existing = session.execute(
                select(Query).where(Query.query_id == spec.id)
            ).scalar_one_or_none()

            if existing is None:
                session.add(
                    Query(
                        query_id=spec.id,
                        prompt=spec.prompt,
                        prompt_hash=new_hash,
                        attachments=attachments_payload,
                        expected_min_tier=spec.expected_min_tier,
                        specializations=spec.specializations,
                        domain_tags=spec.domain_tags or None,
                        notes=spec.notes,
                    )
                )
                _sync_query_golds(session, spec.id, golds, now)
                report.inserted += 1
                continue

            prompt_changed = existing.prompt_hash != new_hash
            gold_changed = _golds_changed(session, spec.id, golds)
            metadata_changed = (
                existing.expected_min_tier != spec.expected_min_tier
                or list(existing.specializations or []) != spec.specializations
                or list(existing.domain_tags or []) != (spec.domain_tags or [])
                or (existing.notes or None) != spec.notes
                or (existing.attachments or None) != attachments_payload
            )

            if not prompt_changed and not metadata_changed and not gold_changed:
                report.unchanged += 1
                continue

            if prompt_changed:
                existing.prompt = spec.prompt
                existing.prompt_hash = new_hash
                existing.attachments = attachments_payload

            existing.expected_min_tier = spec.expected_min_tier
            existing.specializations = spec.specializations
            existing.domain_tags = spec.domain_tags or None
            existing.notes = spec.notes
            existing.attachments = attachments_payload

            if gold_changed or prompt_changed:
                _sync_query_golds(session, spec.id, golds, now)

            report.updated += 1

    return report
