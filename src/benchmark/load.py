"""Load `data/queries.json` into the SQLite DB.

Each query in the JSON file carries an `expected_answer` field that is treated
as the gold-standard reference (these are Opus-level answers from upstream).
There is no separate `gold` generation step — the JSON is the source of truth
for both the prompt and its gold answer.

Semantics:
  - INSERT new queries with gold populated from `expected_answer`.
  - UPDATE metadata + gold when the query already exists.
  - If `prompt` changes, the prompt_hash changes; gold is also refreshed from
    `expected_answer` (so a new prompt always lands with the matching gold).
  - Queries removed from queries.json are NOT deleted from the DB; the DB is
    canonical. `make clean-results` is the explicit destructive path for runs;
    nothing wipes queries automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from .config import QuerySet, hash_prompt, load_queries
from .db import GoldAnswer, Query, session_scope

GOLD_SOURCE_MARKER = "expected_answer (upstream gold)"

# Identity used for the upstream gold in the gold_answers table. The
# upstream queries.json doesn't record which model produced
# expected_answer, so model_id is the literal "upstream" and provider is
# null. update-gold / import-answers add real per-provider rows alongside.
UPSTREAM_GOLD_MODEL = "upstream"
UPSTREAM_GOLD_SOURCE = "upstream"


def _upsert_upstream_gold(session, qid: str, answer: str | None, now) -> None:
    """Mirror queries.json `expected_answer` into gold_answers (or remove it)."""
    from sqlalchemy import select

    row = session.execute(
        select(GoldAnswer)
        .where(GoldAnswer.query_id == qid)
        .where(GoldAnswer.model_id == UPSTREAM_GOLD_MODEL)
    ).scalar_one_or_none()
    if not answer:
        if row is not None:
            session.delete(row)
        return
    if row is None:
        session.add(
            GoldAnswer(
                query_id=qid,
                model_id=UPSTREAM_GOLD_MODEL,
                provider=None,
                answer=answer,
                source=UPSTREAM_GOLD_SOURCE,
                generated_at=now,
            )
        )
    else:
        row.answer = answer
        row.source = UPSTREAM_GOLD_SOURCE
        row.generated_at = now


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
                        gold_answer=spec.expected_answer,
                        gold_model=GOLD_SOURCE_MARKER if spec.expected_answer else None,
                        gold_generated_at=now if spec.expected_answer else None,
                    )
                )
                _upsert_upstream_gold(session, spec.id, spec.expected_answer, now)
                report.inserted += 1
                continue

            prompt_changed = existing.prompt_hash != new_hash
            gold_changed = (existing.gold_answer or None) != (spec.expected_answer or None)
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
                existing.gold_answer = spec.expected_answer
                existing.gold_model = GOLD_SOURCE_MARKER if spec.expected_answer else None
                existing.gold_generated_at = now if spec.expected_answer else None
                _upsert_upstream_gold(session, spec.id, spec.expected_answer, now)

            report.updated += 1

    return report
