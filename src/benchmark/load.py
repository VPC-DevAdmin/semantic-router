"""Load `data/queries.json` into the SQLite DB.

Each query may carry one OR MORE gold/reference answers:
  - legacy `expected_answer` (a single string) → one "upstream" gold, or
  - `expected_answers: [ {answer, source, model?, provider?}, ... ]`
    for several golds (e.g. an upstream reference + a human-reviewed one,
    or one per provider).
Both may coexist; each gold's `model` (default = its `source`) must be
unique within the query. The JSON is the source of truth for the prompt
and its gold(s) — there is no separate `gold` generation step at load.

Semantics:
  - INSERT new queries; sync every gold into the gold_answers table.
  - UPDATE metadata + golds when the query already exists.
  - Prompt change → prompt_hash change; golds re-synced too.
  - Reloading is idempotent (unchanged queries are a no-op).
  - Queries removed from queries.json are NOT deleted (the DB is
    canonical). load only manages the golds the file declares plus the
    legacy "upstream" row — it never deletes update-gold / import-answers
    rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from .config import ExpectedAnswer, QuerySet, hash_prompt, load_queries
from .db import GoldAnswer, Query, session_scope

# Stamped on Query.gold_model (the back-compat single-value mirror) when
# the primary gold is the legacy upstream entry.
GOLD_SOURCE_MARKER = "expected_answer (upstream gold)"

UPSTREAM_GOLD_MODEL = "upstream"


def _primary(golds: list[ExpectedAnswer]) -> ExpectedAnswer | None:
    """The gold mirrored into the back-compat Query.gold_answer field:
    the upstream entry if present, else the first declared gold."""
    for g in golds:
        if g.model_id == UPSTREAM_GOLD_MODEL:
            return g
    return golds[0] if golds else None


def _gold_marker(primary: ExpectedAnswer | None) -> str | None:
    if primary is None:
        return None
    if primary.model_id == UPSTREAM_GOLD_MODEL:
        return GOLD_SOURCE_MARKER
    return f"{GOLD_SOURCE_MARKER} :: {primary.model_id}"


def _existing_golds(session, qid: str) -> dict[str, GoldAnswer]:
    return {
        r.model_id: r
        for r in session.execute(
            select(GoldAnswer).where(GoldAnswer.query_id == qid)
        ).scalars()
    }


def _golds_changed(session, qid: str, golds: list[ExpectedAnswer]) -> bool:
    """True if the file's declared golds differ from what's in the DB.

    Only looks at the model_ids the file declares, plus the legacy
    "upstream" row (so blanking it counts as a change). Other rows
    (update-gold / import-answers, with their own model_ids) are
    intentionally ignored.
    """
    rows = _existing_golds(session, qid)
    want = {g.model_id: g for g in golds}
    for mid, g in want.items():
        r = rows.get(mid)
        if r is None or r.answer != g.answer or r.provider != g.provider:
            return True
    # file dropped the upstream gold → that's also a change
    return UPSTREAM_GOLD_MODEL not in want and UPSTREAM_GOLD_MODEL in rows


def _sync_query_golds(
    session, qid: str, golds: list[ExpectedAnswer], now
) -> None:
    """Upsert every declared gold into gold_answers (PK (qid, model_id)).

    Scoped deletion: if the file no longer declares the legacy
    "upstream" gold, that one row is removed — but rows with other
    model_ids (the common case for update-gold / import-answers) are
    never touched.
    """
    rows = _existing_golds(session, qid)
    want = {g.model_id: g for g in golds}
    for mid, g in want.items():
        r = rows.get(mid)
        if r is None:
            session.add(
                GoldAnswer(
                    query_id=qid,
                    model_id=mid,
                    provider=g.provider,
                    answer=g.answer,
                    generated_at=now,
                )
            )
        else:
            r.provider = g.provider
            r.answer = g.answer
            r.generated_at = now
    if UPSTREAM_GOLD_MODEL not in want and UPSTREAM_GOLD_MODEL in rows:
        session.delete(rows[UPSTREAM_GOLD_MODEL])


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
            golds = spec.golds()
            primary = _primary(golds)
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
                        gold_answer=primary.answer if primary else None,
                        gold_model=_gold_marker(primary),
                        gold_generated_at=now if primary else None,
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
                existing.gold_answer = primary.answer if primary else None
                existing.gold_model = _gold_marker(primary)
                existing.gold_generated_at = now if primary else None
                _sync_query_golds(session, spec.id, golds, now)

            report.updated += 1

    return report
