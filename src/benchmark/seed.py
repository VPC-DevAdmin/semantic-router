"""Idempotent upsert of curated queries into the DB.

Seed semantics:
  - INSERT new queries.
  - UPDATE metadata fields (specializations, domain_tags, expected_min_tier, notes,
    attachments) when the query already exists; this lets us refine taxonomy
    without losing gold answers.
  - If `prompt` changes, the prompt_hash changes; we WIPE gold_answer/gold_model/
    gold_generated_at because the gold no longer applies. This is intentional and
    surfaced in the seed report.
  - Queries removed from queries.yaml are NOT deleted from the DB; the DB is the
    canonical record. `make clean-results` is the explicit destructive path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from .config import QuerySet, hash_prompt, load_queries
from .db import Query, session_scope


@dataclass
class SeedReport:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    gold_invalidated: int = 0
    invalidated_ids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.invalidated_ids is None:
            self.invalidated_ids = []

    def __str__(self) -> str:
        lines = [
            f"  inserted:         {self.inserted}",
            f"  updated:          {self.updated}",
            f"  unchanged:        {self.unchanged}",
            f"  gold invalidated: {self.gold_invalidated}",
        ]
        if self.invalidated_ids:
            lines.append(f"    -> {', '.join(self.invalidated_ids)}")
        return "\n".join(lines)


def seed_from_yaml(yaml_path: Path, db_path: Path) -> SeedReport:
    qs: QuerySet = load_queries(yaml_path)
    report = SeedReport()
    seen_ids: set[str] = set()

    with session_scope(db_path) as session:
        for spec in qs.queries:
            if spec.id in seen_ids:
                raise ValueError(f"duplicate query id in yaml: {spec.id}")
            seen_ids.add(spec.id)

            new_hash = hash_prompt(spec.prompt, spec.attachments)
            existing = session.execute(
                select(Query).where(Query.query_id == spec.id)
            ).scalar_one_or_none()

            attachments_payload = [a.model_dump() for a in spec.attachments] or None

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
                report.inserted += 1
                continue

            prompt_changed = existing.prompt_hash != new_hash
            metadata_changed = (
                existing.expected_min_tier != spec.expected_min_tier
                or list(existing.specializations or []) != spec.specializations
                or list(existing.domain_tags or []) != (spec.domain_tags or [])
                or (existing.notes or None) != spec.notes
                or (existing.attachments or None) != attachments_payload
            )

            if not prompt_changed and not metadata_changed:
                report.unchanged += 1
                continue

            if prompt_changed:
                existing.prompt = spec.prompt
                existing.prompt_hash = new_hash
                existing.attachments = attachments_payload
                if existing.gold_answer is not None:
                    existing.gold_answer = None
                    existing.gold_model = None
                    existing.gold_generated_at = None
                    report.gold_invalidated += 1
                    assert report.invalidated_ids is not None
                    report.invalidated_ids.append(spec.id)

            existing.expected_min_tier = spec.expected_min_tier
            existing.specializations = spec.specializations
            existing.domain_tags = spec.domain_tags or None
            existing.notes = spec.notes
            existing.attachments = attachments_payload
            report.updated += 1

    return report
