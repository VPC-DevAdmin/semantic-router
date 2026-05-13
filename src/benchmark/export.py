"""Emit `demo.json` from the DB — backs `make export`.

`demo.json` is the single artifact that downstream consumers (replay UI,
external judging workflow, slide plots) read. Shape per PLAN.md §7:

  {
    "id": "q00001",
    "prompt": "...",
    "expected_min_tier": 1,
    "specializations": ["general"],
    "domain_tags": [...],
    "routed_tier": 1 | null,
    "routing_metadata": { selected_model, category, reasoning, raw_headers }
                       | null,
    "responses": {
        "gold":   { "tier": 5, "answer": "..." },
        "routed": { "tier": 1, "answer": "..." } | null
    },
    "all_tier_answers": { "tier1": "...", "tier2": "...", ... }
  }

We don't require any specific pass to have completed — whatever's in the
DB at the time of `make export` is what lands in the JSON, with `null`
filling in for missing pieces. That keeps the export step independent
and idempotent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select

from .db import Pass1Result, Query, TierAnswer, session_scope

GOLD_TIER_DEFAULT = 5  # Opus → tier 5 by convention


@dataclass
class ExportSummary:
    output_path: Path
    queries_exported: int = 0
    with_routed_tier: int = 0
    with_routed_answer: int = 0
    tiers_per_query: dict[int, int] = field(default_factory=dict)  # n_tier_answers → count

    def __str__(self) -> str:
        lines = [
            f"  wrote:                  {self.output_path}",
            f"  queries exported:       {self.queries_exported}",
            f"  with routed tier:       {self.with_routed_tier}",
            f"  with routed answer:     {self.with_routed_answer}",
        ]
        if self.tiers_per_query:
            lines.append("  tier-answer coverage:")
            for n, count in sorted(self.tiers_per_query.items()):
                lines.append(f"    {n} tier(s): {count} query(ies)")
        return "\n".join(lines)


def _build_query_entry(
    q: Query,
    p1: Pass1Result | None,
    tier_answers: list[TierAnswer],
    *,
    gold_tier_default: int,
) -> dict[str, Any]:
    """Build one demo.json entry for a single query."""
    # Tier-name keyed map of answers (tier1 → "...", tier2 → "...", etc.).
    # Only successful tier_answers contribute; we expose null for the
    # routed answer below if the routed tier didn't complete.
    by_level: dict[int, TierAnswer] = {
        ta.tier_level: ta for ta in tier_answers if ta.status == "success"
    }
    all_tier_answers = {
        ta.tier_name: ta.response_text
        for ta in tier_answers
        if ta.status == "success" and ta.response_text is not None
    }

    routed_tier = p1.router_selected_tier if p1 is not None else None
    routing_metadata: dict[str, Any] | None = None
    if p1 is not None and p1.status == "success":
        routing_metadata = {
            "selected_model": p1.router_selected_model,
            "selected_tier": p1.router_selected_tier,
            "selected_specs": p1.router_selected_specs,
            "meets_minimum_tier": p1.meets_minimum_tier,
            "matches_specialization": p1.matches_specialization,
            "raw": p1.raw_routing_metadata,
        }

    routed_response: dict[str, Any] | None = None
    if routed_tier is not None and routed_tier in by_level:
        routed_response = {
            "tier": routed_tier,
            "answer": by_level[routed_tier].response_text,
        }

    gold_response: dict[str, Any] | None = None
    if q.gold_answer is not None:
        gold_response = {
            "tier": gold_tier_default,
            "answer": q.gold_answer,
            "source_model": q.gold_model,
        }

    return {
        "id": q.query_id,
        "prompt": q.prompt,
        "expected_min_tier": q.expected_min_tier,
        "specializations": q.specializations,
        "domain_tags": q.domain_tags or [],
        "notes": q.notes,
        "routed_tier": routed_tier,
        "routing_metadata": routing_metadata,
        "responses": {
            "gold": gold_response,
            "routed": routed_response,
        },
        "all_tier_answers": all_tier_answers,
    }


def export_demo_json(
    db_path: Path,
    run_id: int,
    output_path: Path,
    *,
    gold_tier_default: int = GOLD_TIER_DEFAULT,
) -> ExportSummary:
    """Read the DB for `run_id` and write demo.json to `output_path`."""
    summary = ExportSummary(output_path=output_path)
    entries: list[dict[str, Any]] = []

    with session_scope(db_path) as session:
        # All queries, ordered for stable output.
        queries = list(
            session.execute(select(Query).order_by(Query.query_id)).scalars()
        )

        # Pre-fetch the run's pass1 + tier_answers, indexed by query_id.
        p1_by_qid: dict[str, Pass1Result] = {
            p.query_id: p
            for p in session.execute(
                select(Pass1Result).where(Pass1Result.run_id == run_id)
            ).scalars()
        }
        tier_by_qid: dict[str, list[TierAnswer]] = {}
        for ta in session.execute(
            select(TierAnswer).where(TierAnswer.run_id == run_id)
        ).scalars():
            tier_by_qid.setdefault(ta.query_id, []).append(ta)

        for q in queries:
            entry = _build_query_entry(
                q,
                p1_by_qid.get(q.query_id),
                tier_by_qid.get(q.query_id, []),
                gold_tier_default=gold_tier_default,
            )
            entries.append(entry)

            summary.queries_exported += 1
            if entry["routed_tier"] is not None:
                summary.with_routed_tier += 1
            if entry["responses"]["routed"] is not None:
                summary.with_routed_answer += 1
            n_tiers = len(entry["all_tier_answers"])
            summary.tiers_per_query[n_tiers] = summary.tiers_per_query.get(n_tiers, 0) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    return summary
