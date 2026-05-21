"""Emit `the routed-queries JSON` from the DB — backs `make export`.

`the routed-queries JSON` is the single artifact downstream consumers (replay UI,
external judging, slide plots) read. Multi-model shape:

  {
    "id": "q00001",
    "prompt": "...",
    "expected_min_tier": 1,
    "specializations": ["general"],
    "domain_tags": [...],
    "notes": "...",
    "routed_tier": 3 | null,
    "routing_metadata": {...} | null,

    # Per-provider expected answers (the gold set). Populated by
    # `make load` (from queries.json), `make update-gold` (top-tier
    # model calls), and `make import-answers`.
    "expected_answers": [
      {"provider": null, "model": "upstream", "answer": "..."},
      ...
    ],

    # Every model the ROUTED tier fronts got called — one entry each.
    "routed_answers": [
      {"tier": 3, "provider": "OpenAI", "model": "gpt-5-mini",
       "answer": "...", "status": "success", "latency_ms": 1234},
      ...
    ],
  }

Whatever's in the DB at export time is what lands, with empty lists /
null for missing pieces. The export is independent and idempotent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select

from .db import Evaluation, GoldAnswer, Pass1Result, Query, TierAnswer, session_scope


@dataclass
class ExportSummary:
    output_path: Path
    queries_exported: int = 0
    with_routed_tier: int = 0
    with_routed_answer: int = 0  # queries with >=1 successful routed answer
    with_expected: int = 0       # queries with >=1 expected answer
    # (tier_level, model_id) → count of successful routed answers
    routed_answers_per_model: dict[tuple[int, str], int] = field(default_factory=dict)
    # Queries routed to the top tier — no per-model answers needed because
    # the top tier IS the gold reference (every comparison is routed-vs-top).
    top_tier_routed: int = 0
    # Sibling export: evaluations.json. Path is None when no evaluations
    # exist for the run (the file is intentionally not emitted in that case).
    evaluations_path: Path | None = None
    evaluations_written: int = 0
    evaluations_by_evaluator: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"  wrote:                   {self.output_path}",
            f"  queries exported:        {self.queries_exported}",
            f"  with routed tier:        {self.with_routed_tier}",
            f"  with routed answer(s):   {self.with_routed_answer}",
            f"  with expected answer(s): {self.with_expected}",
        ]
        if self.routed_answers_per_model:
            lines.append("  routed answers by tier × model:")
            # Group by tier; longest model name sets the column width.
            by_tier: dict[int, list[tuple[str, int]]] = {}
            for (tier, model), n in self.routed_answers_per_model.items():
                by_tier.setdefault(tier, []).append((model, n))
            name_width = max(
                len(model)
                for (_, model), _ in self.routed_answers_per_model.items()
            )
            total = 0
            for tier in sorted(by_tier):
                for model, n in sorted(by_tier[tier]):
                    lines.append(
                        f"    tier{tier}  {model:<{name_width}}  {n}"
                    )
                    total += n
            lines.append(f"  total routed answers:    {total}")
        if self.top_tier_routed:
            lines.append(
                f"  top-tier routed:         {self.top_tier_routed} "
                f"(no per-model answers — top tier is the gold reference)"
            )
        if self.evaluations_path:
            lines.append(f"  evaluations wrote:       {self.evaluations_path}")
            lines.append(f"  evaluations written:     {self.evaluations_written}")
            for evname, n in sorted(self.evaluations_by_evaluator.items()):
                lines.append(f"    {evname}: {n}")
        return "\n".join(lines)


def _build_query_entry(
    q: Query,
    p1: Pass1Result | None,
    tier_answers: list[TierAnswer],
    gold_answers: list[GoldAnswer],
) -> dict[str, Any]:
    """Build one the routed-queries JSON entry for a single query."""
    routed_tier = p1.router_selected_tier if p1 is not None else None

    routing_metadata: dict[str, Any] | None = None
    if p1 is not None and p1.status == "success":
        routing_metadata = {
            "selected_model": p1.router_selected_model,
            "selected_tier": p1.router_selected_tier,
            "selected_specs": p1.router_selected_specs,
            "meets_minimum_tier": p1.meets_minimum_tier,
            "matches_specialization": p1.matches_specialization,
            # Wall-clock the router took to decide this query (the
            # max_tokens=1 routing probe round-trip, measured in pass1).
            "latency_ms": p1.latency_ms,
            "raw": p1.raw_routing_metadata,
        }

    routed_answers: list[dict[str, Any]] = []
    if routed_tier is not None:
        for ta in sorted(
            (r for r in tier_answers if r.tier_level == routed_tier),
            key=lambda r: (r.model_slot, r.model_id),
        ):
            routed_answers.append(
                {
                    "tier": ta.tier_level,
                    "provider": ta.provider,
                    "model": ta.model_id,
                    "answer": ta.response_text if ta.status == "success" else None,
                    "status": ta.status,
                    "latency_ms": ta.latency_ms,
                }
            )

    expected_answers = [
        {
            "provider": g.provider,
            "model": g.model_id,
            "answer": g.answer,
        }
        for g in sorted(gold_answers, key=lambda g: g.model_id)
    ]

    return {
        "id": q.query_id,
        "prompt": q.prompt,
        "expected_min_tier": q.expected_min_tier,
        "specializations": q.specializations,
        "domain_tags": q.domain_tags or [],
        "notes": q.notes,
        "routed_tier": routed_tier,
        "routing_metadata": routing_metadata,
        "expected_answers": expected_answers,
        "routed_answers": routed_answers,
    }


def export_demo_json(
    db_path: Path,
    run_id: int,
    output_path: Path,
) -> ExportSummary:
    """Read the DB for `run_id` and write the routed-queries JSON to `output_path`."""
    summary = ExportSummary(output_path=output_path)
    entries: list[dict[str, Any]] = []

    with session_scope(db_path) as session:
        queries = list(
            session.execute(select(Query).order_by(Query.query_id)).scalars()
        )

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
        gold_by_qid: dict[str, list[GoldAnswer]] = {}
        for g in session.execute(select(GoldAnswer)).scalars():
            gold_by_qid.setdefault(g.query_id, []).append(g)

        for q in queries:
            entry = _build_query_entry(
                q,
                p1_by_qid.get(q.query_id),
                tier_by_qid.get(q.query_id, []),
                gold_by_qid.get(q.query_id, []),
            )
            entries.append(entry)

            summary.queries_exported += 1
            if entry["routed_tier"] is not None:
                summary.with_routed_tier += 1
            n_routed_ok = sum(
                1 for r in entry["routed_answers"] if r["status"] == "success"
            )
            if n_routed_ok:
                summary.with_routed_answer += 1
            if entry["expected_answers"]:
                summary.with_expected += 1
            # Count successful answers per (tier, model). For routed-to-top-tier
            # queries this list is empty by design (top tier IS the gold).
            for r in entry["routed_answers"]:
                if r["status"] == "success":
                    key = (r["tier"], r["model"])
                    summary.routed_answers_per_model[key] = (
                        summary.routed_answers_per_model.get(key, 0) + 1
                    )
            # Track top-tier-routed queries separately so the summary can
            # explain the "0 routed answers" case without making it look
            # like a coverage gap.
            if entry["routed_tier"] is not None and not entry["routed_answers"]:
                summary.top_tier_routed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))

    # Sibling export: evaluations.json (only if rows exist for this run).
    # Path is conventionally `data/evaluations.json` alongside the routed
    # export, but we derive it from output_path so a custom --output
    # doesn't drop the sibling somewhere surprising.
    _maybe_write_evaluations(db_path, run_id, output_path, summary)
    return summary


def _maybe_write_evaluations(
    db_path: Path,
    run_id: int,
    routed_output_path: Path,
    summary: ExportSummary,
) -> None:
    """Write a sibling evaluations.json next to the routed export, if
    any Evaluation rows exist for this run. No file written on absence
    (clean signal — downstream readers see no file vs. an empty list)."""
    eval_path = routed_output_path.parent / "evaluations.json"

    with session_scope(db_path) as session:
        rows = list(
            session.execute(
                select(Evaluation, Query)
                .join(Query, Evaluation.query_id == Query.query_id)
                .where(Evaluation.run_id == run_id)
                .where(Evaluation.status == "success")
            ).all()
        )

    if not rows:
        return

    entries: list[dict[str, Any]] = []
    for ev, _ in sorted(
        rows,
        key=lambda r: (r[0].query_id, r[0].routed_tier, r[0].routed_model,
                       r[0].gold_model_id, r[0].evaluator),
    ):
        entries.append({
            "eval_id": _eval_id(ev),
            "query_id": ev.query_id,
            "routed_provider": ev.routed_provider,
            "routed_model": ev.routed_model,
            "expected_provider": ev.gold_provider,
            "expected_model": ev.gold_model_id,
            "evaluator": ev.evaluator,
            "verdict": ev.verdict,
            "rationale": ev.rationale,
            "scores": {
                "correctness": ev.correctness,
                "completeness": ev.completeness,
                "fitness_for_purpose": ev.fitness_for_purpose,
            },
        })

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))

    summary.evaluations_path = eval_path
    summary.evaluations_written = len(entries)
    for e in entries:
        summary.evaluations_by_evaluator[e["evaluator"]] = (
            summary.evaluations_by_evaluator.get(e["evaluator"], 0) + 1
        )


def _eval_id(ev: Evaluation) -> str:
    """Stable eval_id with evaluator suffix so multi-evaluator runs don't
    collide. Mirrors evaluations._eval_id; duplicated here to avoid an
    import cycle (export imports tiers transitively)."""

    def _slug(s: str | None) -> str:
        return (s or "unknown").replace(" ", "-").replace(".", "_").lower()

    return (
        f"{ev.query_id}-{_slug(ev.routed_provider)}-{_slug(ev.routed_model)}"
        f"-vs-{_slug(ev.gold_provider)}-{_slug(ev.gold_model_id)}"
        f"--{_slug(ev.evaluator)}"
    )
