"""Diagnostic — list queries where the router under-routed.

Backs `make misroutes`. Reads the latest pass1_results for a run and prints
the rows where `meets_minimum_tier=0` — i.e. the router picked a tier
SMALLER than the query's `expected_min_tier`. Over-routes (router picked
above min) are not flagged here; under-routes are the demo-failure case.

Per-signal scores aren't in our DB (they only appear in the router's log
stream), so this view shows the routing decision and the available routing
metadata (category, reasoning mode, headers). Patterns in the misroutes —
e.g. "T4-expected queries all routed to T2 with category=business" — tell
us which axis or threshold to tweak.
"""
from __future__ import annotations

import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from .db import Pass1Result, Query, Run, session_scope


@dataclass
class Misroute:
    query_id: str
    prompt: str
    expected_min_tier: int
    routed_tier: int | None
    specializations: list[str]
    domain_tags: list[str]
    category: str | None
    reasoning_mode: str | None


def _latest_run(db_path: Path) -> int | None:
    with session_scope(db_path) as s:
        row = s.execute(select(Run).order_by(Run.run_id.desc())).scalars().first()
        return None if row is None else row.run_id


def list_misroutes(db_path: Path, run_id: int | None = None) -> list[Misroute]:
    """All queries the router under-tiered for `run_id` (default: latest run)."""
    if run_id is None:
        run_id = _latest_run(db_path)
    if run_id is None:
        return []

    out: list[Misroute] = []
    with session_scope(db_path) as s:
        rows = s.execute(
            select(Pass1Result, Query)
            .join(Query, Pass1Result.query_id == Query.query_id)
            .where(Pass1Result.run_id == run_id)
            .where(Pass1Result.status == "success")
            .where(Pass1Result.meets_minimum_tier == 0)
            .order_by(Query.expected_min_tier.desc(), Query.query_id)
        ).all()
        for p1, q in rows:
            meta = p1.raw_routing_metadata or {}
            out.append(Misroute(
                query_id=q.query_id,
                prompt=q.prompt,
                expected_min_tier=q.expected_min_tier,
                routed_tier=p1.router_selected_tier,
                specializations=list(q.specializations or []),
                domain_tags=list(q.domain_tags or []),
                category=meta.get("category"),
                reasoning_mode=meta.get("reasoning"),
            ))
    return out


def render_misroutes(misroutes: list[Misroute], *, max_prompt_chars: int = 100) -> str:
    """Pretty-print a list of misroutes plus per-axis summary aggregates."""
    if not misroutes:
        return "No misroutes — every routed query met its expected_min_tier."

    lines: list[str] = []
    lines.append(f"{len(misroutes)} misroute(s):\n")

    for m in misroutes:
        # Use angle brackets — square brackets get eaten by Rich markup parsing.
        specs = ",".join(m.specializations) if m.specializations else "-"
        cat = m.category or "?"
        reasoning = m.reasoning_mode or "?"
        header = (
            f"  {m.query_id}  expected≥{m.expected_min_tier}  "
            f"routed→{m.routed_tier or '?'}  "
            f"specs=<{specs}>  category={cat}  reasoning={reasoning}"
        )
        prompt = textwrap.shorten(m.prompt.replace("\n", " "), max_prompt_chars)
        lines.append(header)
        lines.append(f'    "{prompt}"')
        lines.append("")

    # Aggregates that help us decide where to nudge thresholds.
    lines.append("Breakdown:")

    expected_counter: Counter[int] = Counter(m.expected_min_tier for m in misroutes)
    for tier in sorted(expected_counter, reverse=True):
        lines.append(f"  T{tier} expected:  {expected_counter[tier]} under-routed")

    routed_counter: Counter[int | None] = Counter(m.routed_tier for m in misroutes)
    lines.append("")
    lines.append("Routed-to distribution (for the misroutes):")
    for tier in sorted((t for t in routed_counter if t is not None), reverse=False):
        lines.append(f"  routed→T{tier}:  {routed_counter[tier]}")

    cat_counter: Counter[str] = Counter(m.category or "?" for m in misroutes)
    lines.append("")
    lines.append("Router category breakdown:")
    for cat, n in cat_counter.most_common():
        lines.append(f"  {cat}:  {n}")

    return "\n".join(lines)
