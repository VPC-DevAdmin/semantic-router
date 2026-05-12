"""Aggregate report for a run.

Computes:
  - Pass-1 routing accuracy: % of queries where the router picked a model at
    or above the expected min tier; % where specs matched; unknown-tier count.
  - Pass-2 response quality: per-scorer score histograms and means.
  - Per-specialization breakdown of both passes.

Outputs a human-readable Rich table to stdout and optional CSV/JSON.

A run's `runs` row is the canonical anchor; we don't require status='done' —
partial runs report what they have.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from .db import Pass1Result, Pass2Result, Query, Run, Score, session_scope


@dataclass
class PassOneStats:
    total: int = 0
    success: int = 0
    error: int = 0
    pending: int = 0
    meets_min_tier: int = 0
    matches_spec: int = 0
    unknown_tier: int = 0


@dataclass
class ScorerStats:
    n: int = 0
    histogram: dict[int, int] = field(default_factory=dict)
    mean: float | None = None


@dataclass
class RunReport:
    run_id: int
    status: str
    pass1: PassOneStats = field(default_factory=PassOneStats)
    pass2_success: int = 0
    pass2_error: int = 0
    pass2_pending: int = 0
    scorers: dict[str, ScorerStats] = field(default_factory=dict)
    per_spec_pass1: dict[str, PassOneStats] = field(default_factory=dict)
    per_spec_judge_mean: dict[str, float] = field(default_factory=dict)


def compute_report(db_path: Path, run_id: int) -> RunReport:
    with session_scope(db_path) as session:
        run = session.execute(select(Run).where(Run.run_id == run_id)).scalar_one()
        rep = RunReport(run_id=run_id, status=run.status)

        # Pass 1
        p1_rows = session.execute(
            select(Pass1Result, Query)
            .join(Query, Pass1Result.query_id == Query.query_id)
            .where(Pass1Result.run_id == run_id)
        ).all()

        rep.pass1.total = len(p1_rows)
        per_spec: dict[str, PassOneStats] = defaultdict(PassOneStats)
        for p, q in p1_rows:
            stats = rep.pass1
            stats_buckets: list[PassOneStats] = [stats]
            for spec in (q.specializations or ["?"]):
                bucket = per_spec[spec]
                bucket.total += 1
                stats_buckets.append(bucket)

            if p.status == "success":
                for sb in stats_buckets:
                    sb.success += 1
                if p.meets_minimum_tier == 1:
                    for sb in stats_buckets:
                        sb.meets_min_tier += 1
                if p.matches_specialization == 1:
                    for sb in stats_buckets:
                        sb.matches_spec += 1
                if p.router_selected_tier is None:
                    for sb in stats_buckets:
                        sb.unknown_tier += 1
            elif p.status == "error":
                for sb in stats_buckets:
                    sb.error += 1
            else:
                for sb in stats_buckets:
                    sb.pending += 1
        rep.per_spec_pass1 = dict(per_spec)

        # Pass 2
        p2_rows = session.execute(
            select(Pass2Result).where(Pass2Result.run_id == run_id)
        ).scalars().all()
        for p2 in p2_rows:
            if p2.status == "success":
                rep.pass2_success += 1
            elif p2.status == "error":
                rep.pass2_error += 1
            else:
                rep.pass2_pending += 1

        # Scores: group by scorer/reviewer_id
        score_rows = session.execute(
            select(Score, Query)
            .join(Query, Score.query_id == Query.query_id)
            .where(Score.run_id == run_id)
        ).all()
        scorer_buckets: dict[str, list[int]] = defaultdict(list)
        spec_judge_scores: dict[str, list[int]] = defaultdict(list)
        for sc, q in score_rows:
            key = f"{sc.scorer}:{sc.reviewer_id}"
            scorer_buckets[key].append(sc.score)
            if sc.scorer == "judge":
                for spec in (q.specializations or ["?"]):
                    spec_judge_scores[spec].append(sc.score)

        for key, scores in scorer_buckets.items():
            hist: dict[int, int] = {}
            for s in scores:
                hist[s] = hist.get(s, 0) + 1
            rep.scorers[key] = ScorerStats(
                n=len(scores),
                histogram=hist,
                mean=round(mean(scores), 3) if scores else None,
            )
        for spec, scores in spec_judge_scores.items():
            if scores:
                rep.per_spec_judge_mean[spec] = round(mean(scores), 3)

    return rep


def _pct(num: int, denom: int) -> str:
    return f"{(100.0 * num / denom):5.1f}%" if denom else "  n/a"


def render_console(rep: RunReport, console: Console | None = None) -> None:
    console = console or Console()
    console.rule(f"[bold]Run {rep.run_id}  ({rep.status})[/]")

    p1 = rep.pass1
    t = Table(title="Pass 1 — Routing accuracy", show_header=True, header_style="bold")
    t.add_column("metric")
    t.add_column("count", justify="right")
    t.add_column("of success", justify="right")
    t.add_row("total queries", str(p1.total), "")
    t.add_row("success", str(p1.success), "")
    t.add_row("errors", str(p1.error), "")
    t.add_row("pending", str(p1.pending), "")
    t.add_row("meets min tier", str(p1.meets_min_tier), _pct(p1.meets_min_tier, p1.success))
    t.add_row("matches spec", str(p1.matches_spec), _pct(p1.matches_spec, p1.success))
    t.add_row("unknown tier", str(p1.unknown_tier), _pct(p1.unknown_tier, p1.success))
    console.print(t)

    t2 = Table(title="Pass 2 — Response generation", show_header=True, header_style="bold")
    t2.add_column("status")
    t2.add_column("count", justify="right")
    t2.add_row("success", str(rep.pass2_success))
    t2.add_row("error", str(rep.pass2_error))
    t2.add_row("pending", str(rep.pass2_pending))
    console.print(t2)

    if rep.scorers:
        t3 = Table(title="Scoring", show_header=True, header_style="bold")
        t3.add_column("scorer:reviewer")
        t3.add_column("n", justify="right")
        t3.add_column("mean", justify="right")
        t3.add_column("histogram")
        for key, st in sorted(rep.scorers.items()):
            hist_str = " ".join(f"{k}:{v}" for k, v in sorted(st.histogram.items()))
            t3.add_row(key, str(st.n), str(st.mean), hist_str)
        console.print(t3)

    if rep.per_spec_pass1:
        t4 = Table(title="Per-specialization breakdown", show_header=True, header_style="bold")
        t4.add_column("specialization")
        t4.add_column("total", justify="right")
        t4.add_column("meets tier", justify="right")
        t4.add_column("matches spec", justify="right")
        t4.add_column("judge mean", justify="right")
        for spec in sorted(rep.per_spec_pass1):
            st = rep.per_spec_pass1[spec]
            jm = rep.per_spec_judge_mean.get(spec)
            t4.add_row(
                spec,
                str(st.total),
                _pct(st.meets_min_tier, st.success),
                _pct(st.matches_spec, st.success),
                f"{jm}" if jm is not None else "—",
            )
        console.print(t4)


def to_dict(rep: RunReport) -> dict[str, Any]:
    return {
        "run_id": rep.run_id,
        "status": rep.status,
        "pass1": {
            "total": rep.pass1.total,
            "success": rep.pass1.success,
            "error": rep.pass1.error,
            "pending": rep.pass1.pending,
            "meets_min_tier": rep.pass1.meets_min_tier,
            "matches_spec": rep.pass1.matches_spec,
            "unknown_tier": rep.pass1.unknown_tier,
        },
        "pass2": {
            "success": rep.pass2_success,
            "error": rep.pass2_error,
            "pending": rep.pass2_pending,
        },
        "scorers": {
            k: {"n": v.n, "mean": v.mean, "histogram": v.histogram}
            for k, v in rep.scorers.items()
        },
        "per_spec_pass1": {
            k: {
                "total": v.total,
                "success": v.success,
                "meets_min_tier": v.meets_min_tier,
                "matches_spec": v.matches_spec,
            }
            for k, v in rep.per_spec_pass1.items()
        },
        "per_spec_judge_mean": rep.per_spec_judge_mean,
    }


def export_json(rep: RunReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(rep), indent=2, sort_keys=True))


def export_csv(rep: RunReport, path: Path) -> None:
    """Flat per-spec CSV with topline metrics — convenient for spreadsheets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "run_id", "status", "specialization", "total", "success",
            "meets_min_tier", "matches_spec", "judge_mean",
        ])
        # Topline as one row labelled '_all'.
        w.writerow([
            rep.run_id, rep.status, "_all",
            rep.pass1.total, rep.pass1.success,
            rep.pass1.meets_min_tier, rep.pass1.matches_spec,
            rep.scorers.get(next(iter(rep.scorers), ""), ScorerStats()).mean
            if rep.scorers else "",
        ])
        for spec in sorted(rep.per_spec_pass1):
            st = rep.per_spec_pass1[spec]
            w.writerow([
                rep.run_id, rep.status, spec,
                st.total, st.success,
                st.meets_min_tier, st.matches_spec,
                rep.per_spec_judge_mean.get(spec, ""),
            ])
