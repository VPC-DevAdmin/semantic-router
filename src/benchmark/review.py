"""Human scoring TUI.

Shows pending pass2_results that have not been scored by the named reviewer,
displays prompt / gold / router response side-by-side via Rich, prompts for a
score and optional rationale, and per-row commits to `scores`. Resumable: only
unreviewed rows are shown.

For 10k+ scale a full human pass is impractical; `--sample N --by spec`
performs stratified sampling across specializations so the reviewer covers
the distribution. The judge (M5) handles full-coverage scoring.

The TUI is split from input/output so tests can drive it without a TTY: the
public function `human_review` takes injectable `ask_score` and `ask_rationale`
callables.
"""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from sqlalchemy import select

from .config import ScoringConfig, load_scoring
from .db import Pass2Result, Query, Score, session_scope

SKIP = "__skip__"
QUIT = "__quit__"


@dataclass
class ReviewReport:
    candidates: int = 0
    reviewed: int = 0
    skipped: int = 0
    quit_early: bool = False
    score_histogram: dict[int, int] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"  candidates: {self.candidates}",
            f"  reviewed:   {self.reviewed}",
            f"  skipped:    {self.skipped}",
            f"  quit early: {self.quit_early}",
        ]
        if self.score_histogram:
            lines.append("  histogram:")
            for k in sorted(self.score_histogram):
                lines.append(f"    {k}: {self.score_histogram[k]}")
        return "\n".join(lines)


def _stratified_sample(
    items: list[dict], n: int, key: Callable[[dict], str], rng: random.Random
) -> list[dict]:
    """Approximately-proportional stratified sample of size n grouped by `key`."""
    if n >= len(items):
        return items
    buckets: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        buckets[key(it)].append(it)
    total = len(items)
    out: list[dict] = []
    # Allocate per-bucket and pull a random subset.
    for bucket in buckets.values():
        share = max(1, round(n * len(bucket) / total))
        rng.shuffle(bucket)
        out.extend(bucket[:share])
    rng.shuffle(out)
    return out[:n]


def _gather_candidates(
    db_path: Path,
    run_id: int,
    reviewer_id: str,
) -> list[dict]:
    with session_scope(db_path) as session:
        rows = session.execute(
            select(Pass2Result, Query)
            .join(Query, Pass2Result.query_id == Query.query_id)
            .where(Pass2Result.run_id == run_id)
            .where(Pass2Result.status == "success")
        ).all()
        scored = {
            r[0]
            for r in session.execute(
                select(Score.query_id)
                .where(Score.run_id == run_id)
                .where(Score.scorer == "human")
                .where(Score.reviewer_id == reviewer_id)
            ).all()
        }
        return [
            {
                "query_id": p2.query_id,
                "prompt": q.prompt,
                "gold": q.gold_answer or "(no gold answer)",
                "response": p2.response_text or "",
                "specializations": list(q.specializations or []),
                "router_selected_model": p2.router_selected_model,
            }
            for (p2, q) in rows
            if p2.query_id not in scored and q.gold_answer
        ]


def _render(console: Console, item: dict, rubric: ScoringConfig) -> None:
    console.rule(f"[bold]{item['query_id']}[/]  ({', '.join(item['specializations'])})")
    if item.get("router_selected_model"):
        console.print(f"router picked: [cyan]{item['router_selected_model']}[/]")
    console.print(Panel(item["prompt"], title="Prompt", border_style="blue"))
    console.print(Panel(item["gold"], title="Gold", border_style="green"))
    console.print(Panel(item["response"], title="Router response", border_style="magenta"))
    scale_lines = [f"  {k}: {v}" for k, v in sorted(rubric.scale.items())]
    console.print("[bold]Scale[/]\n" + "\n".join(scale_lines))


def human_review(
    db_path: Path,
    run_id: int,
    *,
    reviewer_id: str,
    scoring_config_path: Path,
    sample: int | None = None,
    by: str | None = None,
    seed: int = 0,
    ask_score: Callable[[ScoringConfig], str] | None = None,
    ask_rationale: Callable[[], str] | None = None,
    console: Console | None = None,
    rubric: ScoringConfig | None = None,
) -> ReviewReport:
    """Drive the human review session.

    `ask_score` returns one of: "1".."5" (or wider per rubric), "s" to skip,
    "q" to quit. `ask_rationale` returns a possibly-empty string.
    """
    rubric = rubric or load_scoring(scoring_config_path)
    console = console or Console()
    ask_score = ask_score or _default_ask_score(console)
    ask_rationale = ask_rationale or _default_ask_rationale(console)

    candidates = _gather_candidates(db_path, run_id, reviewer_id)
    report = ReviewReport(candidates=len(candidates))

    if sample is not None and by:
        rng = random.Random(seed)
        candidates = _stratified_sample(
            candidates,
            sample,
            key=lambda it: (it["specializations"] or ["?"])[0] if by == "specialization" else "?",
            rng=rng,
        )
    elif sample is not None:
        rng = random.Random(seed)
        rng.shuffle(candidates)
        candidates = candidates[:sample]

    max_score = max(rubric.scale)

    for item in candidates:
        _render(console, item, rubric)
        raw = ask_score(rubric).strip().lower()
        if raw == "q":
            report.quit_early = True
            break
        if raw == "s":
            report.skipped += 1
            continue
        try:
            score = int(raw)
        except ValueError:
            console.print(f"[red]invalid input {raw!r}; skipping[/]")
            report.skipped += 1
            continue
        if not 1 <= score <= max_score:
            console.print(f"[red]score must be in 1..{max_score}; skipping[/]")
            report.skipped += 1
            continue

        rationale = ask_rationale()
        with session_scope(db_path) as session:
            session.add(
                Score(
                    run_id=run_id,
                    query_id=item["query_id"],
                    scorer="human",
                    reviewer_id=reviewer_id,
                    score=score,
                    rubric_version=rubric.rubric_version,
                    rationale=rationale or None,
                    scored_at=datetime.now(UTC),
                )
            )
        report.reviewed += 1
        report.score_histogram[score] = report.score_histogram.get(score, 0) + 1

    return report


def _default_ask_score(console: Console) -> Callable[[ScoringConfig], str]:
    def ask(rubric: ScoringConfig) -> str:
        console.print(
            f"[bold]Score 1..{max(rubric.scale)}[/], "
            f"[yellow]s[/] to skip, [red]q[/] to quit: ",
            end="",
        )
        return input()

    return ask


def _default_ask_rationale(console: Console) -> Callable[[], str]:
    def ask() -> str:
        console.print("[dim]Optional rationale (enter to skip):[/] ", end="")
        return input()

    return ask
