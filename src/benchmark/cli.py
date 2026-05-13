"""Typer CLI entrypoint.

Surface (one command per make target):

  init-db        create the SQLite schema
  load           upsert data/queries.json into the DB (gold from `expected_answer`)
  route          pass 1: send each query through the router, capture routing decision
  answer         pass 2: send each query through the router for full LLM responses
  resume         continue an in-progress run (route + answer over pending/error rows)
  judge          LLM-as-judge scoring of `answer` responses
  review         human scoring TUI
  report         aggregate stats + JSON/CSV export
  clean-results  wipe runs/results/scores; preserves queries and gold
  router-smoke   one-shot routing diagnostic
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from sqlalchemy import select

from .config import load_models, load_router_process
from .db import DEFAULT_DB_PATH, Query, init_db, session_scope
from .judge import judge_run
from .load import load_into_db
from .pass1 import run_pass1
from .pass2 import run_pass2
from .report import compute_report, export_csv, export_json, render_console
from .review import human_review
from .router_client import RouterClient, TierLookup
from .router_proc import RouterProcess
from .runs import (
    clean_results,
    create_run,
    latest_active_run,
    mark_finished,
    seed_pending,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

DEFAULT_QUERIES = Path("data/queries.json")
DEFAULT_MODELS = Path("config/models.yaml")
DEFAULT_ROUTER_CONFIG = Path("config/router.yaml")
DEFAULT_JUDGE_CONFIG = Path("config/judge.yaml")
DEFAULT_SCORING_CONFIG = Path("config/scoring.yaml")


# ---------- DB / data ----------

@app.command("init-db")
def init_db_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH, help="Path to SQLite database."),
) -> None:
    """Create the SQLite schema if it doesn't exist."""
    path = init_db(db)
    console.print(f"[green]initialized[/] {path}")


@app.command("load")
def load_cmd(
    queries: Path = typer.Option(DEFAULT_QUERIES, help="Path to queries.json."),
    db: Path = typer.Option(DEFAULT_DB_PATH, help="Path to SQLite database."),
) -> None:
    """Load queries.json into the DB. Gold answers come from `expected_answer`. Idempotent."""
    if not db.exists():
        init_db(db)
        console.print(f"[yellow]created[/] {db}")
    report = load_into_db(queries, db)
    console.print(f"[green]loaded[/] from {queries}")
    console.print(str(report))


# ---------- Helpers shared by route/answer/resume ----------

def _tts_only_query_ids(db: Path) -> set[str]:
    """Queries whose specializations are exclusively `tts`. Skipped from `answer`."""
    out: set[str] = set()
    with session_scope(db) as session:
        for q in session.execute(select(Query)).scalars():
            specs = list(q.specializations or [])
            if specs and all(s == "tts" for s in specs):
                out.add(q.query_id)
    return out


def _resolve_run(db: Path, run: int | None) -> int:
    if run is not None:
        return run
    rid = latest_active_run(db)
    if rid is None:
        console.print(
            "[red]error[/]: no active run; pass --run RUN or start one with `make route`"
        )
        raise typer.Exit(code=2)
    return rid


def _ensure_run(
    db: Path, router_config: Path, models: Path, only: list[str] | None, notes: str | None
) -> int:
    """Find the latest active run, or create a fresh one and seed pending rows."""
    rid = latest_active_run(db)
    if rid is not None:
        return rid
    rid = create_run(
        db, router_config_path=router_config, models_config_path=models, notes=notes
    )
    skip = _tts_only_query_ids(db)
    seed_pending(db, rid, only=only, skip_query_ids=skip)
    return rid


# ---------- Pass 1 (routing) and Pass 2 (answers) ----------

@app.command("route")
def route_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH),
    router_config: Path = typer.Option(DEFAULT_ROUTER_CONFIG),
    models: Path = typer.Option(DEFAULT_MODELS),
    concurrency: int = typer.Option(8, "--concurrency"),
    query_id: list[str] = typer.Option(
        [], "--query-id", help="Restrict to these query IDs (repeatable)."
    ),
    notes: str = typer.Option("", "--notes"),
) -> None:
    """Pass 1: send each query through the router and capture the routing decision.

    Generation is capped at max_tokens=1 — we only care about which model the
    router selects, not the response. Resumable.
    """
    if not db.exists():
        console.print(f"[red]error[/]: db {db} does not exist; run `make setup` first")
        raise typer.Exit(code=2)

    proc_cfg = load_router_process(router_config)
    lookup = TierLookup(load_models(models))
    only = list(query_id) or None

    async def _go() -> int:
        rid = _ensure_run(db, router_config, models, only, notes or None)
        console.print(f"[green]route[/] run={rid}")
        async with RouterProcess(proc_cfg):
            client = RouterClient(proc_cfg, lookup)
            report = await run_pass1(db, rid, router_client=client, concurrency=concurrency)
        console.print(str(report))
        return 0

    raise typer.Exit(code=asyncio.run(_go()))


@app.command("answer")
def answer_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH),
    router_config: Path = typer.Option(DEFAULT_ROUTER_CONFIG),
    models: Path = typer.Option(DEFAULT_MODELS),
    run: int | None = typer.Option(None, "--run", help="Run id (default: latest active)."),
    concurrency: int = typer.Option(8, "--concurrency"),
    max_tokens: int = typer.Option(2048, "--max-tokens"),
) -> None:
    """Pass 2: send each query through the router for the full LLM response.

    The routing decision is captured again (cheap) but the focus is the
    response text. Resumable.
    """
    rid = _resolve_run(db, run)
    proc_cfg = load_router_process(router_config)
    lookup = TierLookup(load_models(models))

    async def _go() -> int:
        async with RouterProcess(proc_cfg):
            client = RouterClient(proc_cfg, lookup)
            report = await run_pass2(
                db, rid, router_client=client, concurrency=concurrency, max_tokens=max_tokens
            )
        console.print(f"[green]answer[/] run={rid}")
        console.print(str(report))
        return 0

    raise typer.Exit(code=asyncio.run(_go()))


@app.command("resume")
def resume_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH),
    router_config: Path = typer.Option(DEFAULT_ROUTER_CONFIG),
    models: Path = typer.Option(DEFAULT_MODELS),
    run: int | None = typer.Option(None, "--run"),
    concurrency: int = typer.Option(8, "--concurrency"),
    answer_max_tokens: int = typer.Option(2048, "--answer-max-tokens"),
) -> None:
    """Resume a run: re-process pending/error rows for route then answer; mark done if clean."""
    rid = _resolve_run(db, run)
    proc_cfg = load_router_process(router_config)
    lookup = TierLookup(load_models(models))

    async def _go() -> int:
        async with RouterProcess(proc_cfg):
            client = RouterClient(proc_cfg, lookup)
            r1 = await run_pass1(db, rid, router_client=client, concurrency=concurrency)
            console.print("[bold]route[/]")
            console.print(str(r1))
            r2 = await run_pass2(
                db, rid, router_client=client,
                concurrency=concurrency, max_tokens=answer_max_tokens,
            )
            console.print("[bold]answer[/]")
            console.print(str(r2))
        if r1.errors == 0 and r2.errors == 0:
            mark_finished(db, rid, status="done")
            console.print(f"[green]run[/] {rid} finished")
        else:
            console.print(f"[yellow]run[/] {rid} still has errors; re-run `resume`")
        return 0

    raise typer.Exit(code=asyncio.run(_go()))


# ---------- Scoring ----------

@app.command("judge")
def judge_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH),
    run: int | None = typer.Option(None, "--run"),
    judge_config: Path = typer.Option(DEFAULT_JUDGE_CONFIG),
    scoring_config: Path = typer.Option(DEFAULT_SCORING_CONFIG),
    concurrency: int = typer.Option(4, "--concurrency"),
) -> None:
    """LLM-as-judge scoring of `answer` responses against gold."""
    rid = _resolve_run(db, run)
    report = asyncio.run(
        judge_run(
            db_path=db, run_id=rid,
            judge_config_path=judge_config,
            scoring_config_path=scoring_config,
            concurrency=concurrency,
        )
    )
    console.print(f"[bold]judge[/] (run {rid})")
    console.print(str(report))
    if report.parse_errors or report.other_errors:
        raise typer.Exit(code=1)


@app.command("review")
def review_cmd(
    reviewer: str = typer.Option(..., "--reviewer", help="Reviewer id (e.g. your username)."),
    db: Path = typer.Option(DEFAULT_DB_PATH),
    run: int | None = typer.Option(None, "--run"),
    scoring_config: Path = typer.Option(DEFAULT_SCORING_CONFIG),
    sample: int | None = typer.Option(None, "--sample"),
    by: str | None = typer.Option(None, "--by"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Interactive human scoring TUI. Resumable per reviewer."""
    rid = _resolve_run(db, run)
    if sample is not None and by is None:
        by = "specialization"
    report = human_review(
        db_path=db, run_id=rid,
        reviewer_id=reviewer,
        scoring_config_path=scoring_config,
        sample=sample, by=by, seed=seed,
    )
    console.print(f"[bold]review[/] (run {rid}, reviewer={reviewer})")
    console.print(str(report))


@app.command("report")
def report_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH),
    run: int | None = typer.Option(None, "--run"),
    json_out: Path | None = typer.Option(None, "--json", help="Write JSON to this path."),
    csv_out: Path | None = typer.Option(None, "--csv", help="Write CSV to this path."),
) -> None:
    """Aggregate stats for a run. Stdout summary plus optional JSON/CSV export."""
    rid = _resolve_run(db, run)
    rep = compute_report(db, rid)
    render_console(rep, console)
    if json_out:
        export_json(rep, json_out)
        console.print(f"[green]wrote[/] {json_out}")
    if csv_out:
        export_csv(rep, csv_out)
        console.print(f"[green]wrote[/] {csv_out}")


# ---------- Utility ----------

@app.command("clean-results")
def clean_results_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt."),
) -> None:
    """Wipe runs/results/scores. Preserves queries and gold answers."""
    if not yes and not typer.confirm("This will delete all run data. Continue?"):
        raise typer.Exit(code=1)
    deleted = clean_results(db)
    for table, n in deleted.items():
        console.print(f"  deleted {n:>5} from {table}")


@app.command("router-smoke")
def router_smoke_cmd(
    prompt: str = typer.Argument(..., help="Prompt to send through the router."),
    router_config: Path = typer.Option(DEFAULT_ROUTER_CONFIG),
    models: Path = typer.Option(DEFAULT_MODELS),
    max_tokens: int = typer.Option(64, "--max-tokens"),
) -> None:
    """Boot router, send one prompt, print decision, tear down. Diagnostic."""
    cfg = load_router_process(router_config)
    lookup = TierLookup(load_models(models))

    async def _run() -> int:
        async with RouterProcess(cfg):
            client = RouterClient(cfg, lookup)
            result = await client.chat(prompt, max_tokens=max_tokens)
        d = result.decision
        console.print(f"[green]selected_model[/]: {d.selected_model}")
        console.print(f"[green]selected_tier[/]:  {d.selected_tier}")
        console.print(f"[green]category[/]:       {d.category}")
        console.print(f"[green]reasoning[/]:      {d.reasoning}")
        console.print(f"[green]cache_hit[/]:      {d.cache_hit}")
        console.print(f"[green]latency_ms[/]:     {result.latency_ms}")
        console.print(
            f"[green]tokens[/]:         prompt={result.prompt_tokens} "
            f"completion={result.completion_tokens}"
        )
        console.print(f"\n[bold]response[/]:\n{result.content}")
        return 0

    raise typer.Exit(code=asyncio.run(_run()))


if __name__ == "__main__":
    app()
