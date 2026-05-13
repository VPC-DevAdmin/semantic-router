"""Typer CLI entrypoint.

Surface (one command per make target):

  init-db        create the SQLite schema
  load           upsert data/queries.json into the DB (gold from `expected_answer`)
  route          for each query: send through router, capture routing decision
  answers        for each query × each tier: call the tier backend directly (TODO)
  export         emit demo.json from the DB (TODO)
  resume         continue an in-progress run over pending/error rows
  clean-results  wipe runs/results; preserves queries and gold
  router-smoke   one-shot routing diagnostic
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from .config import load_models, load_router_process
from .db import DEFAULT_DB_PATH, init_db
from .load import load_into_db
from .pass1 import run_pass1
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


# ---------- Helpers shared by route/resume ----------

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
    seed_pending(db, rid, only=only)
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


@app.command("resume")
def resume_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH),
    router_config: Path = typer.Option(DEFAULT_ROUTER_CONFIG),
    models: Path = typer.Option(DEFAULT_MODELS),
    run: int | None = typer.Option(None, "--run"),
    concurrency: int = typer.Option(8, "--concurrency"),
) -> None:
    """Resume a run: re-process pending/error route rows; mark done if clean.

    Currently only resumes `make route` (pass 1). `make answers` (per-tier
    response collection) is not yet implemented; once it lands this command
    will also resume that pass.
    """
    rid = _resolve_run(db, run)
    proc_cfg = load_router_process(router_config)
    lookup = TierLookup(load_models(models))

    async def _go() -> int:
        async with RouterProcess(proc_cfg):
            client = RouterClient(proc_cfg, lookup)
            r1 = await run_pass1(db, rid, router_client=client, concurrency=concurrency)
            console.print("[bold]route[/]")
            console.print(str(r1))
        if r1.errors == 0:
            mark_finished(db, rid, status="done")
            console.print(f"[green]run[/] {rid} finished")
        else:
            console.print(f"[yellow]run[/] {rid} still has errors; re-run `resume`")
        return 0

    raise typer.Exit(code=asyncio.run(_go()))


# ---------- Utility ----------

@app.command("clean-results")
def clean_results_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt."),
) -> None:
    """Wipe runs and per-pass results. Preserves queries and gold answers."""
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
