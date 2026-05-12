"""Typer CLI entrypoint.

Surfaces M1 commands: `init-db` and `seed`. Later milestones add `gold`, `run`,
`pass1`, `pass2`, `review`, `judge`, `report`.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from sqlalchemy import select

from .config import load_models, load_queries, load_router_process
from .db import DEFAULT_DB_PATH, Query, init_db, session_scope
from .gold import generate_gold
from .judge import judge_run
from .pass1 import run_pass1
from .pass2 import run_pass2
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
from .seed import seed_from_yaml

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

DEFAULT_QUERIES = Path("data/queries.yaml")
DEFAULT_MODELS = Path("config/models.yaml")
DEFAULT_GOLD_CONFIG = Path("config/gold.yaml")
DEFAULT_GOLD_DIR = Path("data/gold")
DEFAULT_ROUTER_CONFIG = Path("config/router.yaml")
DEFAULT_JUDGE_CONFIG = Path("config/judge.yaml")
DEFAULT_SCORING_CONFIG = Path("config/scoring.yaml")


@app.command("init-db")
def init_db_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH, help="Path to SQLite database."),
) -> None:
    """Create the SQLite schema if it doesn't exist."""
    path = init_db(db)
    console.print(f"[green]initialized[/] {path}")


@app.command("seed")
def seed_cmd(
    queries: Path = typer.Option(DEFAULT_QUERIES, help="Path to queries.yaml."),
    db: Path = typer.Option(DEFAULT_DB_PATH, help="Path to SQLite database."),
) -> None:
    """Upsert curated queries into the DB. Idempotent."""
    if not db.exists():
        init_db(db)
        console.print(f"[yellow]created[/] {db}")
    report = seed_from_yaml(queries, db)
    console.print(f"[green]seeded[/] from {queries}")
    console.print(str(report))


@app.command("gold")
def gold_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH, help="Path to SQLite database."),
    gold_config: Path = typer.Option(DEFAULT_GOLD_CONFIG, help="Path to gold.yaml."),
    gold_dir: Path = typer.Option(DEFAULT_GOLD_DIR, help="Directory for gold .md files."),
    refresh: bool = typer.Option(False, "--refresh", help="Regenerate even if gold exists."),
    query_id: list[str] = typer.Option(
        [], "--query-id", help="Restrict to these query IDs (repeatable)."
    ),
    concurrency: int = typer.Option(4, "--concurrency", help="Max concurrent calls."),
) -> None:
    """Generate or refresh gold answers via the configured gold endpoint."""
    if not db.exists():
        console.print(f"[red]error[/]: db {db} does not exist; run `make setup` first")
        raise typer.Exit(code=2)
    report = asyncio.run(
        generate_gold(
            db_path=db,
            gold_config_path=gold_config,
            gold_dir=gold_dir,
            refresh=refresh,
            only=query_id or None,
            concurrency=concurrency,
        )
    )
    console.print("[green]gold[/]")
    console.print(str(report))
    if report.errors:
        raise typer.Exit(code=1)


@app.command("router-smoke")
def router_smoke_cmd(
    prompt: str = typer.Argument(..., help="Prompt to send through the router."),
    router_config: Path = typer.Option(DEFAULT_ROUTER_CONFIG, help="Path to router.yaml."),
    models: Path = typer.Option(DEFAULT_MODELS, help="Path to models.yaml (for tier lookup)."),
    max_tokens: int = typer.Option(64, "--max-tokens", help="Cap generation."),
) -> None:
    """Boot the router (or attach to an external one), send one prompt, print decision, tear down.

    Use this as the M3 acceptance check. Requires `vllm-sr` installed unless
    `external: true` is set in router.yaml.
    """
    cfg = load_router_process(router_config)
    lookup = TierLookup(load_models(models))

    async def _run() -> int:
        async with RouterProcess(cfg) as _:
            client = RouterClient(cfg, lookup)
            result = await client.chat(prompt, max_tokens=max_tokens)
        d = result.decision
        console.print(f"[green]selected_model[/]: {d.selected_model}")
        console.print(f"[green]selected_tier[/]:  {d.selected_tier}")
        console.print(f"[green]category[/]:       {d.category}")
        console.print(f"[green]reasoning[/]:      {d.reasoning}")
        console.print(f"[green]cache_hit[/]:      {d.cache_hit}")
        console.print(f"[green]latency_ms[/]:     {result.latency_ms}")
        console.print(f"[green]tokens[/]:         "
                      f"prompt={result.prompt_tokens} completion={result.completion_tokens}")
        console.print()
        console.print(f"[bold]response[/]:\n{result.content}")
        return 0

    raise typer.Exit(code=asyncio.run(_run()))


def _tts_only_query_ids(db: Path) -> set[str]:
    """Queries whose specializations are exclusively `tts`. Skipped from pass2."""
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
        console.print("[red]error[/]: no active run; pass --run RUN or use `make run`")
        raise typer.Exit(code=2)
    return rid


@app.command("run")
def run_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH, help="Path to SQLite database."),
    router_config: Path = typer.Option(DEFAULT_ROUTER_CONFIG, help="Path to router.yaml."),
    models: Path = typer.Option(DEFAULT_MODELS, help="Path to models.yaml."),
    skip_pass1: bool = typer.Option(False, "--skip-pass1"),
    skip_pass2: bool = typer.Option(False, "--skip-pass2"),
    concurrency: int = typer.Option(8, "--concurrency"),
    pass2_max_tokens: int = typer.Option(2048, "--pass2-max-tokens"),
    query_id: list[str] = typer.Option(
        [], "--query-id", help="Restrict to these query IDs (repeatable)."
    ),
    notes: str = typer.Option("", "--notes"),
) -> None:
    """Create a new run, boot the router, execute pass1 + pass2, tear down."""
    if not db.exists():
        console.print(f"[red]error[/]: db {db} does not exist; run `make setup` first")
        raise typer.Exit(code=2)

    proc_cfg = load_router_process(router_config)
    lookup = TierLookup(load_models(models))
    only = list(query_id) or None

    async def _go() -> int:
        run_id = create_run(db, router_config_path=router_config,
                            models_config_path=models, notes=notes or None)
        console.print(f"[green]run[/] {run_id} created")
        skip_p2 = _tts_only_query_ids(db)
        seed_pending(db, run_id, only=only, skip_query_ids=skip_p2)

        try:
            async with RouterProcess(proc_cfg):
                client = RouterClient(proc_cfg, lookup)
                if not skip_pass1:
                    p1 = await run_pass1(db, run_id, router_client=client,
                                         concurrency=concurrency)
                    console.print("[bold]pass 1[/]")
                    console.print(str(p1))
                if not skip_pass2:
                    p2 = await run_pass2(db, run_id, router_client=client,
                                         concurrency=concurrency,
                                         max_tokens=pass2_max_tokens)
                    console.print("[bold]pass 2[/]")
                    console.print(str(p2))
            mark_finished(db, run_id, status="done")
            console.print(f"[green]run[/] {run_id} finished")
            return 0
        except BaseException:
            mark_finished(db, run_id, status="aborted")
            raise

    raise typer.Exit(code=asyncio.run(_go()))


@app.command("pass1")
def pass1_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH),
    router_config: Path = typer.Option(DEFAULT_ROUTER_CONFIG),
    models: Path = typer.Option(DEFAULT_MODELS),
    run: int | None = typer.Option(None, "--run", help="Run id (default: latest active)."),
    concurrency: int = typer.Option(8, "--concurrency"),
) -> None:
    """Pass 1 only — routing accuracy. Resumable."""
    run_id = _resolve_run(db, run)
    proc_cfg = load_router_process(router_config)
    lookup = TierLookup(load_models(models))

    async def _go() -> int:
        async with RouterProcess(proc_cfg):
            client = RouterClient(proc_cfg, lookup)
            report = await run_pass1(db, run_id, router_client=client,
                                     concurrency=concurrency)
        console.print(f"[bold]pass 1[/] (run {run_id})")
        console.print(str(report))
        return 0

    raise typer.Exit(code=asyncio.run(_go()))


@app.command("pass2")
def pass2_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH),
    router_config: Path = typer.Option(DEFAULT_ROUTER_CONFIG),
    models: Path = typer.Option(DEFAULT_MODELS),
    run: int | None = typer.Option(None, "--run"),
    concurrency: int = typer.Option(8, "--concurrency"),
    max_tokens: int = typer.Option(2048, "--max-tokens"),
) -> None:
    """Pass 2 only — response generation. Resumable."""
    run_id = _resolve_run(db, run)
    proc_cfg = load_router_process(router_config)
    lookup = TierLookup(load_models(models))

    async def _go() -> int:
        async with RouterProcess(proc_cfg):
            client = RouterClient(proc_cfg, lookup)
            report = await run_pass2(db, run_id, router_client=client,
                                     concurrency=concurrency,
                                     max_tokens=max_tokens)
        console.print(f"[bold]pass 2[/] (run {run_id})")
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
    pass2_max_tokens: int = typer.Option(2048, "--pass2-max-tokens"),
) -> None:
    """Resume a run: re-run pass1 and pass2 against pending/error rows, then mark done."""
    run_id = _resolve_run(db, run)
    proc_cfg = load_router_process(router_config)
    lookup = TierLookup(load_models(models))

    async def _go() -> int:
        async with RouterProcess(proc_cfg):
            client = RouterClient(proc_cfg, lookup)
            p1 = await run_pass1(db, run_id, router_client=client, concurrency=concurrency)
            console.print("[bold]pass 1[/]")
            console.print(str(p1))
            p2 = await run_pass2(db, run_id, router_client=client,
                                 concurrency=concurrency, max_tokens=pass2_max_tokens)
            console.print("[bold]pass 2[/]")
            console.print(str(p2))
        if p1.errors == 0 and p2.errors == 0:
            mark_finished(db, run_id, status="done")
            console.print(f"[green]run[/] {run_id} finished")
        else:
            console.print(f"[yellow]run[/] {run_id} still has errors; re-run `resume`")
        return 0

    raise typer.Exit(code=asyncio.run(_go()))


@app.command("judge")
def judge_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH),
    run: int | None = typer.Option(None, "--run", help="Run id (default: latest active)."),
    judge_config: Path = typer.Option(DEFAULT_JUDGE_CONFIG),
    scoring_config: Path = typer.Option(DEFAULT_SCORING_CONFIG),
    concurrency: int = typer.Option(4, "--concurrency"),
) -> None:
    """LLM-as-judge scoring of pass-2 responses against gold."""
    run_id = _resolve_run(db, run)
    report = asyncio.run(
        judge_run(
            db_path=db,
            run_id=run_id,
            judge_config_path=judge_config,
            scoring_config_path=scoring_config,
            concurrency=concurrency,
        )
    )
    console.print(f"[bold]judge[/] (run {run_id})")
    console.print(str(report))
    if report.parse_errors or report.other_errors:
        raise typer.Exit(code=1)


@app.command("review")
def review_cmd(
    reviewer: str = typer.Option(..., "--reviewer", help="Reviewer id (e.g. your username)."),
    db: Path = typer.Option(DEFAULT_DB_PATH),
    run: int | None = typer.Option(None, "--run"),
    scoring_config: Path = typer.Option(DEFAULT_SCORING_CONFIG),
    sample: int | None = typer.Option(
        None, "--sample", help="Stratified sample size; omit to review all."
    ),
    by: str | None = typer.Option(
        None, "--by",
        help="Sampling stratum: 'specialization' (default when --sample is set).",
    ),
    seed: int = typer.Option(0, "--seed", help="Sampling seed for reproducibility."),
) -> None:
    """Interactive human scoring TUI. Resumable per reviewer."""
    run_id = _resolve_run(db, run)
    if sample is not None and by is None:
        by = "specialization"
    report = human_review(
        db_path=db,
        run_id=run_id,
        reviewer_id=reviewer,
        scoring_config_path=scoring_config,
        sample=sample,
        by=by,
        seed=seed,
    )
    console.print(f"[bold]review[/] (run {run_id}, reviewer={reviewer})")
    console.print(str(report))


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


@app.command("validate-config")
def validate_config_cmd(
    models: Path = typer.Option(DEFAULT_MODELS, help="Path to models.yaml."),
    queries: Path = typer.Option(DEFAULT_QUERIES, help="Path to queries.yaml."),
) -> None:
    """Validate config files without touching the DB."""
    m = load_models(models)
    q = load_queries(queries)
    console.print(f"[green]ok[/] models.yaml: {len(m.tiers)} tier(s)")
    console.print(f"[green]ok[/] queries.yaml: {len(q.queries)} query(ies)")


if __name__ == "__main__":
    app()
