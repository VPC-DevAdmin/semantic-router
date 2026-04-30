"""Typer CLI entrypoint.

Surfaces M1 commands: `init-db` and `seed`. Later milestones add `gold`, `run`,
`pass1`, `pass2`, `review`, `judge`, `report`.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from .config import load_models, load_queries, load_router_process
from .db import DEFAULT_DB_PATH, init_db
from .gold import generate_gold
from .router_client import RouterClient, TierLookup
from .router_proc import RouterProcess
from .seed import seed_from_yaml

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

DEFAULT_QUERIES = Path("data/queries.yaml")
DEFAULT_MODELS = Path("config/models.yaml")
DEFAULT_GOLD_CONFIG = Path("config/gold.yaml")
DEFAULT_GOLD_DIR = Path("data/gold")
DEFAULT_ROUTER_CONFIG = Path("config/router.yaml")


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
