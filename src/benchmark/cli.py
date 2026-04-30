"""Typer CLI entrypoint.

Surfaces M1 commands: `init-db` and `seed`. Later milestones add `gold`, `run`,
`pass1`, `pass2`, `review`, `judge`, `report`.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from .config import load_models, load_queries
from .db import DEFAULT_DB_PATH, init_db
from .gold import generate_gold
from .seed import seed_from_yaml

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

DEFAULT_QUERIES = Path("data/queries.yaml")
DEFAULT_MODELS = Path("config/models.yaml")
DEFAULT_GOLD_CONFIG = Path("config/gold.yaml")
DEFAULT_GOLD_DIR = Path("data/gold")


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
