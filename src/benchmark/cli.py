"""Typer CLI entrypoint.

Surface (one command per make target):

  init-db        create the SQLite schema
  load           upsert data/queries.json into the DB (golds from `expected_answers[]`)
  route          for each query: send through router, capture routing decision
  answers        for each query × each tier: call the tier backend directly
  export         emit demo.json from the DB
  resume         continue an in-progress run over pending/error rows
  clean-results  wipe runs/results; preserves queries and gold
  router-smoke   one-shot routing diagnostic
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from .answers import run_answers, run_smoke
from .config import load_models, load_router_process
from .db import DEFAULT_DB_PATH, Query, init_db, session_scope
from .export import export_demo_json
from .import_answers import import_answers_file
from .load import load_into_db
from .misroutes import list_misroutes, render_misroutes
from .pass1 import run_pass1
from .router_client import RouterClient, TierLookup
from .router_proc import RouterProcess
from .runs import (
    clean_results,
    create_run,
    latest_active_run,
    mark_finished,
    reset_answers,
    reset_pass1,
    seed_pending,
    seed_pending_answers,
)
from .scores import DEFAULT_APISERVER, report_scores
from .update_gold import update_gold_answers

# Load .env from CWD on every CLI start. Existing env vars win (so shell /
# CI overrides take precedence over file values).
load_dotenv(override=False)

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

DEFAULT_QUERIES = Path("data/queries.json")
DEFAULT_TIERS = Path("config/tiers")
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
    """Load queries.json into the DB. Each query's `expected_answers[]` is
    synced into the gold_answers table (one row per declared model). Idempotent.
    """
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
    models: Path = typer.Option(DEFAULT_TIERS),
    concurrency: int = typer.Option(8, "--concurrency"),
    query_id: list[str] = typer.Option(
        [], "--query-id", help="Restrict to these query IDs (repeatable)."
    ),
    notes: str = typer.Option("", "--notes"),
    run_new: bool = typer.Option(
        False, "--run-new",
        help="Delete existing pass1_results for the active run, then re-seed.",
    ),
) -> None:
    """Pass 1: send each query through the router and capture the routing decision.

    Generation is capped at max_tokens=1 — we only care about which model the
    router selects, not the response. Resumable; rows in status='error' are
    retried automatically on the next invocation. With --run-new, all
    pass1_results for the active run are dropped first.
    """
    if not db.exists():
        console.print(f"[red]error[/]: db {db} does not exist; run `make setup` first")
        raise typer.Exit(code=2)

    proc_cfg = load_router_process(router_config)
    lookup = TierLookup(load_models(models))
    only = list(query_id) or None

    async def _go() -> int:
        rid = _ensure_run(db, router_config, models, only, notes or None)
        if run_new:
            n = reset_pass1(db, rid)
            console.print(f"[yellow]--run-new[/]: deleted {n} pass1_results row(s)")
            from .runs import seed_pending  # noqa: F401  (already imported above)
            seed_pending(db, rid, only=only)
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
    models: Path = typer.Option(DEFAULT_TIERS),
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


@app.command("answers")
def answers_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH),
    models: Path = typer.Option(DEFAULT_TIERS),
    router_config: Path = typer.Option(DEFAULT_ROUTER_CONFIG, help="Used only for run provenance."),
    run: int | None = typer.Option(
        None, "--run", help="Run id (default: latest active or create new)."
    ),
    concurrency: int = typer.Option(8, "--concurrency"),
    max_tokens: int = typer.Option(2048, "--max-tokens"),
    query_id: list[str] = typer.Option(
        [], "--query-id", help="Restrict to these query IDs (repeatable)."
    ),
    tier: int | None = typer.Option(
        None, "--tier",
        help=(
            "Restrict to a single tier level (1-5). Useful for exercising a "
            "just-wired backend without re-hitting other tiers. Combined with "
            "--run-new, only this tier's rows are deleted and re-seeded."
        ),
    ),
    notes: str = typer.Option("", "--notes"),
    run_new: bool = typer.Option(
        False, "--run-new",
        help=(
            "Delete existing tier_answers for the active run, then re-seed. "
            "With --tier, only that tier's rows are deleted."
        ),
    ),
    mock_endpoint: str | None = typer.Option(
        None, "--mock-endpoint",
        help=(
            "Override every tier's endpoint (e.g. http://localhost:8811/v1). "
            "Used for pipeline verification against the local OAI mock."
        ),
    ),
    smoke: bool = typer.Option(
        False, "--smoke",
        help=(
            "Connectivity probe only: send a tiny chat request to every "
            "(tier, model) that a real run would call, report OK/error per "
            "endpoint, and exit. No DB writes, no run id needed. Verifies "
            "URL / API key / model name without spending real tokens."
        ),
    ),
) -> None:
    """For each routed query: call every model the picked tier fronts.

    The tier comes from `pass1_results.router_selected_tier` (set by
    `make route`). One row per (query, tier, model). Errors mark the row
    `status='error'`; the pass keeps going and a subsequent `make answers`
    retries them.

    With `--smoke`, runs a connectivity probe instead — no DB writes.
    """
    # ── smoke path: tiny probe per (tier, model), no DB, no run id ─────
    if smoke:
        models_cfg = load_models(models)
        if tier is not None and not any(t.level == tier for t in models_cfg.tiers):
            console.print(
                f"[red]error[/]: --tier {tier} not present in {models}; "
                f"known levels: {sorted(t.level for t in models_cfg.tiers)}"
            )
            raise typer.Exit(code=2)
        if mock_endpoint:
            console.print(f"[yellow]MOCK[/]: smoke probing against {mock_endpoint}")

        def _smoke_progress(line: str) -> None:
            console.print(line, markup=False, highlight=False)

        smoke_report = asyncio.run(
            run_smoke(
                models_cfg,
                concurrency=concurrency,
                tier_level=tier,
                mock_endpoint=mock_endpoint,
                progress=_smoke_progress,
            )
        )
        console.print("[bold]smoke[/]")
        console.print(str(smoke_report))
        # Exit non-zero on any error so CI / scripts can gate on it.
        raise typer.Exit(code=1 if smoke_report.errors else 0)

    if not db.exists():
        console.print(f"[red]error[/]: db {db} does not exist; run `make setup` first")
        raise typer.Exit(code=2)

    models_cfg = load_models(models)
    only = list(query_id) or None

    if run is not None:
        rid = run
    else:
        rid = latest_active_run(db)
        if rid is None:
            rid = create_run(
                db,
                router_config_path=router_config,
                models_config_path=models,
                notes=notes or None,
            )
            console.print(f"[green]answers[/] created run={rid}")

    if tier is not None and not any(t.level == tier for t in models_cfg.tiers):
        console.print(
            f"[red]error[/]: --tier {tier} not present in {models}; known levels: "
            f"{sorted(t.level for t in models_cfg.tiers)}"
        )
        raise typer.Exit(code=2)

    if run_new:
        n = reset_answers(db, rid, tier_level=tier)
        scope = f" for tier {tier}" if tier is not None else ""
        console.print(f"[yellow]--run-new[/]: deleted {n} tier_answers row(s){scope}")

    seed_result = seed_pending_answers(db, rid, models_cfg, only=only)
    if seed_result.replaced:
        console.print(
            f"[yellow]re-seeded[/] {seed_result.replaced} stale row(s) "
            f"(wrong tier, or a model no longer configured for the routed tier)"
        )
    if seed_result.seeded:
        console.print(
            f"[green]seeded[/] {seed_result.seeded} new pending row(s) "
            f"(one per model in each query's routed tier)"
        )
    if seed_result.skipped_top_tier:
        console.print(
            f"[dim]skipped[/] {seed_result.skipped_top_tier} top-tier-routed "
            f"query(ies) — the top tier is the gold reference; its answers "
            f"come from `make update-gold`, not `make answers`."
        )
    if not (seed_result.seeded or seed_result.replaced):
        if seed_result.kept:
            console.print(
                f"[dim]note[/]: {seed_result.kept} row(s) already at the correct "
                f"tier+model — re-running the worker on any pending/error rows."
            )
        else:
            console.print(
                "[yellow]note[/]: nothing to do — run `make route` first to "
                "record routing decisions."
            )

    if mock_endpoint:
        console.print(f"[yellow]MOCK[/]: routing all tier calls to {mock_endpoint}")
    if tier is not None:
        console.print(f"[yellow]--tier[/]: restricting worker to tier {tier} rows")

    # markup=False/highlight=False: the lines contain "[ 12/110]" which
    # rich would otherwise try to parse as style markup.
    def _progress(line: str) -> None:
        console.print(line, markup=False, highlight=False)

    report = asyncio.run(
        run_answers(
            db,
            rid,
            models=models_cfg,
            concurrency=concurrency,
            max_tokens=max_tokens,
            mock_endpoint=mock_endpoint,
            tier_level=tier,
            progress=_progress,
        )
    )
    console.print(f"[bold]answers[/] (run {rid})")
    console.print(str(report))
    # Errors are now expected (retry on next run); always exit 0.


@app.command("misroutes")
def misroutes_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH),
    run: int | None = typer.Option(
        None, "--run", help="Run id (default: latest run)."
    ),
) -> None:
    """List queries where the router picked a tier BELOW the expected minimum.

    Diagnostic for routing-accuracy tuning. The output groups misroutes by
    expected tier, routed tier, and router category so we can see whether
    the under-routes cluster on one axis (e.g. mostly judgment-heavy queries
    landing in T2) before tuning thresholds in router-exemplars.yaml.
    """
    misroutes = list_misroutes(db, run_id=run)
    # markup=False so query content and bracket-laden output passes through
    # without Rich trying to parse it as styling tags.
    console.print(render_misroutes(misroutes), markup=False)


@app.command("scores")
def scores_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH),
    run: int | None = typer.Option(None, "--run"),
    apiserver: str = typer.Option(
        DEFAULT_APISERVER,
        "--apiserver",
        help="vllm-sr apiserver base URL (defaults to http://localhost:8080).",
    ),
) -> None:
    """For each misroute, fetch per-signal scores from /api/v1/eval and show
    how far each signal is from its threshold.

    Diagnostic for tuning: tells us whether under-routes are "just barely
    missed" (score 0.40 vs threshold 0.42) or "wildly off" (score 0.15).
    Requires the router stack to be up (`make route` or its containers
    still running).
    """
    report = asyncio.run(report_scores(db, run_id=run, apiserver=apiserver))
    console.print(report, markup=False)


@app.command("import-answers")
def import_answers_cmd(
    file: Path = typer.Argument(..., help="Markdown file with ## qNNNNN sections."),
    tier: int = typer.Option(..., "--tier", help="Tier level (1-5) these answers represent."),
    model: str = typer.Option(
        ..., "--model",
        help="The model id these answers are from (per-tier unique key).",
    ),
    provider: str | None = typer.Option(
        None, "--provider",
        help="Optional provider label (Anthropic / OpenAI / Google) → demo.json.",
    ),
    db: Path = typer.Option(DEFAULT_DB_PATH),
    models: Path = typer.Option(DEFAULT_TIERS),
    run: int | None = typer.Option(None, "--run", help="Run id (default: latest active)."),
) -> None:
    """Import pre-generated tier answers from a markdown file.

    Each `## qNNNNN — Title` (or `## qNNNNN: Title`) section's body is
    stored as that query's answer for (tier, model). Useful when answers
    are produced outside the harness — e.g., manual prompting via a chat
    UI, or pre-generated reference responses for a specific model.

    Upsert behavior: if a tier_answers row already exists at
    (run_id, query_id, tier_level, model), its response_text is
    overwritten and status set to 'success'. Otherwise a new row is
    inserted. Idempotent: re-run with the same file+model to refresh.
    """
    if not file.exists():
        console.print(f"[red]error[/]: file not found: {file}")
        raise typer.Exit(code=2)
    if not db.exists():
        console.print(f"[red]error[/]: db {db} does not exist; run `make setup` first")
        raise typer.Exit(code=2)

    models_cfg = load_models(models)
    if not any(t.level == tier for t in models_cfg.tiers):
        console.print(
            f"[red]error[/]: --tier {tier} not present in {models}; known levels: "
            f"{sorted(t.level for t in models_cfg.tiers)}"
        )
        raise typer.Exit(code=2)

    rid = _resolve_run(db, run)
    result = import_answers_file(
        db, rid,
        tier_level=tier,
        model_id=model,
        provider=provider,
        file_path=file,
        models=models_cfg,
    )
    console.print(
        f"[bold]import-answers[/] (run {rid}, tier {tier}, model {model}, "
        f"from {file.name})"
    )
    console.print(str(result))


@app.command("update-gold")
def update_gold_cmd(
    query_id: list[str] = typer.Option(
        [], "--query-id", "-q",
        help="Query ID to regenerate gold for (repeatable). Narrowest scope.",
    ),
    tier: int | None = typer.Option(
        None, "--tier",
        help="Regenerate gold for every query with this expected_min_tier.",
    ),
    db: Path = typer.Option(DEFAULT_DB_PATH),
    models: Path = typer.Option(DEFAULT_TIERS),
    max_tokens: int = typer.Option(2048, "--max-tokens"),
    concurrency: int = typer.Option(4, "--concurrency"),
    yes: bool = typer.Option(
        False, "--yes",
        help="Skip the confirmation prompt for the full-set (no scope) case.",
    ),
) -> None:
    """Regenerate per-provider gold by calling every top-tier model.

    The top tier IS the gold tier. This calls EACH model configured for
    the top tier (Anthropic Opus, OpenAI GPT-5, …) with each query's
    prompt and upserts one `gold_answers` row per (query, top-tier model)
    — so the demo's `expected_answers[]` ends up with a gold per
    provider.

    Scope (most → least specific):
      --query-id q00046 -q q00050   just those queries
      --tier 5                      every query with expected_min_tier == 5
      (no scope)                    EVERY query — prompts for confirmation
                                    unless --yes

    Separate from `make answers` on purpose: answers CONSUMES gold,
    this PRODUCES it. Per-query failures are reported and never clobber
    the existing gold rows.
    """
    if not db.exists():
        console.print(f"[red]error[/]: db {db} does not exist; run `make setup` first")
        raise typer.Exit(code=2)
    if query_id and tier is not None:
        console.print(
            "[red]error[/]: pass --query-id OR --tier, not both"
        )
        raise typer.Exit(code=2)

    from sqlalchemy import select as _select

    if query_id:
        qids = list(query_id)
        scope = f"{len(qids)} named query(ies)"
    elif tier is not None:
        with session_scope(db) as s:
            qids = [
                qid for (qid,) in s.execute(
                    _select(Query.query_id).where(Query.expected_min_tier == tier)
                ).all()
            ]
        scope = f"tier {tier} ({len(qids)} query(ies) with expected_min_tier={tier})"
        if not qids:
            console.print(
                f"[red]error[/]: no queries with expected_min_tier={tier}"
            )
            raise typer.Exit(code=2)
    else:
        # No scope → the whole set. This is the biggest blast radius
        # (a full-set Opus regen) so it's the only path that confirms.
        with session_scope(db) as s:
            qids = [
                qid for (qid,) in s.execute(_select(Query.query_id)).all()
            ]
        scope = f"ALL {len(qids)} query(ies)"
        if not yes:
            confirmed = typer.confirm(
                f"Regenerate gold for {scope} by calling every top-tier "
                f"model? This overwrites the per-provider gold rows and "
                f"costs N × (# top-tier models) calls.",
                default=False,
            )
            if not confirmed:
                console.print("[yellow]aborted[/] (no changes made)")
                raise typer.Exit(code=1)

    if not qids:
        console.print("[red]error[/]: no matching queries")
        raise typer.Exit(code=2)

    console.print(f"[bold]update-gold[/]: regenerating gold for {scope}")
    models_cfg = load_models(models)
    result = asyncio.run(
        update_gold_answers(
            db,
            query_ids=qids,
            models=models_cfg,
            max_tokens=max_tokens,
            concurrency=concurrency,
        )
    )
    console.print(str(result))
    console.print(
        "[dim]note[/]: re-run `make answers` so top-tier rows pick up the "
        "regenerated gold."
    )


@app.command("export")
def export_cmd(
    db: Path = typer.Option(DEFAULT_DB_PATH),
    run: int | None = typer.Option(None, "--run"),
    output: Path = typer.Option(Path("demo.json"), "--output", "-o"),
) -> None:
    """Write demo.json from the DB. Defaults to the latest active run."""
    rid = _resolve_run(db, run)
    summary = export_demo_json(db, rid, output)
    console.print(f"[bold]export[/] (run {rid})")
    console.print(str(summary))


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


@app.command("start-llm")
def start_llm_cmd(
    tiers: Path = typer.Option(DEFAULT_TIERS, "--tiers"),
) -> None:
    """Launch local-CPU tier backends defined in config/tiers/*.yaml.

    Dispatches on each tier's `backend.kind`. Tiers with `kind: remote` or
    `kind: placeholder` are skipped.
    """
    from .start_llm import start_local_tiers
    start_local_tiers(tiers)


@app.command("stop-llm")
def stop_llm_cmd(
    tiers: Path = typer.Option(DEFAULT_TIERS, "--tiers"),
) -> None:
    """Stop local-CPU tier backends defined in config/tiers/*.yaml."""
    from .start_llm import stop_local_tiers
    stop_local_tiers(tiers)


@app.command("router-smoke")
def router_smoke_cmd(
    prompt: str = typer.Argument(..., help="Prompt to send through the router."),
    router_config: Path = typer.Option(DEFAULT_ROUTER_CONFIG),
    models: Path = typer.Option(DEFAULT_TIERS),
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
