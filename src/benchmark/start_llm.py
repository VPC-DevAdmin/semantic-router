"""Local-engine launcher — backs `make start_LLM` and `make stop_LLM`.

Tier-agnostic by design: the launcher walks every env slot whose URL
resolves to localhost / 127.0.0.1, looks the slot's served model name
up in `config/local_models.yaml`, picks the matching per-CPU-vendor
launch block (amd / intel), and executes the verbatim argv with three
placeholders filled in — `{port}` (from the slot's URL), `{served_name}`
(from the slot's MODEL), and `{container_name}` (rendered from the
recipe's `container_name` template).

Adding a new local model: add a top-level entry in
`config/local_models.yaml` with the launch argv. The engine can be
anything — vLLM in docker, SGLang, a bare binary launched with
`nohup … &`. The launcher just runs the argv.

Stopping is symmetric: same env walk, run the recipe's `stop` argv.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from .config import (
    DEFAULT_LOCAL_MODELS_PATH,
    LocalModelLibrary,
    LocalModelRecipe,
    ModelsConfig,
    TierModel,
    load_models,
    localhost_port,
)

console = Console()

DEFAULT_TIERS_DIR = Path("config/tiers")


# ─────────────────────────────────────────────────────────────────────────
# Subprocess + CPU-vendor helpers (monkey-patchable seams for tests)
# ─────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, streaming output. Echoes the command for transparency."""
    console.print(f"[dim]$ {' '.join(shlex.quote(c) for c in cmd)}[/]")
    return subprocess.run(cmd, check=check)


def detect_cpu_vendor() -> str:
    """Return 'amd' or 'intel' from /proc/cpuinfo's vendor_id.

    Raises on anything we don't recognise — the launcher refuses to
    guess. Only Linux is supported (the harness runs on Linux dev boxes
    + the GPU server; Mac dev is for code, not for `make start_LLM`).
    """
    try:
        info = Path("/proc/cpuinfo").read_text()
    except FileNotFoundError as e:
        raise RuntimeError(
            "no /proc/cpuinfo — `make start_LLM` only supports Linux hosts"
        ) from e
    for line in info.splitlines():
        if line.startswith("vendor_id"):
            vid = line.split(":", 1)[1].strip()
            if vid == "GenuineIntel":
                return "intel"
            if vid == "AuthenticAMD":
                return "amd"
            raise RuntimeError(f"unrecognised CPU vendor_id={vid!r}")
    raise RuntimeError("no vendor_id line in /proc/cpuinfo")


# ─────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────

def _walk_local_slots(models: ModelsConfig) -> list[tuple[TierModel, int]]:
    """Every (TierModel, host_port) pair whose URL hits localhost.

    Dedupes by (served_model_name, port) — if a model+port appears in
    two tiers (unusual but possible), we only launch one container.
    Preserves first-seen order so output is stable.
    """
    seen: set[tuple[str, int]] = set()
    out: list[tuple[TierModel, int]] = []
    for tier in models.tiers:
        for m in tier.resolved_models():
            port = localhost_port(m.url)
            if port is None:
                continue
            key = (m.served_model_name, port)
            if key in seen:
                continue
            seen.add(key)
            out.append((m, port))
    return out


def _render(arg: str, ctx: dict[str, str]) -> str:
    """str.format with a clear error when the recipe references an unknown
    placeholder (anything beyond port / served_name / container_name)."""
    try:
        return arg.format(**ctx)
    except KeyError as e:
        raise ValueError(
            f"recipe argv references unknown placeholder {{{e.args[0]}}}; "
            f"only {{port}}, {{served_name}}, {{container_name}} are supported"
        ) from e


def _build_context(m: TierModel, port: int, recipe: LocalModelRecipe) -> dict[str, str]:
    base = {"port": str(port), "served_name": m.served_model_name}
    container_name = _render(recipe.container_name, base)
    base["container_name"] = container_name
    return base


def _resolve_recipe(library: LocalModelLibrary, m: TierModel) -> LocalModelRecipe:
    recipe = library.recipes.get(m.served_model_name)
    if recipe is None:
        known = sorted(library.recipes)
        raise SystemExit(
            f"no recipe for {m.served_model_name!r} in local_models.yaml. "
            f"Known recipes: {known}. Add one before running this command."
        )
    return recipe


def start_local_engines(
    tiers_dir: Path = DEFAULT_TIERS_DIR,
    library_path: Path = DEFAULT_LOCAL_MODELS_PATH,
    *,
    run: Callable[[list[str]], object] | None = None,
    vendor: str | None = None,
) -> None:
    """For every env slot pointing at localhost: stop any pre-existing
    container with the same name, then launch the recipe's `start` argv."""
    runner = run or _run
    if not library_path.exists():
        console.print(
            f"[red]error[/]: missing local model library at {library_path}.\n"
            f"  Create it with one entry per locally-launched model — see "
            f"the file header for the schema."
        )
        sys.exit(2)
    library = LocalModelLibrary.load(library_path)
    cpu_vendor = vendor or detect_cpu_vendor()
    models = load_models(tiers_dir)
    targets = _walk_local_slots(models)
    if not targets:
        console.print(
            "[dim]no local slots in .env — every TIER{N}_{i}_URL is remote; "
            "nothing to launch.[/]"
        )
        return
    console.print(
        f"[bold]start_LLM[/]: vendor={cpu_vendor}, {len(targets)} local "
        f"slot(s) to launch"
    )
    for m, port in targets:
        recipe = _resolve_recipe(library, m)
        spec = recipe.for_vendor(cpu_vendor)
        ctx = _build_context(m, port, recipe)
        # Idempotent: stop any container left over with the same name.
        runner([_render(a, ctx) for a in spec.stop])
        runner([_render(a, ctx) for a in spec.start])
        console.print(
            f"  [green]launched[/] {ctx['container_name']} "
            f"({m.served_model_name}) on port {port}"
        )


def stop_local_engines(
    tiers_dir: Path = DEFAULT_TIERS_DIR,
    library_path: Path = DEFAULT_LOCAL_MODELS_PATH,
    *,
    run: Callable[[list[str]], object] | None = None,
    vendor: str | None = None,
) -> None:
    """Symmetric: same env walk, run each recipe's `stop` argv."""
    runner = run or _run
    if not library_path.exists():
        return  # nothing to stop if no library was ever defined
    library = LocalModelLibrary.load(library_path)
    cpu_vendor = vendor or detect_cpu_vendor()
    models = load_models(tiers_dir)
    for m, port in _walk_local_slots(models):
        recipe = library.recipes.get(m.served_model_name)
        if recipe is None:
            continue  # nothing to stop for an unrecognised model
        spec = recipe.for_vendor(cpu_vendor)
        ctx = _build_context(m, port, recipe)
        runner([_render(a, ctx) for a in spec.stop])
        console.print(
            f"  [yellow]stopped[/] {ctx['container_name']} "
            f"({m.served_model_name})"
        )


# Back-compat aliases — the CLI calls these names.
start_local_tiers = start_local_engines
stop_local_tiers = stop_local_engines
