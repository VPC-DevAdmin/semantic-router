"""Config loaders.

Configs validated here:
  - config/tiers/*.yaml  : one tier per file — single source of truth for
                           tier metadata, endpoint, identity, and backend
                           provisioning. Replaces the old models.yaml.
  - router.yaml          : process-management config for the router subprocess
  - queries.json         : curated query set with embedded gold answers

Queries are JSON (not YAML) because the source files come from upstream as
JSON and the format is structured data, not human-edited config.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# Specialization names as they appear in queries.json. Extend here if upstream
# data adds a new category.
SPECIALIZATIONS = {
    "general",
    "coding",
    "math",
    "reasoning",
    "creative_writing",
    "vision",
    "tts",
}


class TierModel(BaseModel):
    """One callable model endpoint within a tier.

    A tier can front several models (Anthropic Opus, OpenAI GPT-5,
    Google Gemini, …) so `make answers` shows "how the answer changes
    across providers". Each entry corresponds to one indexed env slot
    `TIER{N}_{i}_*`. `served_model_name` must be unique within a tier
    (it's the per-tier DB / demo.json key). `provider` is an optional
    human label.
    """
    slot: int
    url: str
    served_model_name: str
    api_key_env: str | None = None
    provider: str | None = None
    timeout_s: int = 60
    max_tokens: int | None = None
    extra_body: dict | None = None


class TierConfig(BaseModel):
    """Pure tier metadata — no endpoint, no backend, no per-model launch
    config. All "which model, where, with what key" lives in `.env`
    (indexed `TIER{N}_{i}_*` slots, surfaced as `tier.models`). The
    launch recipe for a local model — image, cpuset, vLLM args, sampler
    knobs — lives in `config/local_models.yaml`, keyed by the model's
    served name.
    """
    name: str
    level: int = Field(ge=1, le=5)
    specializations: list[str]
    timeout_s: int = 60

    # Per-tier generation-cap default for `make answers`. None → use the
    # global --max-tokens (Makefile MAXTOK, default 2048). Per-slot
    # overrides via TIER{N}_{i}_MAX_TOKENS still win.
    max_tokens: int | None = None

    # What the router emits in x-vsr-selected-model (used to map a
    # routing decision back to a tier).
    router_alias: str

    # Callable models, populated from `.env` by apply_tier_env_overrides.
    # Empty list means no model is configured for this tier (the router
    # may still emit `router_alias`, but `make answers` has nothing to
    # call). Each entry corresponds to one `TIER{N}_{i}_*` slot.
    models: list[TierModel] = Field(default_factory=list)

    @property
    def model_id(self) -> str:
        """The name the router emits in x-vsr-selected-model.

        The compiled router-config names every model card after the tier
        id (`tier1`…`tier5`), and `make route` defaults to the local OAI
        mock — the routing pass only needs the decision headers, not a
        real completion. So the router's emitted name == `router_alias`.

        `make answers` is what calls real models (TIER{N}_{i}_MODEL),
        and it bypasses the router entirely, so per-vendor identifiers
        never need to flow through the router-config or this property.
        """
        return self.router_alias

    def resolved_models(self) -> list[TierModel]:
        """The tier's callable models. Just `self.models` — there is no
        YAML fallback synthesis. If env doesn't provide slots, returns []."""
        return self.models

    @field_validator("specializations")
    @classmethod
    def _check_specs(cls, v: list[str]) -> list[str]:
        unknown = set(v) - SPECIALIZATIONS
        if unknown:
            raise ValueError(f"unknown specializations: {sorted(unknown)}")
        return v


# ─────────────────────────────────────────────────────────────────────────
# Local model recipe library — `config/local_models.yaml`
# ─────────────────────────────────────────────────────────────────────────
# `make start_LLM` is tier-agnostic. It walks every env slot whose URL
# resolves to localhost, looks up the slot's `served_model_name` in this
# library, and executes the matching per-CPU-vendor launch command.
# Recipes also carry an optional `extra_body` that the harness merges
# into chat requests against this model (Qwen3 thinking flag, sampler).

class LocalLauncherSpec(BaseModel):
    """The verbatim argv to start (and stop) the engine for one CPU vendor.

    Three placeholders are filled in by the dispatcher: `{port}` (from
    the env slot's URL), `{served_name}` (from the slot's MODEL), and
    `{container_name}` (rendered from the recipe's `container_name`
    template). Everything else lives literally in the argv — there's no
    structured "fields" to extract; the recipe is the contract.
    """
    model_config = {"extra": "forbid"}
    start: list[str]
    stop: list[str]


class LocalModelRecipe(BaseModel):
    """One model's full launch + request-time configuration."""
    model_config = {"extra": "forbid"}

    description: str = ""
    # Container/process name template. Placeholders: {served_name}, {port}.
    container_name: str = "vllm-{served_name}"
    # Optional knobs merged into chat requests against this model
    # (Qwen3 chat_template_kwargs, sampler params, etc.). Per-slot env
    # overrides (TIER{N}_{i}_THINKING) deep-merge on top.
    extra_body: dict | None = None
    # Per-CPU-vendor launch specs. At least one must be present; the
    # dispatcher errors with a clear message if the host CPU vendor has
    # no matching block.
    amd: LocalLauncherSpec | None = None
    intel: LocalLauncherSpec | None = None

    def for_vendor(self, vendor: str) -> LocalLauncherSpec:
        spec = getattr(self, vendor, None)
        if spec is None:
            present = [v for v in ("amd", "intel") if getattr(self, v)]
            raise ValueError(
                f"recipe has no {vendor!r} launcher block "
                f"(available: {present or '(none)'})"
            )
        return spec


class LocalModelLibrary(BaseModel):
    """The whole `config/local_models.yaml` file — model id → recipe."""
    recipes: dict[str, LocalModelRecipe]

    @classmethod
    def load(cls, path: Path) -> LocalModelLibrary:
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"{path}: top level must be a mapping of model_id → recipe"
            )
        return cls(recipes={
            k: LocalModelRecipe.model_validate(v) for k, v in raw.items()
        })


# Merge `over` into `base` recursively. Used to layer per-slot env
# overrides (e.g. THINKING) on top of recipe.extra_body — slot wins on
# leaf conflicts; nested dicts merge.
def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def localhost_port(url: str) -> int | None:
    """Return the port if `url` points at localhost / 127.0.0.1, else None.

    A URL like `http://localhost:8001/v1` → 8001. Any non-loopback host
    (vendor APIs, internal IPs) returns None, signalling "no local
    container to launch / no recipe lookup needed".
    """
    from urllib.parse import urlparse
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in ("localhost", "127.0.0.1"):
        return None
    if parsed.port is not None:
        return parsed.port
    return 80 if parsed.scheme == "http" else 443


class ModelsConfig(BaseModel):
    tiers: list[TierConfig]

    def by_name(self, name: str) -> TierConfig:
        for t in self.tiers:
            if t.name == name:
                return t
        raise KeyError(f"no tier named {name!r}")

    def by_level(self, level: int) -> TierConfig:
        for t in self.tiers:
            if t.level == level:
                return t
        raise KeyError(f"no tier with level {level}")


class RouterProcessConfig(BaseModel):
    """How to launch and reach the vLLM Semantic Router.

    The router is a Go binary (`vllm-sr`) that exposes an apiserver (default 8080)
    and an Envoy frontend (default 8801). We don't pass a config *into* the router
    here — the router manages its own config — but we do tell the harness how to
    invoke the binary and where to find its endpoints.
    """

    binary: str = "vllm-sr"
    serve_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    apiserver_host: str = "127.0.0.1"
    apiserver_port: int = 8080
    frontend_host: str = "127.0.0.1"
    # 8899 matches the default Envoy listener that `vllm-sr serve` generates
    # in its auto-created config.yaml. Update if your router config differs.
    frontend_port: int = 8899
    ready_timeout_s: int = 120
    stop_timeout_s: int = 15
    log_path: str | None = None  # if set, captures launcher stdout+stderr there
    auto_model_name: str = "auto"  # what to send as `model` to invoke routing
    # If True, expect the router stack to already be running; don't spawn or
    # stop it. Useful for shared dev stacks or CI.
    external: bool = False
    # If True, run `vllm-sr stop` to tear down the Docker stack when the
    # harness exits. Default False: leave the stack running so repeat
    # benchmark runs don't pay the multi-second cold-start cost.
    stop_on_exit: bool = False


class Attachment(BaseModel):
    type: Literal["image", "audio"]
    path: str


class ExpectedAnswer(BaseModel):
    """One gold/reference answer for a query.

    A query can carry several (e.g. one per provider). Each becomes a
    row in the `gold_answers` table and a `demo.json` `expected_answers[]`
    entry.

      answer    the reference text (required)
      model     per-query unique key (→ gold_answers.model_id and
                demo.json `model`) — required
      provider  optional label (Anthropic / OpenAI / Google) → demo.json

    Extra fields are rejected — keeps queries.json strictly conformant
    so a downstream loader can rely on the shape.
    """
    model_config = {"extra": "forbid"}

    answer: str
    model: str
    provider: str | None = None

    @property
    def model_id(self) -> str:
        return self.model


class QuerySpec(BaseModel):
    """One curated benchmark query.

    Extra fields are rejected — a stray legacy `expected_answer` (or a
    typo) raises at load time rather than being silently ignored.
    """
    model_config = {"extra": "forbid"}

    id: str
    prompt: str
    # One or more gold/reference answers. A single-gold query is just a
    # one-entry list — there is no separate scalar form. `model` is the
    # per-query unique key (becomes gold_answers.model_id and demo.json
    # `model`).
    expected_answers: list[ExpectedAnswer] = Field(default_factory=list)
    expected_min_tier: int = Field(ge=1, le=5)
    specializations: list[str]
    domain_tags: list[str] = Field(default_factory=list)
    attachments: list[Attachment] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("specializations")
    @classmethod
    def _check_specs(cls, v: list[str]) -> list[str]:
        # specializations are downstream-only (sort / review / the
        # post-hoc matches_specialization metric) — they do NOT drive
        # routing. So we don't enforce a whitelist here; whatever labels
        # the source files use are stored verbatim. The tier YAMLs DO
        # keep the whitelist (small, author-edited; cheap typo catch).
        if not v:
            raise ValueError("at least one specialization required")
        return v

    @model_validator(mode="after")
    def _check_golds_unique(self) -> QuerySpec:
        seen: set[str] = set()
        for g in self.expected_answers:
            if g.model_id in seen:
                raise ValueError(
                    f"query {self.id}: duplicate gold model id "
                    f"{g.model_id!r} — each expected answer needs a unique "
                    f"`model` within the query."
                )
            seen.add(g.model_id)
        return self


class QuerySet(BaseModel):
    queries: list[QuerySpec]


def _read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _env_bool(val: str) -> bool | None:
    """Parse a human-written boolean. Returns None if unrecognized."""
    v = val.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return None


def _with_thinking(extra: dict | None, enabled: bool) -> dict:
    """Return a copy of `extra` with chat_template_kwargs.enable_thinking set.

    Qwen3's chat template gates the hidden <think> chain on this flag.
    Sibling keys (sampler params, other chat_template_kwargs) are
    preserved. Pure: never mutates the input.
    """
    out = dict(extra) if isinstance(extra, dict) else {}
    ctk = dict(out.get("chat_template_kwargs") or {})
    ctk["enable_thinking"] = enabled
    out["chat_template_kwargs"] = ctk
    return out


def _int_env(name: str, raw: str, unit: str = "") -> int:
    """Parse an int env var, raising a clear error naming the var."""
    try:
        return int(raw)
    except ValueError as e:
        suffix = f" {unit}" if unit else ""
        raise ValueError(f"{name} must be an integer{suffix}, got {raw!r}") from e


def _bool_env(name: str, raw: str) -> bool:
    flag = _env_bool(raw)
    if flag is None:
        raise ValueError(
            f"{name} must be a boolean (true/false/1/0/yes/no), got {raw!r}"
        )
    return flag


SLOT_SUFFIXES = (
    "URL", "MODEL", "API_KEY", "PROVIDER", "TIMEOUT", "MAX_TOKENS", "THINKING",
)


def _slot_var(n: int, i: int, suffix: str) -> str:
    """Env var name for tier `n` slot `i` (i ≥ 1)."""
    return f"TIER{n}_{i}_{suffix}"


def _build_tier_model(tier: TierConfig, i: int) -> TierModel | None:
    """Assemble one TierModel for slot `i` from `TIER{n}_{i}_*` env vars.

    The slot exists iff URL and MODEL are both set. Both are required —
    there's no YAML fallback for either; the tier YAML is metadata only.
    Returns None when slot `i` is absent (signals end of discovery).
    Per-slot request-body extras (THINKING) attach here; any
    model-defined defaults (from `config/local_models.yaml`'s
    `extra_body`) merge in later via `apply_recipe_extras`.
    """
    n = tier.level

    def g(suffix: str) -> str:
        return os.environ.get(_slot_var(n, i, suffix), "").strip()

    url = g("URL")
    model = g("MODEL")
    if not url and not model:
        return None  # gap → discovery stops here
    if not url:
        raise ValueError(
            f"{_slot_var(n, i, 'MODEL')} is set but "
            f"{_slot_var(n, i, 'URL')} is not — a model slot needs both "
            f"URL and MODEL (the tier YAML carries no endpoint fallback)."
        )
    if not model:
        raise ValueError(
            f"{_slot_var(n, i, 'URL')} is set but "
            f"{_slot_var(n, i, 'MODEL')} is not — a model slot needs both "
            f"URL and MODEL (the URL alone can't identify a model)."
        )

    key_env = _slot_var(n, i, "API_KEY") if g("API_KEY") else None
    provider = g("PROVIDER") or None

    timeout = (
        _int_env(_slot_var(n, i, "TIMEOUT"), g("TIMEOUT"), "seconds")
        if g("TIMEOUT") else tier.timeout_s
    )
    max_tokens = (
        _int_env(_slot_var(n, i, "MAX_TOKENS"), g("MAX_TOKENS"))
        if g("MAX_TOKENS") else tier.max_tokens
    )

    extra: dict | None = None
    if g("THINKING"):
        extra = _with_thinking(
            None, _bool_env(_slot_var(n, i, "THINKING"), g("THINKING"))
        )

    return TierModel(
        slot=i,
        url=url,
        served_model_name=model,
        api_key_env=key_env,
        provider=provider,
        timeout_s=timeout,
        max_tokens=max_tokens,
        extra_body=extra,
    )


def apply_recipe_extras(tier: TierConfig, library: LocalModelLibrary) -> None:
    """Merge each local model's recipe.extra_body into the slot's
    extra_body. Per-slot env overrides (e.g. THINKING) win on leaf
    conflicts via deep-merge. No-op for slots that don't hit localhost
    or have no matching recipe.
    """
    for m in tier.models:
        if localhost_port(m.url) is None:
            continue
        recipe = library.recipes.get(m.served_model_name)
        if recipe is None or not recipe.extra_body:
            continue
        # Recipe is the base (model-default knobs); slot is the override.
        slot_extras = m.extra_body or {}
        m.extra_body = _deep_merge(recipe.extra_body, slot_extras)


def apply_tier_env_overrides(tier: TierConfig) -> TierConfig:
    """Discover callable model slots for `tier` from `TIER{N}_{i}_*` env vars.

    All slots are indexed: `TIER{N}_1_URL/MODEL/API_KEY/PROVIDER/TIMEOUT/
    MAX_TOKENS/THINKING`, `TIER{N}_2_*`, … `i` starts at 1 — there is no
    bare/slot-0 form. Slots must be contiguous; discovery stops at the
    first missing slot.

    Per-slot semantics:
      • MODEL — required.
      • URL — falls back to the tier YAML's `endpoint.url` (same endpoint,
        different model name).
      • API_KEY — env var NAME is recorded on the slot; the actual key
        lives in os.environ.
      • PROVIDER — optional label, surfaces verbatim in demo.json.
      • TIMEOUT / MAX_TOKENS — per-slot overrides; else inherit the tier
        YAML's `timeout_s` / `max_tokens`.
      • THINKING (true/false) — flips Qwen3's chat_template_kwargs
        .enable_thinking inside the slot's extra_body (vendor reasoning
        controls go in the tier YAML's backend.extra_body instead).

    `served_model_name` must be unique within a tier — it's the per-tier
    key in the DB and in demo.json.

    If a stale bare `TIER{N}_*` env var is set (the old single-model
    form, no longer supported), we raise with a migration hint so it
    doesn't silently no-op.

    Mutates and returns the same TierConfig.
    """
    n = tier.level

    # Reject any bare TIER{N}_<suffix> — slots are indexed only.
    for suffix in SLOT_SUFFIXES:
        if os.environ.get(f"TIER{n}_{suffix}", "").strip():
            raise ValueError(
                f"TIER{n}_{suffix} is not a supported env var. Slots are "
                f"indexed from 1 — use TIER{n}_1_{suffix} (slot 1), "
                f"TIER{n}_2_{suffix} (slot 2), and so on. There is no "
                f"bare/slot-0 form."
            )

    models: list[TierModel] = []
    i = 1
    while True:
        m = _build_tier_model(tier, i)
        if m is None:
            break
        models.append(m)
        i += 1

    seen: dict[str, int] = {}
    for m in models:
        if m.served_model_name in seen:
            raise ValueError(
                f"tier {n}: duplicate model name {m.served_model_name!r} "
                f"in slots {seen[m.served_model_name]} and {m.slot} — model "
                f"names must be unique within a tier (used as a DB/JSON key)."
            )
        seen[m.served_model_name] = m.slot
    tier.models = models

    return tier


DEFAULT_LOCAL_MODELS_PATH = Path("config/local_models.yaml")


def load_tiers(
    tiers_dir: Path,
    local_models_path: Path | None = DEFAULT_LOCAL_MODELS_PATH,
) -> ModelsConfig:
    """Load every `*.yaml` in `tiers_dir` as one TierConfig; sort by level.

    Tier YAMLs are metadata-only (name, level, specializations,
    router_alias, timeout_s, max_tokens). All callable model slots
    come from `.env` (indexed `TIER{N}_{i}_*`); the loader applies env
    overrides per tier.

    If `local_models_path` exists, recipes for each model name surface
    as merged extra_body knobs (e.g. Qwen3 thinking + sampler) on slots
    that hit localhost. Per-slot env overrides win on leaf conflicts.

    Files named starting with `_` are skipped (reserved for partials).
    """
    tiers: list[TierConfig] = []
    paths = sorted(p for p in tiers_dir.glob("*.yaml") if not p.name.startswith("_"))
    if not paths:
        raise FileNotFoundError(f"no tier yaml files in {tiers_dir}")
    seen_levels: set[int] = set()
    for path in paths:
        try:
            tier = TierConfig.model_validate(_read_yaml(path))
        except Exception as e:
            raise ValueError(f"failed to load {path}: {e}") from e
        if tier.level in seen_levels:
            raise ValueError(f"duplicate tier level {tier.level} in {path}")
        seen_levels.add(tier.level)
        apply_tier_env_overrides(tier)
        tiers.append(tier)
    tiers.sort(key=lambda t: t.level)

    if local_models_path and local_models_path.exists():
        library = LocalModelLibrary.load(local_models_path)
        for tier in tiers:
            apply_recipe_extras(tier, library)

    return ModelsConfig(tiers=tiers)


def load_models(path: Path) -> ModelsConfig:
    """Backward-compatible loader.

    If `path` is a directory, scan it for per-tier YAMLs. If it's a file,
    error out — the old single-file `models.yaml` is no longer supported.
    The default path in the CLI is `config/tiers/`; callers passing the
    old `config/models.yaml` will get a helpful error.
    """
    if path.is_dir():
        return load_tiers(path)
    raise ValueError(
        f"{path} is not a directory; per-tier YAMLs live under config/tiers/. "
        f"The old single-file models.yaml format was removed; see "
        f"config/tiers/README.md for the new layout."
    )


def load_router_process(path: Path) -> RouterProcessConfig:
    raw = _read_yaml(path) or {}
    # Tolerate the M0 placeholder file ({placeholder: true}) by treating it as defaults.
    if isinstance(raw, dict) and raw.get("placeholder"):
        raw = {}
    return RouterProcessConfig.model_validate(raw)


def load_queries(path: Path) -> QuerySet:
    """Load queries from JSON. Accepts either a top-level list or {"queries": [...]}."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        raw = {"queries": raw}
    return QuerySet.model_validate(raw)


def hash_file(path: Path) -> str:
    """SHA-256 of the file contents (or directory's sorted file contents).

    Used to stamp run_id with config provenance. When `path` is a directory,
    we hash a deterministic concatenation of `<relpath>\\0<bytes>\\0` for
    every regular file under it, sorted by relpath. Anchors run_id provenance
    to the entire tier-config dir, not a single file.
    """
    h = hashlib.sha256()
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file():
                h.update(str(p.relative_to(path)).encode("utf-8"))
                h.update(b"\x00")
                h.update(p.read_bytes())
                h.update(b"\x00")
    else:
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def hash_prompt(prompt: str, attachments: list[Attachment]) -> str:
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    for a in attachments:
        h.update(b"\x00")
        h.update(a.type.encode("utf-8"))
        h.update(b"\x00")
        h.update(a.path.encode("utf-8"))
    return h.hexdigest()[:16]
