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


class TierEndpoint(BaseModel):
    """How `make answers` reaches a tier (direct OAI call, bypassing router)."""
    url: str
    api_key_env: str | None = None


class BackendSpec(BaseModel):
    """How `make start_LLM` provisions this tier. `kind` is the dispatcher key."""
    # docker_vllm_dual_socket | remote | placeholder
    kind: str
    # Everything else is kind-specific. Pydantic in "permissive" mode here:
    # we keep extra fields rather than reject, so individual backend kinds
    # can carry their own params without each needing a new model class.
    model_config = {"extra": "allow"}


class TierModel(BaseModel):
    """One callable model endpoint within a tier.

    A tier can front several models — e.g. Tier 5 served by Anthropic
    Opus, OpenAI GPT-5, and Google Gemini Pro — so `make answers` can
    show "how the answer changes if you're on OpenAI/Google instead of
    Anthropic". Slot 0 is the bare `TIER{N}_*` / YAML default; slots 1..
    come from `TIER{N}_{i}_*` env vars. `provider` is an optional human
    label (Anthropic / OpenAI / Google) surfaced verbatim in demo.json.

    `served_model_name` must be unique within a tier — it is the per-tier
    model key used as a DB primary-key component and a demo.json key.
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
    name: str
    level: int = Field(ge=1, le=5)
    specializations: list[str]
    timeout_s: int = 60

    # Per-tier generation cap for `make answers`. None → fall back to the
    # global --max-tokens (Makefile MAXTOK, default 2048). Override per
    # tier via TIER{N}_MAX_TOKENS so a slow local tier can be given a
    # bigger budget to actually finish a good answer.
    max_tokens: int | None = None

    # Identity:
    #   router_alias    = what the router emits in x-vsr-selected-model.
    #   served_model_name = what the upstream serves; sent in the body's
    #                       `model` field on direct OAI calls.
    # For local tiers these are usually the same. For vendor APIs
    # (Anthropic), `served_model_name` is the real vendor model id.
    router_alias: str
    served_model_name: str

    endpoint: TierEndpoint
    backend: BackendSpec

    # Optional human label for the bare/slot-0 model (Anthropic / OpenAI /
    # Google). Settable in YAML or via TIER{N}_PROVIDER; flows to demo.json.
    provider: str | None = None

    # Callable models for this tier. Populated by apply_tier_env_overrides
    # (slot 0 from YAML/bare env + slots 1.. from TIER{N}_{i}_* env). Left
    # empty when a TierConfig is built directly (tests) — use
    # `resolved_models()` which synthesizes slot 0 from the legacy fields.
    models: list[TierModel] = Field(default_factory=list)

    # Convenience: `model_id` returns `router_alias` for the dominant legacy
    # caller (TierLookup, which maps router header → tier). New code should
    # use the explicit `router_alias` or `served_model_name` fields.
    @property
    def model_id(self) -> str:
        return self.router_alias

    def resolved_models(self) -> list[TierModel]:
        """The tier's callable models — always at least one.

        If `models` was populated (by env-override loading) it is
        returned as-is. Otherwise a single slot-1 model is synthesized
        from the tier YAML's `endpoint` / `served_model_name` defaults
        so directly-constructed TierConfigs (tests, programmatic use)
        still resolve to a usable model.
        """
        if self.models:
            return self.models
        return [
            TierModel(
                slot=1,
                url=self.endpoint.url,
                served_model_name=self.served_model_name,
                api_key_env=self.endpoint.api_key_env,
                provider=self.provider,
                timeout_s=self.timeout_s,
                max_tokens=self.max_tokens,
                extra_body=getattr(self.backend, "extra_body", None),
            )
        ]

    @field_validator("specializations")
    @classmethod
    def _check_specs(cls, v: list[str]) -> list[str]:
        unknown = set(v) - SPECIALIZATIONS
        if unknown:
            raise ValueError(f"unknown specializations: {sorted(unknown)}")
        return v


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


def _set_qwen_thinking(tier: TierConfig, enabled: bool) -> None:
    """Flip Qwen3's enable_thinking in the tier's slot-0 backend.extra_body.

    BackendSpec has extra='allow', so reassigning a fresh dict makes
    answers.py's `tier.backend.model_dump()` pick it up. Vendor reasoning
    controls (OpenAI reasoning_effort, Anthropic thinking budgets) are out
    of scope here; set those in the tier YAML's `backend.extra_body`.
    """
    tier.backend.extra_body = _with_thinking(
        getattr(tier.backend, "extra_body", None), enabled
    )


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

    The slot exists iff its MODEL (or URL) is set. URL falls back to the
    tier YAML's `endpoint.url` so you can run the same endpoint with a
    different model name without repeating the URL. MODEL is required.
    Returns None when slot `i` is absent (signals end of discovery).
    """
    n = tier.level

    def g(suffix: str) -> str:
        return os.environ.get(_slot_var(n, i, suffix), "").strip()

    url = g("URL")
    model = g("MODEL")
    if not url and not model:
        return None  # gap → discovery stops here
    if not model:
        raise ValueError(
            f"{_slot_var(n, i, 'URL')} is set but "
            f"{_slot_var(n, i, 'MODEL')} is not — a model slot needs "
            f"a model name (the URL alone can't identify a model)."
        )
    url = url or tier.endpoint.url

    key_env = _slot_var(n, i, "API_KEY") if g("API_KEY") else tier.endpoint.api_key_env
    provider = g("PROVIDER") or None

    timeout = (
        _int_env(_slot_var(n, i, "TIMEOUT"), g("TIMEOUT"), "seconds")
        if g("TIMEOUT") else tier.timeout_s
    )
    max_tokens = (
        _int_env(_slot_var(n, i, "MAX_TOKENS"), g("MAX_TOKENS"))
        if g("MAX_TOKENS") else tier.max_tokens
    )

    yaml_extra = getattr(tier.backend, "extra_body", None)
    extra = dict(yaml_extra) if isinstance(yaml_extra, dict) else None
    if g("THINKING"):
        extra = _with_thinking(extra, _bool_env(_slot_var(n, i, "THINKING"), g("THINKING")))

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


def load_tiers(tiers_dir: Path) -> ModelsConfig:
    """Load every `*.yaml` in `tiers_dir` as one TierConfig; sort by level.

    This is the single source of truth for tier configuration. Files named
    starting with `_` are skipped (reserved for partials / templates).

    After loading each tier, `TIER{N}_URL`/`TIER{N}_MODEL`/`TIER{N}_API_KEY`
    env vars (if set and non-empty) override the corresponding YAML fields
    — so .env is the single place to flip endpoint config across tiers.
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
