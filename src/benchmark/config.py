"""Config loaders.

Configs validated here:
  - models.yaml    : tier endpoints (OAI-compatible)
  - router.yaml    : process-management config for the router subprocess
  - queries.json   : curated query set with embedded gold answers

Queries are JSON (not YAML) because the source files come from upstream as
JSON and the format is structured data, not human-edited config.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

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


class TierConfig(BaseModel):
    name: str
    level: int = Field(ge=1, le=5)
    endpoint: str
    model_id: str
    api_key_env: str | None = None
    specializations: list[str]
    timeout_s: int = 60

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


class QuerySpec(BaseModel):
    id: str
    prompt: str
    expected_answer: str | None = None  # gold answer (from upstream); optional for unscored sets
    expected_min_tier: int = Field(ge=1, le=5)
    specializations: list[str]
    domain_tags: list[str] = Field(default_factory=list)
    attachments: list[Attachment] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("specializations")
    @classmethod
    def _check_specs(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one specialization required")
        unknown = set(v) - SPECIALIZATIONS
        if unknown:
            raise ValueError(f"unknown specializations: {sorted(unknown)}")
        return v


class QuerySet(BaseModel):
    queries: list[QuerySpec]


def _read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models(path: Path) -> ModelsConfig:
    return ModelsConfig.model_validate(_read_yaml(path))


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
    """SHA-256 of the file contents, used to stamp run_id with config provenance."""
    h = hashlib.sha256()
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
