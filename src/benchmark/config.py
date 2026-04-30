"""Pydantic-validated YAML config loaders.

Configs validated here:
  - models.yaml   : tier endpoints (OAI-compatible)
  - gold.yaml     : gold model endpoint
  - judge.yaml    : LLM-as-judge endpoint
  - scoring.yaml  : rubric and score scale
  - queries.yaml  : curated query set

Router config (router.yaml) is passed straight to the router subprocess and not
parsed here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

SPECIALIZATIONS = {"general", "code", "math", "reasoning", "creative", "vision", "tts"}


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


class EndpointConfig(BaseModel):
    """Shape shared by gold.yaml and judge.yaml."""

    endpoint: str
    model_id: str
    api_key_env: str | None = None
    timeout_s: int = 120
    temperature: float = 0.0
    max_tokens: int | None = None


class ScoringConfig(BaseModel):
    rubric_version: str
    scale: dict[int, str]
    dimensions: list[str] = Field(default_factory=list)

    @field_validator("scale")
    @classmethod
    def _check_scale(cls, v: dict[int, str]) -> dict[int, str]:
        if sorted(v.keys()) != list(range(1, len(v) + 1)) or len(v) < 3:
            raise ValueError("scale must be a 1..N dict with N>=3")
        return v


class Attachment(BaseModel):
    type: Literal["image", "audio"]
    path: str


class QuerySpec(BaseModel):
    id: str
    prompt: str
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


def load_endpoint(path: Path) -> EndpointConfig:
    return EndpointConfig.model_validate(_read_yaml(path))


def load_scoring(path: Path) -> ScoringConfig:
    return ScoringConfig.model_validate(_read_yaml(path))


def load_queries(path: Path) -> QuerySet:
    raw = _read_yaml(path)
    # queries.yaml is a flat list at the top level for ergonomic editing
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
