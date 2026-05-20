"""Test helpers shared across the suite."""
from __future__ import annotations

import json
from pathlib import Path

from benchmark.config import ModelsConfig, TierConfig, TierModel
from benchmark.db import init_db
from benchmark.load import load_into_db


def write_queries(tmp_path: Path, queries: list[dict]) -> Path:
    """Write a list of query dicts as JSON; return the path."""
    p = tmp_path / "queries.json"
    p.write_text(json.dumps(queries))
    return p


def bootstrap_db(tmp_path: Path, queries: list[dict]) -> Path:
    """Create empty DB, write queries.json, load — return DB path."""
    db = tmp_path / "test.db"
    init_db(db)
    qjson = write_queries(tmp_path, queries)
    load_into_db(qjson, db)
    return db


def make_router_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "router.yaml"
    p.write_text("placeholder: true\n")
    return p


def make_tier(
    *,
    level: int,
    name: str | None = None,
    router_alias: str | None = None,
    served_model_name: str | None = None,
    url: str | None = None,
    api_key_env: str | None = None,
    specializations: list[str] | None = None,
    timeout_s: int = 60,
) -> TierConfig:
    """Build a minimal TierConfig for tests. All identifiers default to
    `tier{level}`. The tier comes pre-populated with a single slot-1
    model so `tier.models[0]` works without going through env override."""
    n = name or f"tier{level}"
    served = served_model_name or n
    return TierConfig(
        name=n,
        level=level,
        specializations=specializations or ["general"],
        timeout_s=timeout_s,
        router_alias=router_alias or n,
        models=[
            TierModel(
                slot=1,
                url=url or f"http://localhost:880{level}/v1",
                served_model_name=served,
                api_key_env=api_key_env,
                timeout_s=timeout_s,
            ),
        ],
    )


def make_models(levels: list[int]) -> ModelsConfig:
    return ModelsConfig(tiers=[make_tier(level=lvl) for lvl in levels])


def make_models_yaml(tmp_path: Path) -> Path:
    """Create a minimal per-tier YAML dir used for run-config provenance hashing."""
    d = tmp_path / "tiers"
    d.mkdir(exist_ok=True)
    (d / "tier1.yaml").write_text(
        "name: tier1\n"
        "level: 1\n"
        "specializations: [general]\n"
        "router_alias: tier1\n"
    )
    return d
