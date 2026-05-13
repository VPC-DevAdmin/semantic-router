"""Test helpers shared across the suite."""
from __future__ import annotations

import json
from pathlib import Path

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


def make_models_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "models.yaml"
    p.write_text("tiers: []\n")
    return p
