"""Pass 1 (routing decision) tests with a fake RouterClient.

The real RouterClient is exercised in test_router_client.py; here we focus
on per-pass logic: tier comparison, spec match, status transitions, resume
semantics, error isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import select

from benchmark.db import Pass1Result, session_scope
from benchmark.pass1 import run_pass1
from benchmark.router_client import RouterResult, RoutingDecision
from benchmark.runs import create_run, seed_pending

from ._helpers import bootstrap_db, make_models_yaml, make_router_yaml

QUERIES = [
    {"id": "q1", "prompt": "easy", "expected_min_tier": 1, "specializations": ["general"]},
    {
        "id": "q2", "prompt": "hard math",
        "expected_min_tier": 4, "specializations": ["math", "reasoning"],
    },
    {"id": "q3", "prompt": "coder", "expected_min_tier": 2, "specializations": ["coding"]},
]


def _bootstrap(tmp_path: Path) -> tuple[Path, int]:
    db = bootstrap_db(tmp_path, QUERIES)
    rid = create_run(
        db,
        router_config_path=make_router_yaml(tmp_path),
        models_config_path=make_models_yaml(tmp_path),
    )
    seed_pending(db, rid)
    return db, rid


@dataclass
class _FakeClient:
    responses: dict[str, tuple[str, int | None, list[str] | None, str]]
    error_substr: str | None = None

    async def chat(self, prompt, *, attachments=None, max_tokens=None, **_):
        if self.error_substr and self.error_substr in prompt:
            raise RuntimeError("router unhappy")
        for needle, (model, tier, specs, content) in self.responses.items():
            if needle in prompt:
                return RouterResult(
                    content=content,
                    decision=RoutingDecision(
                        selected_model=model, selected_tier=tier, selected_specs=specs,
                        category=None, reasoning=None, cache_hit=False,
                    ),
                    prompt_tokens=10, completion_tokens=20, latency_ms=42,
                    raw_body={"model": model}, raw_headers={},
                )
        raise AssertionError(f"no fake response for: {prompt!r}")


@pytest.mark.asyncio
async def test_meets_and_misses_min_tier(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    client = _FakeClient(
        responses={
            "easy":      ("tier1-tiny",  1, ["general"], "x"),
            "hard math": ("tier3-mid",   3, ["general", "coding", "math"], "x"),  # below 4
            "coder":     ("tier4-large", 4, ["general", "coding", "math", "reasoning"], "x"),
        }
    )
    report = await run_pass1(db, rid, router_client=client)
    assert report.attempted == 3
    assert report.succeeded == 3
    assert report.errors == 0
    assert report.meets_min_tier == 2

    with session_scope(db) as s:
        rows = {r.query_id: r for r in s.execute(select(Pass1Result)).scalars()}
        assert rows["q1"].meets_minimum_tier == 1
        assert rows["q2"].meets_minimum_tier == 0
        assert rows["q3"].meets_minimum_tier == 1


@pytest.mark.asyncio
async def test_specialization_match(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    client = _FakeClient(
        responses={
            "easy":      ("m", 1, ["general"], "x"),
            "hard math": ("m", 4, ["general", "coding"], "x"),   # missing math+reasoning
            "coder":     ("m", 2, ["coding"], "x"),
        }
    )
    await run_pass1(db, rid, router_client=client)
    with session_scope(db) as s:
        rows = {r.query_id: r for r in s.execute(select(Pass1Result)).scalars()}
        assert rows["q1"].matches_specialization == 1
        assert rows["q2"].matches_specialization == 0
        assert rows["q3"].matches_specialization == 1


@pytest.mark.asyncio
async def test_unknown_tier(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    client = _FakeClient(
        responses={
            "easy":      ("exotic-1", None, None, "x"),
            "hard math": ("exotic-2", None, None, "x"),
            "coder":     ("exotic-3", None, None, "x"),
        }
    )
    report = await run_pass1(db, rid, router_client=client)
    assert report.unknown_tier == 3
    assert report.meets_min_tier == 0
    with session_scope(db) as s:
        for r in s.execute(select(Pass1Result)).scalars():
            assert r.meets_minimum_tier is None
            assert r.matches_specialization is None
            assert r.status == "success"


@pytest.mark.asyncio
async def test_error_isolation(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    client = _FakeClient(
        responses={
            "easy":      ("m", 1, ["general"], "x"),
            "coder":     ("m", 2, ["coding"], "x"),
            "hard math": ("m", 4, ["math", "reasoning"], "x"),
        },
        error_substr="hard math",
    )
    report = await run_pass1(db, rid, router_client=client)
    assert report.succeeded == 2
    assert report.errors == 1
    assert report.error_ids[0][0] == "q2"


@pytest.mark.asyncio
async def test_resume_skips_success_rows(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    client = _FakeClient(
        responses={
            "easy":      ("m", 1, ["general"], "x"),
            "hard math": ("m", 4, ["math", "reasoning"], "x"),
            "coder":     ("m", 2, ["coding"], "x"),
        },
        error_substr="hard math",
    )
    await run_pass1(db, rid, router_client=client)
    happy = _FakeClient(
        responses={"hard math": ("m", 4, ["math", "reasoning"], "x")}
    )
    report = await run_pass1(db, rid, router_client=happy)
    assert report.attempted == 1
    with session_scope(db) as s:
        rows = {r.query_id: r for r in s.execute(select(Pass1Result)).scalars()}
        assert all(r.status == "success" for r in rows.values())
