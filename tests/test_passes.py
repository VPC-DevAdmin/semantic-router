"""Pass 1 + Pass 2 tests with a fake RouterClient.

The real RouterClient is exercised in test_router_client.py; here we focus on
the per-pass logic: tier comparison, spec match, status transitions, resume
semantics, error isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import select

from benchmark.db import Pass1Result, Pass2Result, init_db, session_scope
from benchmark.pass1 import run_pass1
from benchmark.pass2 import run_pass2
from benchmark.router_client import RouterResult, RoutingDecision
from benchmark.runs import create_run, seed_pending
from benchmark.seed import seed_from_yaml

QUERIES = """
- id: q1
  prompt: "easy"
  expected_min_tier: 1
  specializations: [general]
- id: q2
  prompt: "hard math"
  expected_min_tier: 4
  specializations: [math, reasoning]
- id: q3
  prompt: "code"
  expected_min_tier: 2
  specializations: [code]
"""


def _bootstrap(tmp_path: Path) -> tuple[Path, int]:
    db_path = tmp_path / "t.db"
    qyaml = tmp_path / "queries.yaml"
    qyaml.write_text(QUERIES)
    init_db(db_path)
    seed_from_yaml(qyaml, db_path)

    r_yaml = tmp_path / "router.yaml"
    r_yaml.write_text("placeholder: true\n")
    m_yaml = tmp_path / "models.yaml"
    m_yaml.write_text("tiers: []\n")

    rid = create_run(db_path, router_config_path=r_yaml, models_config_path=m_yaml)
    seed_pending(db_path, rid)
    return db_path, rid


@dataclass
class _FakeClient:
    """Mimics RouterClient.chat for tests.

    `responses` maps prompt-substring → (selected_model, tier, specs, content).
    Anything matching `error_substr` raises.
    """

    responses: dict[str, tuple[str, int | None, list[str] | None, str]]
    error_substr: str | None = None

    async def chat(self, prompt, *, attachments=None, max_tokens=None, **_):
        if self.error_substr and self.error_substr in prompt:
            raise RuntimeError("router unhappy")
        for needle, (model, tier, specs, content) in self.responses.items():
            if needle in prompt:
                d = RoutingDecision(
                    selected_model=model,
                    selected_tier=tier,
                    selected_specs=specs,
                    category=None,
                    reasoning=None,
                    cache_hit=False,
                )
                return RouterResult(
                    content=content,
                    decision=d,
                    prompt_tokens=10,
                    completion_tokens=20,
                    latency_ms=42,
                    raw_body={"model": model},
                    raw_headers={},
                )
        raise AssertionError(f"no fake response for: {prompt!r}")


# ---- Pass 1 ----

@pytest.mark.asyncio
async def test_pass1_meets_and_misses_min_tier(tmp_path: Path) -> None:
    db_path, rid = _bootstrap(tmp_path)
    client = _FakeClient(
        responses={
            "easy":      ("tier1-tiny",  1, ["general"], "x"),
            "hard math": ("tier3-mid",   3, ["general", "code", "math"], "x"),  # below 4
            "code":      ("tier4-large", 4, ["general", "code", "math", "reasoning"], "x"),
        }
    )
    report = await run_pass1(db_path, rid, router_client=client)

    assert report.attempted == 3
    assert report.succeeded == 3
    assert report.errors == 0
    # q1 expects tier>=1 (got 1), q2 expects tier>=4 (got 3), q3 expects tier>=2 (got 4).
    assert report.meets_min_tier == 2

    with session_scope(db_path) as s:
        rows = {r.query_id: r for r in s.execute(select(Pass1Result)).scalars()}
        assert rows["q1"].meets_minimum_tier == 1
        assert rows["q2"].meets_minimum_tier == 0
        assert rows["q3"].meets_minimum_tier == 1
        assert rows["q1"].status == "success"


@pytest.mark.asyncio
async def test_pass1_specialization_match(tmp_path: Path) -> None:
    db_path, rid = _bootstrap(tmp_path)
    client = _FakeClient(
        responses={
            "easy":      ("m", 1, ["general"], "x"),
            "hard math": ("m", 4, ["general", "code"], "x"),  # missing math+reasoning
            "code":      ("m", 2, ["code"], "x"),
        }
    )
    await run_pass1(db_path, rid, router_client=client)

    with session_scope(db_path) as s:
        rows = {r.query_id: r for r in s.execute(select(Pass1Result)).scalars()}
        assert rows["q1"].matches_specialization == 1
        assert rows["q2"].matches_specialization == 0
        assert rows["q3"].matches_specialization == 1


@pytest.mark.asyncio
async def test_pass1_unknown_tier(tmp_path: Path) -> None:
    db_path, rid = _bootstrap(tmp_path)
    client = _FakeClient(
        responses={
            "easy":      ("exotic-1", None, None, "x"),
            "hard math": ("exotic-2", None, None, "x"),
            "code":      ("exotic-3", None, None, "x"),
        }
    )
    report = await run_pass1(db_path, rid, router_client=client)
    assert report.unknown_tier == 3
    assert report.meets_min_tier == 0  # all NULL → not counted

    with session_scope(db_path) as s:
        for r in s.execute(select(Pass1Result)).scalars():
            assert r.meets_minimum_tier is None
            assert r.matches_specialization is None
            assert r.status == "success"


@pytest.mark.asyncio
async def test_pass1_error_isolation(tmp_path: Path) -> None:
    db_path, rid = _bootstrap(tmp_path)
    client = _FakeClient(
        responses={
            "easy": ("m", 1, ["general"], "x"),
            "code": ("m", 2, ["code"], "x"),
            "hard math": ("m", 4, ["math", "reasoning"], "x"),
        },
        error_substr="hard math",
    )
    report = await run_pass1(db_path, rid, router_client=client)
    assert report.succeeded == 2
    assert report.errors == 1
    assert report.error_ids[0][0] == "q2"

    with session_scope(db_path) as s:
        rows = {r.query_id: r for r in s.execute(select(Pass1Result)).scalars()}
        assert rows["q1"].status == "success"
        assert rows["q2"].status == "error"
        assert "router unhappy" in (rows["q2"].error_msg or "")
        assert rows["q3"].status == "success"


@pytest.mark.asyncio
async def test_pass1_resume_skips_success_rows(tmp_path: Path) -> None:
    db_path, rid = _bootstrap(tmp_path)
    # First run errors on q2.
    client = _FakeClient(
        responses={
            "easy": ("m", 1, ["general"], "x"),
            "hard math": ("m", 4, ["math", "reasoning"], "x"),
            "code": ("m", 2, ["code"], "x"),
        },
        error_substr="hard math",
    )
    await run_pass1(db_path, rid, router_client=client)

    # Resume with a happy client — should only re-run q2.
    happy = _FakeClient(
        responses={
            "hard math": ("m", 4, ["math", "reasoning"], "x"),
        }
    )
    report = await run_pass1(db_path, rid, router_client=happy)
    assert report.attempted == 1
    assert report.succeeded == 1

    with session_scope(db_path) as s:
        rows = {r.query_id: r for r in s.execute(select(Pass1Result)).scalars()}
        assert all(r.status == "success" for r in rows.values())


# ---- Pass 2 ----

@pytest.mark.asyncio
async def test_pass2_persists_response(tmp_path: Path) -> None:
    db_path, rid = _bootstrap(tmp_path)
    client = _FakeClient(
        responses={
            "easy": ("model-x", 1, ["general"], "the answer to easy"),
            "hard math": ("model-y", 4, ["math"], "proof goes here"),
            "code": ("model-z", 2, ["code"], "code answer"),
        }
    )
    report = await run_pass2(db_path, rid, router_client=client, max_tokens=512)
    assert report.succeeded == 3
    assert report.errors == 0

    with session_scope(db_path) as s:
        rows = {r.query_id: r for r in s.execute(select(Pass2Result)).scalars()}
        assert rows["q1"].response_text == "the answer to easy"
        assert rows["q1"].router_selected_model == "model-x"
        assert rows["q1"].prompt_tokens == 10
        assert rows["q1"].completion_tokens == 20
        assert rows["q1"].latency_ms == 42
        assert rows["q1"].status == "success"


@pytest.mark.asyncio
async def test_pass2_resume_skips_success(tmp_path: Path) -> None:
    db_path, rid = _bootstrap(tmp_path)
    bad = _FakeClient(
        responses={
            "easy": ("m", 1, ["general"], "ok"),
            "code": ("m", 2, ["code"], "ok"),
            "hard math": ("m", 4, ["math"], "ok"),
        },
        error_substr="hard math",
    )
    await run_pass2(db_path, rid, router_client=bad)

    happy = _FakeClient(
        responses={"hard math": ("m", 4, ["math"], "fixed")}
    )
    report = await run_pass2(db_path, rid, router_client=happy)
    assert report.attempted == 1
    assert report.succeeded == 1

    with session_scope(db_path) as s:
        q2 = s.execute(select(Pass2Result).where(Pass2Result.query_id == "q2")).scalar_one()
        assert q2.response_text == "fixed"
        assert q2.status == "success"
