"""LLM-as-judge tests with a fake OAIClient.

Covers verdict parsing (clean JSON, fenced JSON, JSON-with-prose, bad JSON),
score persistence, idempotency per (run, query, judge), and skip-no-gold.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from benchmark.config import EndpointConfig, ScoringConfig
from benchmark.db import Pass2Result, Query, Score, init_db, session_scope
from benchmark.judge import _parse_verdict, build_judge_prompt, judge_run
from benchmark.runs import create_run, seed_pending
from benchmark.seed import seed_from_yaml
from benchmark.tiers import ChatResult

QUERIES = """
- id: j1
  prompt: "Q1"
  expected_min_tier: 1
  specializations: [general]
- id: j2
  prompt: "Q2"
  expected_min_tier: 1
  specializations: [general]
"""


def _rubric() -> ScoringConfig:
    return ScoringConfig(
        rubric_version="v1",
        scale={1: "unusable", 2: "weak", 3: "ok", 4: "close", 5: "matches"},
    )


def _cfg() -> EndpointConfig:
    return EndpointConfig(
        endpoint="https://example.invalid/v1",
        model_id="judge-model",
        api_key_env=None,
        timeout_s=10,
        temperature=0.0,
        max_tokens=256,
    )


def _bootstrap(tmp_path: Path) -> tuple[Path, int]:
    db = tmp_path / "t.db"
    qy = tmp_path / "queries.yaml"
    qy.write_text(QUERIES)
    init_db(db)
    seed_from_yaml(qy, db)

    r_yaml = tmp_path / "router.yaml"
    r_yaml.write_text("placeholder: true\n")
    m_yaml = tmp_path / "models.yaml"
    m_yaml.write_text("tiers: []\n")
    rid = create_run(db, router_config_path=r_yaml, models_config_path=m_yaml)
    seed_pending(db, rid)

    # Populate pass2 success + gold for j1 and j2.
    with session_scope(db) as s:
        for qid, resp in [("j1", "candidate-1"), ("j2", "candidate-2")]:
            p2 = s.execute(
                select(Pass2Result).where(Pass2Result.query_id == qid)
            ).scalar_one()
            p2.response_text = resp
            p2.status = "success"
            p2.router_selected_model = "test-model"
            q = s.execute(select(Query).where(Query.query_id == qid)).scalar_one()
            q.gold_answer = f"gold-for-{qid}"
            q.gold_model = "gold-model"
    return db, rid


# ---- Verdict parsing ----

def test_parse_verdict_clean_json() -> None:
    text = '{"score": 4, "rationale": "close to gold"}'
    s, r = _parse_verdict(text, 5)
    assert s == 4
    assert r == "close to gold"


def test_parse_verdict_with_fences() -> None:
    text = "```json\n{\"score\": 3, \"rationale\": \"ok\"}\n```"
    s, r = _parse_verdict(text, 5)
    assert s == 3
    assert r == "ok"


def test_parse_verdict_with_prose() -> None:
    text = "Sure thing.\n\n{\"score\": 5, \"rationale\": \"matches gold\"}\n\nLet me know."
    s, _ = _parse_verdict(text, 5)
    assert s == 5


def test_parse_verdict_out_of_range() -> None:
    with pytest.raises(ValueError, match="outside"):
        _parse_verdict('{"score": 7, "rationale": "x"}', 5)


def test_parse_verdict_no_json() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        _parse_verdict("I cannot evaluate this.", 5)


def test_build_judge_prompt_includes_all() -> None:
    p = build_judge_prompt(_rubric(), "the query", "the gold", "the response")
    assert "the query" in p
    assert "the gold" in p
    assert "the response" in p
    assert "Rubric" in p


# ---- Judge run ----

def _client_returning(content: str | dict[str, str]):
    """If `content` is a dict, route by query-id substring."""
    client = AsyncMock()

    async def chat(prompt, **kwargs):
        if isinstance(content, dict):
            for needle, body in content.items():
                if needle in prompt:
                    return ChatResult(
                        content=body, model="judge-model",
                        prompt_tokens=10, completion_tokens=10, latency_ms=5, raw={},
                    )
            raise AssertionError(f"no fake reply for: {prompt}")
        return ChatResult(
            content=content, model="judge-model",
            prompt_tokens=10, completion_tokens=10, latency_ms=5, raw={},
        )

    client.chat.side_effect = chat
    return client


@pytest.mark.asyncio
async def test_judge_persists_scores(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    client = _client_returning('{"score": 4, "rationale": "close"}')
    report = await judge_run(
        db, rid,
        judge_config_path=Path("/dev/null"),
        scoring_config_path=Path("/dev/null"),
        client=client, cfg=_cfg(), rubric=_rubric(),
    )
    assert report.scored == 2
    assert report.score_histogram == {4: 2}

    with session_scope(db) as s:
        scores = list(s.execute(select(Score).where(Score.run_id == rid)).scalars())
        assert len(scores) == 2
        assert all(sc.scorer == "judge" for sc in scores)
        assert all(sc.reviewer_id == "judge-model" for sc in scores)
        assert all(sc.score == 4 for sc in scores)
        assert all(sc.rubric_version == "v1" for sc in scores)


@pytest.mark.asyncio
async def test_judge_idempotent_per_judge_model(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    client = _client_returning('{"score": 3, "rationale": "ok"}')

    r1 = await judge_run(
        db, rid,
        judge_config_path=Path("/dev/null"), scoring_config_path=Path("/dev/null"),
        client=client, cfg=_cfg(), rubric=_rubric(),
    )
    assert r1.scored == 2

    r2 = await judge_run(
        db, rid,
        judge_config_path=Path("/dev/null"), scoring_config_path=Path("/dev/null"),
        client=client, cfg=_cfg(), rubric=_rubric(),
    )
    assert r2.scored == 0
    assert r2.skipped_already_scored == 2


@pytest.mark.asyncio
async def test_judge_parse_errors_isolated(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    client = _client_returning({
        "Q1": "not json at all",
        "Q2": '{"score": 2, "rationale": "weak"}',
    })
    report = await judge_run(
        db, rid,
        judge_config_path=Path("/dev/null"), scoring_config_path=Path("/dev/null"),
        client=client, cfg=_cfg(), rubric=_rubric(),
    )
    assert report.scored == 1
    assert report.parse_errors == 1
    assert report.error_ids[0][0] == "j1"

    with session_scope(db) as s:
        scored = list(s.execute(select(Score).where(Score.run_id == rid)).scalars())
        assert len(scored) == 1
        assert scored[0].query_id == "j2"


@pytest.mark.asyncio
async def test_judge_skips_no_gold(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    # Strip gold from j1.
    with session_scope(db) as s:
        q = s.execute(select(Query).where(Query.query_id == "j1")).scalar_one()
        q.gold_answer = None

    client = _client_returning('{"score": 3, "rationale": "ok"}')
    report = await judge_run(
        db, rid,
        judge_config_path=Path("/dev/null"), scoring_config_path=Path("/dev/null"),
        client=client, cfg=_cfg(), rubric=_rubric(),
    )
    assert report.scored == 1
    assert report.skipped_no_gold == 1
