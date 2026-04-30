"""Gold generation tests with a fake OAIClient."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from benchmark.config import EndpointConfig
from benchmark.db import Query, init_db, session_scope
from benchmark.gold import generate_gold
from benchmark.seed import seed_from_yaml
from benchmark.tiers import ChatResult


QUERIES = """
- id: g001
  prompt: "Easy"
  expected_min_tier: 1
  specializations: [general]
- id: g002
  prompt: "Code please"
  expected_min_tier: 2
  specializations: [code]
- id: g003
  prompt: "Read this aloud"
  expected_min_tier: 3
  specializations: [tts]
"""


def _fake_client(content: str = "stub answer", fail_ids: set[str] | None = None):
    fail_ids = fail_ids or set()
    client = AsyncMock()

    async def chat(prompt, **kwargs):
        # Heuristic: identify which query based on prompt.
        if "Easy" in prompt and "g001" in fail_ids:
            raise RuntimeError("boom")
        if "Code" in prompt and "g002" in fail_ids:
            raise RuntimeError("boom")
        return ChatResult(
            content=content,
            model="gold-model",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=5,
            raw={},
        )

    client.chat.side_effect = chat
    return client


def _cfg() -> EndpointConfig:
    return EndpointConfig(
        endpoint="https://example.invalid/v1",
        model_id="gold-model",
        api_key_env=None,
        timeout_s=10,
        temperature=0.0,
        max_tokens=128,
    )


def _seed(tmp_path: Path) -> tuple[Path, Path]:
    db_path = tmp_path / "test.db"
    yaml_path = tmp_path / "queries.yaml"
    yaml_path.write_text(QUERIES)
    init_db(db_path)
    seed_from_yaml(yaml_path, db_path)
    return db_path, yaml_path


@pytest.mark.asyncio
async def test_skips_tts_and_generates_others(tmp_path: Path) -> None:
    db_path, _ = _seed(tmp_path)
    gold_dir = tmp_path / "gold"

    report = await generate_gold(
        db_path=db_path,
        gold_config_path=Path("/dev/null"),
        gold_dir=gold_dir,
        client=_fake_client("the answer"),
        cfg=_cfg(),
    )

    assert report.generated == 2
    assert report.skipped_tts == 1
    assert report.skipped_existing == 0
    assert report.errors == 0

    with session_scope(db_path) as session:
        rows = {q.query_id: q for q in session.execute(select(Query)).scalars()}
        assert rows["g001"].gold_answer == "the answer"
        assert rows["g002"].gold_answer == "the answer"
        assert rows["g003"].gold_answer is None  # tts skipped

    assert (gold_dir / "g001.md").exists()
    assert (gold_dir / "g002.md").exists()
    assert not (gold_dir / "g003.md").exists()


@pytest.mark.asyncio
async def test_resume_skips_existing(tmp_path: Path) -> None:
    db_path, _ = _seed(tmp_path)
    gold_dir = tmp_path / "gold"

    # First pass — only g001 gets gold (manually set), g002 still missing.
    with session_scope(db_path) as session:
        q = session.execute(select(Query).where(Query.query_id == "g001")).scalar_one()
        q.gold_answer = "preexisting"
        q.gold_model = "gold-model"
        q.gold_generated_at = datetime.now(UTC)

    report = await generate_gold(
        db_path=db_path,
        gold_config_path=Path("/dev/null"),
        gold_dir=gold_dir,
        client=_fake_client("fresh"),
        cfg=_cfg(),
    )

    assert report.generated == 1
    assert report.skipped_existing == 1
    assert report.skipped_tts == 1

    with session_scope(db_path) as session:
        rows = {q.query_id: q for q in session.execute(select(Query)).scalars()}
        assert rows["g001"].gold_answer == "preexisting"  # untouched
        assert rows["g002"].gold_answer == "fresh"


@pytest.mark.asyncio
async def test_refresh_overwrites(tmp_path: Path) -> None:
    db_path, _ = _seed(tmp_path)
    gold_dir = tmp_path / "gold"

    with session_scope(db_path) as session:
        for qid in ("g001", "g002"):
            q = session.execute(select(Query).where(Query.query_id == qid)).scalar_one()
            q.gold_answer = "stale"
            q.gold_model = "old-model"
            q.gold_generated_at = datetime.now(UTC)

    report = await generate_gold(
        db_path=db_path,
        gold_config_path=Path("/dev/null"),
        gold_dir=gold_dir,
        refresh=True,
        client=_fake_client("fresh"),
        cfg=_cfg(),
    )

    assert report.generated == 2
    assert report.skipped_existing == 0

    with session_scope(db_path) as session:
        rows = {q.query_id: q for q in session.execute(select(Query)).scalars()}
        assert rows["g001"].gold_answer == "fresh"
        assert rows["g002"].gold_answer == "fresh"
        assert rows["g001"].gold_model == "gold-model"


@pytest.mark.asyncio
async def test_per_row_error_isolation(tmp_path: Path) -> None:
    db_path, _ = _seed(tmp_path)
    gold_dir = tmp_path / "gold"

    report = await generate_gold(
        db_path=db_path,
        gold_config_path=Path("/dev/null"),
        gold_dir=gold_dir,
        client=_fake_client("ok", fail_ids={"g001"}),
        cfg=_cfg(),
    )

    assert report.generated == 1  # g002
    assert report.errors == 1
    assert report.skipped_tts == 1
    assert report.error_ids[0][0] == "g001"

    with session_scope(db_path) as session:
        rows = {q.query_id: q for q in session.execute(select(Query)).scalars()}
        assert rows["g001"].gold_answer is None
        assert rows["g002"].gold_answer == "ok"


@pytest.mark.asyncio
async def test_only_filter(tmp_path: Path) -> None:
    db_path, _ = _seed(tmp_path)
    gold_dir = tmp_path / "gold"

    report = await generate_gold(
        db_path=db_path,
        gold_config_path=Path("/dev/null"),
        gold_dir=gold_dir,
        only=["g001"],
        client=_fake_client("a"),
        cfg=_cfg(),
    )

    assert report.generated == 1
    with session_scope(db_path) as session:
        rows = {q.query_id: q for q in session.execute(select(Query)).scalars()}
        assert rows["g001"].gold_answer == "a"
        assert rows["g002"].gold_answer is None
