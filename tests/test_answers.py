"""Per-tier answer collection tests with a fake OAIClient per tier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from benchmark.answers import (
    _extra_body_by_level,
    _max_tokens_by_level,
    run_answers,
)
from benchmark.config import BackendSpec, ModelsConfig, TierConfig, TierEndpoint
from benchmark.db import Pass1Result, TierAnswer, session_scope
from benchmark.runs import create_run, seed_pending_answers
from benchmark.tiers import ChatResult

from ._helpers import bootstrap_db, make_models, make_models_yaml, make_router_yaml

_models = make_models


QUERIES = [
    {"id": "q1", "prompt": "easy", "expected_min_tier": 1, "specializations": ["general"]},
    {
        "id": "q2", "prompt": "harder", "expected_min_tier": 3,
        "specializations": ["coding"],
    },
    {
        "id": "qtts", "prompt": "say it", "expected_min_tier": 2,
        "specializations": ["tts"],
    },
]


def _bootstrap(tmp_path: Path) -> tuple[Path, int]:
    db = bootstrap_db(tmp_path, QUERIES)
    rid = create_run(
        db,
        router_config_path=make_router_yaml(tmp_path),
        models_config_path=make_models_yaml(tmp_path),
    )
    return db, rid


def _record_pass1(db: Path, rid: int, query_id: str, tier_level: int) -> None:
    """Insert a successful pass1_results row so seed_pending_answers picks it up."""
    from datetime import UTC, datetime

    with session_scope(db) as s:
        s.add(
            Pass1Result(
                run_id=rid,
                query_id=query_id,
                router_selected_model=f"tier{tier_level}",
                router_selected_tier=tier_level,
                router_selected_specs=["general"],
                meets_minimum_tier=1,
                matches_specialization=1,
                latency_ms=10,
                raw_routing_metadata={"category": None, "reasoning": "off"},
                status="success",
                attempted_at=datetime.now(UTC),
            )
        )


@dataclass
class _FakeClient:
    """Echoes a deterministic per-tier response so we can assert routing."""

    tier_level: int
    fail_on_query: str | None = None

    async def chat(self, prompt: str, *, attachments=None, max_tokens=None, **_: Any) -> ChatResult:
        if self.fail_on_query and self.fail_on_query in prompt:
            raise RuntimeError(f"tier{self.tier_level} unhappy with {prompt!r}")
        return ChatResult(
            content=f"[tier{self.tier_level}] {prompt}",
            model=f"tier{self.tier_level}",
            prompt_tokens=5,
            completion_tokens=10,
            latency_ms=10 * self.tier_level,
            raw={},
        )


def _clients(levels: list[int], fail_on_query: str | None = None) -> dict[int, _FakeClient]:
    return {lvl: _FakeClient(tier_level=lvl, fail_on_query=fail_on_query) for lvl in levels}


@dataclass
class _CapturingClient:
    """Records the `extra` and `max_tokens` kwargs passed to chat()."""

    tier_level: int
    seen_extra: list = None  # type: ignore[assignment]
    seen_max_tokens: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.seen_extra = []
        self.seen_max_tokens = []

    async def chat(
        self, prompt: str, *, attachments=None, max_tokens=None, extra=None, **_: Any
    ) -> ChatResult:
        self.seen_extra.append(extra)
        self.seen_max_tokens.append(max_tokens)
        return ChatResult(
            content=f"[tier{self.tier_level}] {prompt}",
            model=f"tier{self.tier_level}",
            prompt_tokens=5, completion_tokens=10,
            latency_ms=10, raw={},
        )


def _tier_with_extra_body(level: int, extra_body: dict | None) -> TierConfig:
    backend_kwargs: dict[str, Any] = {"kind": "remote"}
    if extra_body is not None:
        backend_kwargs["extra_body"] = extra_body
    return TierConfig(
        name=f"tier{level}",
        level=level,
        specializations=["general"],
        router_alias=f"tier{level}",
        served_model_name=f"tier{level}",
        endpoint=TierEndpoint(url=f"http://localhost:880{level}/v1"),
        backend=BackendSpec(**backend_kwargs),
    )


# ---- extra_body plumbing ----

def test_extra_body_by_level_reads_backend_config() -> None:
    models = ModelsConfig(tiers=[
        _tier_with_extra_body(1, {"chat_template_kwargs": {"enable_thinking": False}}),
        _tier_with_extra_body(2, None),  # no extra_body
        _tier_with_extra_body(5, {"foo": "bar"}),
    ])
    got = _extra_body_by_level(models)
    assert got == {
        1: {"chat_template_kwargs": {"enable_thinking": False}},
        5: {"foo": "bar"},
    }
    assert 2 not in got  # tiers without extra_body are absent


@pytest.mark.asyncio
async def test_run_answers_forwards_extra_body_to_client(tmp_path: Path) -> None:
    """A tier's backend.extra_body must reach the chat() call so
    provider knobs like enable_thinking=false actually take effect."""
    db = bootstrap_db(tmp_path, QUERIES)
    rid = create_run(
        db,
        router_config_path=make_router_yaml(tmp_path),
        models_config_path=make_models_yaml(tmp_path),
    )
    _record_pass1(db, rid, "q1", tier_level=1)
    _record_pass1(db, rid, "q2", tier_level=2)

    models = ModelsConfig(tiers=[
        _tier_with_extra_body(1, {"chat_template_kwargs": {"enable_thinking": False}}),
        _tier_with_extra_body(2, None),
        _tier_with_extra_body(3, None),
    ])
    seed_pending_answers(db, rid, models)

    cap = {1: _CapturingClient(1), 2: _CapturingClient(2), 3: _CapturingClient(3)}
    await run_answers(db, rid, models=models, clients_by_level=cap)

    # Tier 1 had extra_body → forwarded; tier 2 had none → extra is None.
    assert cap[1].seen_extra == [{"chat_template_kwargs": {"enable_thinking": False}}]
    assert cap[2].seen_extra == [None]


# ---- per-tier max_tokens plumbing ----

def _tier_with_max_tokens(level: int, max_tokens: int | None) -> TierConfig:
    t = _tier_with_extra_body(level, None)
    t.max_tokens = max_tokens
    return t


def test_max_tokens_by_level_reads_per_tier_cap() -> None:
    models = ModelsConfig(tiers=[
        _tier_with_max_tokens(1, 4096),
        _tier_with_max_tokens(2, None),  # no per-tier cap
        _tier_with_max_tokens(5, 256),
    ])
    got = _max_tokens_by_level(models)
    assert got == {1: 4096, 5: 256}
    assert 2 not in got  # tiers without a cap fall back to the global default


@pytest.mark.asyncio
async def test_run_answers_per_tier_max_tokens_overrides_global(tmp_path: Path) -> None:
    """A tier's max_tokens wins; tiers without one get the global --max-tokens."""
    db = bootstrap_db(tmp_path, QUERIES)
    rid = create_run(
        db,
        router_config_path=make_router_yaml(tmp_path),
        models_config_path=make_models_yaml(tmp_path),
    )
    _record_pass1(db, rid, "q1", tier_level=1)
    _record_pass1(db, rid, "q2", tier_level=2)

    models = ModelsConfig(tiers=[
        _tier_with_max_tokens(1, 4096),   # per-tier override
        _tier_with_max_tokens(2, None),   # falls back to global
        _tier_with_max_tokens(3, None),
    ])
    seed_pending_answers(db, rid, models)

    cap = {1: _CapturingClient(1), 2: _CapturingClient(2), 3: _CapturingClient(3)}
    await run_answers(db, rid, models=models, clients_by_level=cap, max_tokens=512)

    assert cap[1].seen_max_tokens == [4096]   # per-tier cap honored
    assert cap[2].seen_max_tokens == [512]    # global default fallback


# ---- seed_pending_answers ----

def test_seed_one_row_per_routed_query(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", tier_level=1)
    _record_pass1(db, rid, "q2", tier_level=3)
    models = _models([1, 2, 3])
    result = seed_pending_answers(db, rid, models)
    # 2 routed queries → 2 rows. qtts is excluded (TTS-only) and has no pass1 row.
    assert result.seeded == 2
    assert result.replaced == 0
    assert result.kept == 0
    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.run_id == rid)).scalars().all()
        assert {(r.query_id, r.tier_level) for r in rows} == {("q1", 1), ("q2", 3)}
        assert all(r.status == "pending" for r in rows)
        assert {r.tier_name for r in rows} == {"tier1", "tier3"}


def test_seed_skips_unrouted_queries(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", tier_level=1)
    # q2 has no pass1 row → not seeded.
    assert seed_pending_answers(db, rid, _models([1, 2, 3])).seeded == 1


def test_seed_idempotent_when_tier_unchanged(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", tier_level=1)
    models = _models([1, 2])
    r1 = seed_pending_answers(db, rid, models)
    assert r1.seeded == 1
    r2 = seed_pending_answers(db, rid, models)
    # Re-seeded, nothing changed: row was at tier 1, still at tier 1.
    assert r2.seeded == 0
    assert r2.replaced == 0
    assert r2.kept == 1


# ---- top-tier gold-fill ----

GOLD_QUERIES = [
    {"id": "g1", "prompt": "trivial", "expected_min_tier": 1,
     "specializations": ["general"], "expected_answer": "gold for g1"},
    {"id": "g5", "prompt": "frontier", "expected_min_tier": 5,
     "specializations": ["general"], "expected_answer": "GOLD ANSWER for g5"},
    {"id": "g5b", "prompt": "frontier-no-gold", "expected_min_tier": 5,
     "specializations": ["general"]},  # no expected_answer
]


def _bootstrap_gold(tmp_path: Path) -> tuple[Path, int]:
    db = bootstrap_db(tmp_path, GOLD_QUERIES)
    rid = create_run(
        db,
        router_config_path=make_router_yaml(tmp_path),
        models_config_path=make_models_yaml(tmp_path),
    )
    return db, rid


def test_seed_gold_fills_top_tier_routed_query(tmp_path: Path) -> None:
    """A query routed to the top tier (level 5) with a gold answer is
    filled from gold as status='success' — no LLM call needed."""
    db, rid = _bootstrap_gold(tmp_path)
    _record_pass1(db, rid, "g5", tier_level=5)
    models = _models([1, 2, 3, 4, 5])

    result = seed_pending_answers(db, rid, models)
    assert result.gold_filled == 1
    assert result.seeded == 0

    with session_scope(db) as s:
        row = s.execute(
            select(TierAnswer).where(TierAnswer.query_id == "g5")
        ).scalar_one()
    assert row.tier_level == 5
    assert row.status == "success"
    assert row.response_text == "GOLD ANSWER for g5"


def test_seed_non_top_tier_not_gold_filled(tmp_path: Path) -> None:
    """A query routed BELOW the top tier still seeds a pending row even
    if it has a gold answer (gold only short-circuits the TOP tier)."""
    db, rid = _bootstrap_gold(tmp_path)
    _record_pass1(db, rid, "g1", tier_level=1)
    models = _models([1, 2, 3, 4, 5])

    result = seed_pending_answers(db, rid, models)
    assert result.gold_filled == 0
    assert result.seeded == 1

    with session_scope(db) as s:
        row = s.execute(
            select(TierAnswer).where(TierAnswer.query_id == "g1")
        ).scalar_one()
    assert row.status == "pending"


def test_seed_top_tier_without_gold_falls_back_to_pending(tmp_path: Path) -> None:
    """Top-tier-routed query with NO gold answer can't be short-circuited —
    it seeds a normal pending row for the worker to fill."""
    db, rid = _bootstrap_gold(tmp_path)
    _record_pass1(db, rid, "g5b", tier_level=5)
    models = _models([1, 2, 3, 4, 5])

    result = seed_pending_answers(db, rid, models)
    assert result.gold_filled == 0
    assert result.seeded == 1

    with session_scope(db) as s:
        row = s.execute(
            select(TierAnswer).where(TierAnswer.query_id == "g5b")
        ).scalar_one()
    assert row.status == "pending"


def test_seed_upgrades_pending_top_tier_row_to_gold(tmp_path: Path) -> None:
    """If a pending top-tier row already exists (e.g. seeded before the
    gold answer was loaded), a later seed upgrades it to gold-success
    without an LLM call."""
    db, rid = _bootstrap_gold(tmp_path)
    _record_pass1(db, rid, "g5b", tier_level=5)  # no gold yet → pending
    models = _models([1, 2, 3, 4, 5])

    r1 = seed_pending_answers(db, rid, models)
    assert r1.seeded == 1  # g5b has no gold → pending

    # Now give g5b a gold answer (simulating an update-gold run) and
    # re-seed: the pending row should upgrade to gold-success.
    from benchmark.db import Query
    with session_scope(db) as s:
        q = s.execute(
            select(Query).where(Query.query_id == "g5b")
        ).scalar_one()
        q.gold_answer = "freshly generated gold for g5b"

    r2 = seed_pending_answers(db, rid, models)
    assert r2.gold_filled == 1
    assert r2.kept == 0

    with session_scope(db) as s:
        rows = s.execute(
            select(TierAnswer).where(TierAnswer.query_id == "g5b")
        ).scalars().all()
    assert len(rows) == 1
    assert rows[0].status == "success"
    assert rows[0].response_text == "freshly generated gold for g5b"


@pytest.mark.asyncio
async def test_run_answers_skips_gold_filled_rows(tmp_path: Path) -> None:
    """End-to-end: a gold-filled top-tier row is already success, so the
    worker doesn't call the (fake) client for it."""
    db, rid = _bootstrap_gold(tmp_path)
    _record_pass1(db, rid, "g1", tier_level=1)   # pending → worker runs it
    _record_pass1(db, rid, "g5", tier_level=5)   # gold-filled → skipped
    models = _models([1, 2, 3, 4, 5])
    seed_pending_answers(db, rid, models)

    report = await run_answers(
        db, rid, models=models, clients_by_level=_clients([1, 2, 3, 4, 5])
    )
    # Only g1 was pending; g5 was already success from gold.
    assert report.attempted == 1
    assert report.succeeded == 1

    with session_scope(db) as s:
        by_qid = {
            r.query_id: r
            for r in s.execute(
                select(TierAnswer).where(TierAnswer.run_id == rid)
            ).scalars().all()
        }
    assert by_qid["g5"].response_text == "GOLD ANSWER for g5"   # untouched
    assert by_qid["g1"].response_text.startswith("[tier1]")     # worker-generated


def test_seed_skips_tts_only(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "qtts", tier_level=2)
    assert seed_pending_answers(db, rid, _models([1, 2, 3])).seeded == 0


def test_seed_replaces_stale_tier(tmp_path: Path) -> None:
    """After `make route` re-runs and picks a different tier for a query,
    the next `make answers` should detect the mismatch and re-seed."""
    db, rid = _bootstrap(tmp_path)
    models = _models([1, 2, 3])

    # First round: router picked tier 1.
    _record_pass1(db, rid, "q1", tier_level=1)
    r1 = seed_pending_answers(db, rid, models)
    assert r1.seeded == 1

    # Router re-runs and now picks tier 3 for the same query. Update the
    # pass1 row to reflect that.
    with session_scope(db) as s:
        p1 = s.execute(
            select(Pass1Result).where(Pass1Result.query_id == "q1")
        ).scalar_one()
        p1.router_selected_tier = 3

    # Next seed call should DELETE the stale tier_answer (tier_level=1) and
    # insert a fresh pending one at tier_level=3.
    r2 = seed_pending_answers(db, rid, models)
    assert r2.replaced == 1
    assert r2.seeded == 0
    assert r2.kept == 0

    with session_scope(db) as s:
        rows = s.execute(
            select(TierAnswer).where(TierAnswer.query_id == "q1")
        ).scalars().all()
        # Exactly one row, at the new tier.
        assert len(rows) == 1
        assert rows[0].tier_level == 3
        assert rows[0].status == "pending"


# ---- run_answers ----

@pytest.mark.asyncio
async def test_run_answers_persists_routed_tier(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", tier_level=1)
    _record_pass1(db, rid, "q2", tier_level=3)
    models = _models([1, 2, 3])
    seed_pending_answers(db, rid, models)
    report = await run_answers(
        db, rid,
        models=models,
        clients_by_level=_clients([1, 2, 3]),
    )
    assert report.attempted == 2
    assert report.succeeded == 2
    assert report.errors == 0

    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.run_id == rid)).scalars().all()
        assert all(r.status == "success" for r in rows)
        for r in rows:
            expected_prompt = "easy" if r.query_id == "q1" else "harder"
            assert r.response_text == f"[tier{r.tier_level}] {expected_prompt}"


@pytest.mark.asyncio
async def test_run_answers_errors_dont_fail_the_pass(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", tier_level=1)
    _record_pass1(db, rid, "q2", tier_level=2)
    models = _models([1, 2])
    seed_pending_answers(db, rid, models)
    clients = _clients([1, 2], fail_on_query="harder")  # q2 (prompt "harder") fails at tier 2
    report = await run_answers(db, rid, models=models, clients_by_level=clients)
    assert report.succeeded == 1
    assert report.errors == 1
    qids = {qid for qid, _, _ in report.error_rows}
    assert qids == {"q2"}


@pytest.mark.asyncio
async def test_run_answers_resume_only_retries_errors(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", tier_level=1)
    _record_pass1(db, rid, "q2", tier_level=2)
    models = _models([1, 2])
    seed_pending_answers(db, rid, models)

    # First run: q2 errors.
    failing = _clients([1, 2], fail_on_query="harder")
    r1 = await run_answers(db, rid, models=models, clients_by_level=failing)
    assert r1.errors == 1

    # Resume with all-happy clients — should only re-run the 1 errored row.
    r2 = await run_answers(db, rid, models=models, clients_by_level=_clients([1, 2]))
    assert r2.attempted == 1
    assert r2.succeeded == 1
    assert r2.errors == 0


@pytest.mark.asyncio
async def test_run_answers_tier_filter_only_processes_that_tier(tmp_path: Path) -> None:
    """`tier_level=1` should skip rows whose tier isn't 1, leaving them pending."""
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", tier_level=1)
    _record_pass1(db, rid, "q2", tier_level=3)
    models = _models([1, 2, 3])
    seed_pending_answers(db, rid, models)

    report = await run_answers(
        db, rid,
        models=models,
        clients_by_level=_clients([1, 2, 3]),
        tier_level=1,
    )
    # Only q1 (tier 1) ran; q2's tier-3 row is left pending.
    assert report.attempted == 1
    assert report.succeeded == 1
    assert report.errors == 0

    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.run_id == rid)).scalars().all()
        by_qid = {r.query_id: r for r in rows}
    assert by_qid["q1"].status == "success"
    assert by_qid["q2"].status == "pending"


@pytest.mark.asyncio
async def test_reset_answers_tier_filter_only_deletes_that_tier(tmp_path: Path) -> None:
    """`reset_answers(tier_level=1)` deletes only tier-1 rows."""
    from benchmark.runs import reset_answers

    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", tier_level=1)
    _record_pass1(db, rid, "q2", tier_level=3)
    models = _models([1, 2, 3])
    seed_pending_answers(db, rid, models)
    # Run them to success so we have non-pending rows to selectively reset.
    await run_answers(db, rid, models=models, clients_by_level=_clients([1, 2, 3]))

    deleted = reset_answers(db, rid, tier_level=1)
    assert deleted == 1

    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.run_id == rid)).scalars().all()
        remaining = {(r.query_id, r.tier_level) for r in rows}
    # q2's tier-3 row survives; q1's tier-1 row is gone.
    assert remaining == {("q2", 3)}


def test_build_clients_for_mock_uses_mock_url() -> None:
    """`_build_clients_for_mock` constructs one OAIClient per tier, all pointing
    at the mock URL, regardless of what each tier's configured endpoint is."""
    from benchmark.answers import _build_clients_for_mock

    models = _models([1, 2, 3])
    mock = "http://localhost:8811/v1"
    clients = _build_clients_for_mock(models, mock)

    assert set(clients.keys()) == {1, 2, 3}
    for level, client in clients.items():
        # OAIClient.endpoint strips trailing /. Check the start matches.
        assert client.endpoint == mock.rstrip("/"), (
            f"tier {level} client endpoint {client.endpoint!r} != mock {mock!r}"
        )


@pytest.mark.asyncio
async def test_run_answers_unknown_tier_recorded_as_error(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    # Router picked tier 9, which has no entry in our 2-tier model config.
    _record_pass1(db, rid, "q1", tier_level=9)
    seed_pending_answers(db, rid, _models([1, 2]))  # still seeds the row
    report = await run_answers(
        db, rid,
        models=_models([1, 2]),
        clients_by_level={1: _FakeClient(tier_level=1), 2: _FakeClient(tier_level=2)},
    )
    assert report.errors == 1
    assert all("level=9" in msg for _, _, msg in report.error_rows)
