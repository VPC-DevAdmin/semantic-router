"""make answers: multi-model seeding + iteration."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from benchmark import answers as answers_mod
from benchmark.answers import (
    _clients_by_model,
    _clients_for_mock,
    _extra_body_by_model,
    _max_tokens_by_model,
    run_answers,
    run_smoke,
)
from benchmark.config import (
    BackendSpec,
    ModelsConfig,
    TierConfig,
    TierEndpoint,
    TierModel,
)
from benchmark.db import Pass1Result, TierAnswer, session_scope
from benchmark.runs import create_run, reset_answers, seed_pending_answers
from benchmark.tiers import ChatResult

from ._helpers import bootstrap_db, make_models, make_models_yaml, make_router_yaml

_models = make_models

QUERIES = [
    {"id": "q1", "prompt": "easy", "expected_min_tier": 1, "specializations": ["general"]},
    {"id": "q2", "prompt": "harder", "expected_min_tier": 3, "specializations": ["coding"]},
    {"id": "qtts", "prompt": "say it", "expected_min_tier": 2, "specializations": ["tts"]},
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
    with session_scope(db) as s:
        s.add(Pass1Result(
            run_id=rid, query_id=query_id,
            router_selected_model=f"tier{tier_level}",
            router_selected_tier=tier_level,
            router_selected_specs=["general"],
            meets_minimum_tier=1, matches_specialization=1, latency_ms=10,
            raw_routing_metadata={"category": None, "reasoning": "off"},
            status="success", attempted_at=datetime.now(UTC),
        ))


@dataclass
class _FakeClient:
    """Echoes a deterministic per-(tier,model) response."""

    tier_level: int
    model_id: str = ""
    fail_on_query: str | None = None

    async def chat(self, prompt: str, *, attachments=None, max_tokens=None,
                   extra=None, **_: Any) -> ChatResult:
        if self.fail_on_query and self.fail_on_query in prompt:
            raise RuntimeError(f"tier{self.tier_level} unhappy with {prompt!r}")
        tag = self.model_id or f"tier{self.tier_level}"
        return ChatResult(
            content=f"[{tag}] {prompt}", model=tag,
            prompt_tokens=5, completion_tokens=10,
            latency_ms=10 * self.tier_level, raw={},
        )


@dataclass
class _CapturingClient:
    """Records the `extra` and `max_tokens` kwargs passed to chat()."""

    tier_level: int
    seen_extra: list = None  # type: ignore[assignment]
    seen_max_tokens: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.seen_extra = []
        self.seen_max_tokens = []

    async def chat(self, prompt: str, *, attachments=None, max_tokens=None,
                   extra=None, **_: Any) -> ChatResult:
        self.seen_extra.append(extra)
        self.seen_max_tokens.append(max_tokens)
        return ChatResult(
            content=f"[tier{self.tier_level}] {prompt}", model=f"tier{self.tier_level}",
            prompt_tokens=5, completion_tokens=10, latency_ms=10, raw={},
        )


def _clients(levels: list[int], fail_on_query: str | None = None) -> dict:
    """One _FakeClient per (level, 'tier{level}') — matches make_models keys."""
    return {
        (lvl, f"tier{lvl}"): _FakeClient(
            tier_level=lvl, model_id=f"tier{lvl}", fail_on_query=fail_on_query
        )
        for lvl in levels
    }


def _tier(level: int, models: list[TierModel] | None = None) -> TierConfig:
    t = TierConfig(
        name=f"tier{level}", level=level, specializations=["general"],
        router_alias=f"tier{level}", served_model_name=f"tier{level}",
        endpoint=TierEndpoint(url=f"http://localhost:880{level}/v1"),
        backend=BackendSpec(kind="remote"),
    )
    if models is not None:
        t.models = models
    return t


def _multi(level: int, names: list[str]) -> TierConfig:
    return _tier(level, [
        TierModel(slot=i, url=f"http://h{level}/v1", served_model_name=n,
                  provider=f"P{i}")
        for i, n in enumerate(names)
    ])


# ---- client / knob maps ----

def test_clients_by_model_keyed_by_level_and_model() -> None:
    models = ModelsConfig(tiers=[_multi(3, ["gpt", "gemini"]), _tier(1)])
    clients = _clients_by_model(models)
    assert set(clients) == {(3, "gpt"), (3, "gemini"), (1, "tier1")}


def test_extra_body_and_max_tokens_by_model() -> None:
    t = _tier(2, [
        TierModel(slot=0, url="http://x/v1", served_model_name="a",
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                  max_tokens=4096),
        TierModel(slot=1, url="http://x/v1", served_model_name="b"),
    ])
    models = ModelsConfig(tiers=[t])
    assert _extra_body_by_model(models) == {
        (2, "a"): {"chat_template_kwargs": {"enable_thinking": False}}
    }
    assert _max_tokens_by_model(models) == {(2, "a"): 4096}


def test_clients_for_mock_points_every_model_at_mock() -> None:
    models = ModelsConfig(tiers=[_multi(3, ["gpt", "gemini"]), _tier(1)])
    mock = "http://localhost:8811/v1"
    clients = _clients_for_mock(models, mock)
    assert set(clients) == {(3, "gpt"), (3, "gemini"), (1, "tier1")}
    assert all(c.endpoint == mock.rstrip("/") for c in clients.values())


# ---- seed_pending_answers ----

def test_seed_one_row_per_routed_query_single_model(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", 1)
    _record_pass1(db, rid, "q2", 3)
    result = seed_pending_answers(db, rid, _models([1, 2, 3, 4, 5]))
    assert (result.seeded, result.replaced, result.kept) == (2, 0, 0)
    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.run_id == rid)).scalars().all()
    assert {(r.query_id, r.tier_level, r.model_id) for r in rows} == {
        ("q1", 1, "tier1"), ("q2", 3, "tier3"),
    }
    assert all(r.status == "pending" for r in rows)


def test_seed_one_row_per_model_in_routed_tier(tmp_path: Path) -> None:
    """The routed tier fronts several models → one row per model."""
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q2", 3)
    models = ModelsConfig(tiers=[_tier(1), _multi(3, ["gpt-5", "gemini", "claude"]), _tier(5)])
    result = seed_pending_answers(db, rid, models)
    assert result.seeded == 3
    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.query_id == "q2")).scalars().all()
    assert {r.model_id for r in rows} == {"gpt-5", "gemini", "claude"}
    assert {r.provider for r in rows} == {"P0", "P1", "P2"}
    assert all(r.tier_level == 3 and r.status == "pending" for r in rows)


def test_seed_skips_unrouted_and_tts(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", 1)
    _record_pass1(db, rid, "qtts", 2)  # tts-only → excluded
    assert seed_pending_answers(db, rid, _models([1, 2, 3])).seeded == 1


def test_seed_idempotent_when_unchanged(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", 1)
    models = _models([1, 2])
    assert seed_pending_answers(db, rid, models).seeded == 1
    r2 = seed_pending_answers(db, rid, models)
    assert (r2.seeded, r2.replaced, r2.kept) == (0, 0, 1)


def test_seed_skips_top_tier_routed_queries(tmp_path: Path) -> None:
    """The top tier is the gold reference — make answers must NOT call it.
    A top-tier-routed query seeds NO rows (its answers come from
    update-gold / upstream)."""
    db = bootstrap_db(tmp_path, [
        {"id": "g5", "prompt": "frontier", "expected_min_tier": 5,
         "specializations": ["general"],
         "expected_answers": [{"answer": "GOLD for g5", "model": "Opus"}]},
    ])
    rid = create_run(
        db, router_config_path=make_router_yaml(tmp_path),
        models_config_path=make_models_yaml(tmp_path),
    )
    _record_pass1(db, rid, "g5", 5)
    models = ModelsConfig(tiers=[_tier(1), _multi(5, ["opus", "gpt-5"])])
    result = seed_pending_answers(db, rid, models)
    assert result.seeded == 0
    assert result.skipped_top_tier == 1
    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.query_id == "g5")).scalars().all()
    assert rows == []  # no model calls for the gold tier


def test_seed_deletes_stale_top_tier_rows(tmp_path: Path) -> None:
    """If a query is re-routed to the top tier, any prior lower-tier rows
    are dropped and nothing is seeded."""
    db, rid = _bootstrap(tmp_path)
    models = ModelsConfig(tiers=[_tier(1), _multi(5, ["opus"])])
    _record_pass1(db, rid, "q1", 1)
    assert seed_pending_answers(db, rid, models).seeded == 1
    with session_scope(db) as s:
        s.execute(
            select(Pass1Result).where(Pass1Result.query_id == "q1")
        ).scalar_one().router_selected_tier = 5
    r2 = seed_pending_answers(db, rid, models)
    assert r2.skipped_top_tier == 1 and r2.replaced == 1 and r2.seeded == 0
    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.query_id == "q1")).scalars().all()
    assert rows == []


def test_seed_replaces_stale_tier(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    models = _models([1, 2, 3, 4, 5])
    _record_pass1(db, rid, "q1", 1)
    assert seed_pending_answers(db, rid, models).seeded == 1
    with session_scope(db) as s:
        s.execute(
            select(Pass1Result).where(Pass1Result.query_id == "q1")
        ).scalar_one().router_selected_tier = 3
    r2 = seed_pending_answers(db, rid, models)
    assert (r2.replaced, r2.seeded, r2.kept) == (1, 1, 0)
    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.query_id == "q1")).scalars().all()
    assert len(rows) == 1 and rows[0].tier_level == 3


def test_seed_replaces_dropped_model(tmp_path: Path) -> None:
    """A model removed from the routed tier's config → its stale row deleted."""
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q2", 3)
    seed_pending_answers(db, rid, ModelsConfig(tiers=[_multi(3, ["a", "b"]), _tier(5)]))
    r2 = seed_pending_answers(db, rid, ModelsConfig(tiers=[_multi(3, ["a"]), _tier(5)]))
    assert r2.replaced == 1 and r2.kept == 1
    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.query_id == "q2")).scalars().all()
    assert {r.model_id for r in rows} == {"a"}


# ---- run_answers ----

@pytest.mark.asyncio
async def test_run_answers_persists_each_model(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", 1)
    _record_pass1(db, rid, "q2", 3)
    models = _models([1, 2, 3, 4, 5])
    seed_pending_answers(db, rid, models)
    report = await run_answers(db, rid, models=models, clients_by_model=_clients([1, 2, 3]))
    assert (report.attempted, report.succeeded, report.errors) == (2, 2, 0)
    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.run_id == rid)).scalars().all()
    for r in rows:
        prompt = "easy" if r.query_id == "q1" else "harder"
        assert r.response_text == f"[tier{r.tier_level}] {prompt}"
        assert r.status == "success"


@pytest.mark.asyncio
async def test_run_answers_calls_every_model_in_routed_tier(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q2", 3)
    models = ModelsConfig(tiers=[_multi(3, ["gpt-5", "gemini"]), _tier(5)])
    seed_pending_answers(db, rid, models)
    clients = {
        (3, "gpt-5"): _FakeClient(3, "gpt-5"),
        (3, "gemini"): _FakeClient(3, "gemini"),
    }
    report = await run_answers(db, rid, models=models, clients_by_model=clients)
    assert (report.attempted, report.succeeded) == (2, 2)
    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.query_id == "q2")).scalars().all()
    by_model = {r.model_id: r.response_text for r in rows}
    assert by_model == {
        "gpt-5": "[gpt-5] harder", "gemini": "[gemini] harder",
    }


@pytest.mark.asyncio
async def test_run_answers_errors_dont_fail_pass_and_resume(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", 1)
    _record_pass1(db, rid, "q2", 2)
    models = _models([1, 2, 5])
    seed_pending_answers(db, rid, models)
    r1 = await run_answers(
        db, rid, models=models, clients_by_model=_clients([1, 2], fail_on_query="harder")
    )
    assert r1.succeeded == 1 and r1.errors == 1
    assert {qid for qid, _, _, _ in r1.error_rows} == {"q2"}
    # Resume with healthy clients — only the errored row retries.
    r2 = await run_answers(db, rid, models=models, clients_by_model=_clients([1, 2]))
    assert (r2.attempted, r2.succeeded, r2.errors) == (1, 1, 0)


@pytest.mark.asyncio
async def test_run_answers_tier_filter(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", 1)
    _record_pass1(db, rid, "q2", 3)
    models = _models([1, 2, 3, 4, 5])
    seed_pending_answers(db, rid, models)
    report = await run_answers(
        db, rid, models=models, clients_by_model=_clients([1, 2, 3]), tier_level=1
    )
    assert report.attempted == 1
    with session_scope(db) as s:
        by_qid = {
            r.query_id: r.status
            for r in s.execute(select(TierAnswer).where(TierAnswer.run_id == rid)).scalars()
        }
    assert by_qid == {"q1": "success", "q2": "pending"}


@pytest.mark.asyncio
async def test_reset_answers_tier_filter(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", 1)
    _record_pass1(db, rid, "q2", 3)
    models = _models([1, 2, 3, 4, 5])
    seed_pending_answers(db, rid, models)
    await run_answers(db, rid, models=models, clients_by_model=_clients([1, 2, 3]))
    assert reset_answers(db, rid, tier_level=1) == 1
    with session_scope(db) as s:
        rows = s.execute(select(TierAnswer).where(TierAnswer.run_id == rid)).scalars().all()
    assert {(r.query_id, r.tier_level) for r in rows} == {("q2", 3)}


@pytest.mark.asyncio
async def test_run_answers_missing_client_is_error(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", 1)
    models = _models([1, 5])
    seed_pending_answers(db, rid, models)
    report = await run_answers(db, rid, models=models, clients_by_model={})
    assert report.errors == 1
    assert all("no model" in msg for _, _, _, msg in report.error_rows)
    with session_scope(db) as s:
        row = s.execute(select(TierAnswer).where(TierAnswer.query_id == "q1")).scalar_one()
    assert row.status == "error"


@pytest.mark.asyncio
async def test_run_answers_forwards_extra_body_and_max_tokens(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q1", 1)
    _record_pass1(db, rid, "q2", 2)
    t1 = _tier(1, [TierModel(slot=0, url="http://x/v1", served_model_name="tier1",
                             extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                             max_tokens=4096)])
    t2 = _tier(2, [TierModel(slot=0, url="http://x/v1", served_model_name="tier2")])
    models = ModelsConfig(tiers=[t1, t2, _tier(5)])
    seed_pending_answers(db, rid, models)
    cap = {(1, "tier1"): _CapturingClient(1), (2, "tier2"): _CapturingClient(2)}
    await run_answers(db, rid, models=models, clients_by_model=cap, max_tokens=512)
    assert cap[(1, "tier1")].seen_extra == [{"chat_template_kwargs": {"enable_thinking": False}}]
    assert cap[(1, "tier1")].seen_max_tokens == [4096]
    assert cap[(2, "tier2")].seen_extra == [None]
    assert cap[(2, "tier2")].seen_max_tokens == [512]


@pytest.mark.asyncio
async def test_run_answers_progress_lines_name_model_and_provider(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _record_pass1(db, rid, "q2", 3)
    models = ModelsConfig(tiers=[_multi(3, ["gpt-5", "gemini"]), _tier(5)])
    seed_pending_answers(db, rid, models)
    lines: list[str] = []
    clients = {(3, "gpt-5"): _FakeClient(3, "gpt-5"), (3, "gemini"): _FakeClient(3, "gemini")}
    await run_answers(db, rid, models=models, clients_by_model=clients, progress=lines.append)
    joined = "\n".join(lines)
    assert "gpt-5 (P0)" in joined and "gemini (P1)" in joined
    assert " OK " in joined and "... running" in joined


# ---- run_smoke (SMOKE=true) ----

def _patch_smoke_clients(monkeypatch, mapping: dict) -> None:
    """Override `client_from_model` so smoke uses our fake clients."""
    monkeypatch.setattr(
        answers_mod, "client_from_model",
        lambda m: mapping[m.served_model_name],
    )


@pytest.mark.asyncio
async def test_run_smoke_probes_every_non_top_tier_model(monkeypatch) -> None:
    """Default scope: every model in every non-top tier."""
    models = ModelsConfig(tiers=[
        _multi(1, ["t1a", "t1b"]),
        _multi(3, ["t3a", "t3b"]),
        _multi(5, ["t5a"]),  # top tier — must be skipped by default
    ])
    seen: list[tuple[str, int | None]] = []

    class _Probe:
        def __init__(self, name): self.name = name
        async def chat(self, prompt, *, max_tokens=None, extra=None, **_):
            seen.append((self.name, max_tokens))
            return ChatResult(content="pong", model=self.name,
                              prompt_tokens=2, completion_tokens=1,
                              latency_ms=7, raw={})

    fakes = {n: _Probe(n) for n in ("t1a", "t1b", "t3a", "t3b", "t5a")}
    _patch_smoke_clients(monkeypatch, fakes)

    lines: list[str] = []
    report = await run_smoke(models, progress=lines.append)
    assert (report.attempted, report.ok, report.errors) == (4, 4, 0)
    # Top tier not probed.
    assert {n for n, _ in seen} == {"t1a", "t1b", "t3a", "t3b"}
    # Tiny budget per probe (the contract).
    assert all(tok == 16 for _, tok in seen)
    # Progress line names the model + provider.
    joined = "\n".join(lines)
    assert "t1a (P0)" in joined and " OK " in joined


@pytest.mark.asyncio
async def test_run_smoke_classifies_errors_per_endpoint(monkeypatch) -> None:
    """A failing probe doesn't kill the rest; outcomes are labelled."""
    from benchmark.tiers import ChatError
    models = ModelsConfig(tiers=[
        _multi(3, ["good", "bad-key", "bad-model"]),
        _multi(5, ["t5"]),
    ])

    class _OK:
        async def chat(self, *a, **kw):
            return ChatResult(content="pong", model="good",
                              prompt_tokens=2, completion_tokens=1,
                              latency_ms=5, raw={})

    class _BadKey:
        async def chat(self, *a, **kw):
            raise ChatError("HTTP 401 from https://x/v1: invalid_api_key")

    class _BadModel:
        async def chat(self, *a, **kw):
            raise ChatError(
                "HTTP 404 from https://x/v1 model='bad-model': "
                "the model 'bad-model' does not exist — model-name mismatch"
            )

    fakes = {"good": _OK(), "bad-key": _BadKey(), "bad-model": _BadModel()}
    _patch_smoke_clients(monkeypatch, fakes)

    report = await run_smoke(models)
    assert (report.attempted, report.ok, report.errors) == (3, 1, 2)
    labels = {row[1]: row[3] for row in report.error_rows}
    assert "401" in labels["bad-key"]
    assert "does not exist" in labels["bad-model"]


@pytest.mark.asyncio
async def test_run_smoke_tier_filter_includes_top_tier(monkeypatch) -> None:
    """Explicit --tier=N probes that tier (even if it's the top tier),
    so the user can verify update-gold credentials too."""
    models = ModelsConfig(tiers=[
        _multi(1, ["t1"]),
        _multi(5, ["opus", "gpt5"]),
    ])

    class _OK:
        def __init__(self, n): self.n = n
        async def chat(self, *a, **kw):
            return ChatResult(content="pong", model=self.n,
                              prompt_tokens=2, completion_tokens=1,
                              latency_ms=5, raw={})

    fakes = {n: _OK(n) for n in ("t1", "opus", "gpt5")}
    _patch_smoke_clients(monkeypatch, fakes)

    report = await run_smoke(models, tier_level=5)
    assert (report.attempted, report.ok) == (2, 2)
    # tier 1 not probed when tier_level=5 is explicit.


@pytest.mark.asyncio
async def test_run_smoke_mock_endpoint_skips_client_from_model(monkeypatch) -> None:
    """With a mock endpoint, the smoke builds clients pointing at the
    mock URL directly (no api_key resolution, no provider extra_body)."""
    models = ModelsConfig(tiers=[_multi(1, ["a"]), _multi(5, ["top"])])

    captured: list[str] = []

    class _FakeOAIClient:
        def __init__(self, *, endpoint, model_id, api_key, timeout_s):
            captured.append(endpoint)
            self.endpoint = endpoint
        async def chat(self, *a, **kw):
            return ChatResult(content="pong", model="a", prompt_tokens=1,
                              completion_tokens=1, latency_ms=1, raw={})

    monkeypatch.setattr(answers_mod, "OAIClient", _FakeOAIClient)
    # client_from_model should NOT be reached in mock mode.
    monkeypatch.setattr(
        answers_mod, "client_from_model",
        lambda m: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    report = await run_smoke(models, mock_endpoint="http://mock/v1")
    assert report.ok == 1  # only non-top "a"
    assert captured == ["http://mock/v1"]
