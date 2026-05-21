"""Tests for the LLM-judge evaluator pipeline.

Covers four layers:

  1. Env walker (config.load_evaluators) — discovers EVALUATOR_N_*
     slots, blank/missing handled, partial-config errors.
  2. Prompt builder + JSON-array parser — pure functions, no DB.
  3. Seeder (seed_pending_evaluations) — cross-product of routed
     answers × gold answers × evaluators, idempotent.
  4. Batch worker (run_evaluations) — async fan-out with the
     OAIClient mocked, asserts per-row resume + status flips.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from benchmark.config import EvaluatorSlot, load_evaluators
from benchmark.db import Evaluation, GoldAnswer, Pass1Result, TierAnswer, session_scope
from benchmark.evaluations import (
    DEFAULT_BATCH_SIZE,
    JudgeParseError,
    _eval_id,
    build_judge_prompt,
    parse_judge_response,
    run_evaluations,
    seed_pending_evaluations,
)
from benchmark.runs import create_run
from benchmark.tiers import ChatResult

from ._helpers import bootstrap_db, make_models_yaml, make_router_yaml

QUERIES = [
    {
        "id": "q1", "prompt": "What is 2+2?",
        "expected_answers": [
            {"answer": "4.", "model": "Opus", "provider": "Anthropic"},
        ],
        "expected_min_tier": 1, "specializations": ["general"],
    },
    {
        "id": "q2", "prompt": "Explain quantum entanglement.",
        "expected_answers": [
            {"answer": "Long answer.", "model": "Opus", "provider": "Anthropic"},
        ],
        "expected_min_tier": 3, "specializations": ["reasoning"],
    },
]


# ─────────────────────────────────────────────────────────────────────
# 1. Env walker
# ─────────────────────────────────────────────────────────────────────

def test_load_evaluators_empty_env(monkeypatch) -> None:
    for k in list(os.environ):
        if k.startswith("EVALUATOR_"):
            monkeypatch.delenv(k, raising=False)
    assert load_evaluators() == []


def test_load_evaluators_one_slot(monkeypatch) -> None:
    for k in list(os.environ):
        if k.startswith("EVALUATOR_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EVALUATOR_1_URL", "https://api.anthropic.com/v1")
    monkeypatch.setenv("EVALUATOR_1_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("EVALUATOR_1_API_KEY", "sk-ant-test")
    monkeypatch.setenv("EVALUATOR_1_PROVIDER", "Anthropic")
    monkeypatch.setenv("EVALUATOR_1_MAX_TOKENS", "8192")

    [slot] = load_evaluators()
    assert slot.slot == 1
    assert slot.served_model_name == "claude-sonnet-4-6"
    assert slot.api_key_env == "EVALUATOR_1_API_KEY"
    assert slot.provider == "Anthropic"
    assert slot.max_tokens == 8192


def test_load_evaluators_stops_at_first_gap(monkeypatch) -> None:
    """A missing slot ends discovery — same discipline as the tier
    walker. So EVALUATOR_3_* without 2 is silently invisible."""
    for k in list(os.environ):
        if k.startswith("EVALUATOR_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EVALUATOR_1_URL", "u1")
    monkeypatch.setenv("EVALUATOR_1_MODEL", "m1")
    monkeypatch.setenv("EVALUATOR_3_URL", "u3")  # gap at 2 — 3 ignored
    monkeypatch.setenv("EVALUATOR_3_MODEL", "m3")
    [slot] = load_evaluators()
    assert slot.served_model_name == "m1"


def test_load_evaluators_partial_slot_raises(monkeypatch) -> None:
    for k in list(os.environ):
        if k.startswith("EVALUATOR_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EVALUATOR_1_URL", "u1")  # MODEL missing
    with pytest.raises(ValueError, match="MODEL is not"):
        load_evaluators()


# ─────────────────────────────────────────────────────────────────────
# 2. Prompt + parser
# ─────────────────────────────────────────────────────────────────────

def _item(qid: str, routed: str, gold: str, *, prompt: str = "?") -> dict:
    return {
        "eval_id": _eval_id(qid, None, routed, None, gold, "judge-X"),
        "query_id": qid,
        "prompt": prompt,
        "routed_tier": 1,
        "routed_model": routed,
        "routed_answer": f"answer-from-{routed}",
        "gold_model": gold,
        "gold_answer": f"gold-from-{gold}",
    }


def test_eval_id_includes_evaluator() -> None:
    e = _eval_id("q00001", "Local-Qwen3", "Qwen3-1.7B",
                 "OpenAI", "GPT-5.5", "claude-sonnet-4-6")
    assert e.startswith("q00001-local-qwen3-qwen3-1_7b-vs-openai-gpt-5_5")
    assert e.endswith("--claude-sonnet-4-6")
    # The double-dash before the evaluator suffix is the disambiguator.
    assert "--" in e and e.split("--")[1] == "claude-sonnet-4-6"


def test_build_judge_prompt_dedupes_references_per_query() -> None:
    items = [
        _item("q1", "tiny", "opus", prompt="2+2"),
        _item("q1", "tiny", "gpt5", prompt="2+2"),
        _item("q1", "huge", "opus", prompt="2+2"),
        _item("q1", "huge", "gpt5", prompt="2+2"),
    ]
    text = build_judge_prompt(items)
    # The prompt appears once per query, not per item.
    assert text.count("2+2") == 1
    # Each unique gold appears once.
    assert text.count("gold-from-opus") == 1
    assert text.count("gold-from-gpt5") == 1
    # Each unique routed model appears once.
    assert text.count("answer-from-tiny") == 1
    assert text.count("answer-from-huge") == 1
    # The pairing list at the end has one line per eval_id.
    for it in items:
        assert it["eval_id"] in text


def test_parse_judge_response_happy_path() -> None:
    resp = json.dumps([
        {"eval_id": "a", "verdict": "Adequate", "rationale": "fine",
         "scores": {"correctness": 4, "completeness": 4,
                  "fitness_for_purpose": 4, "soundness": 4}},
        {"eval_id": "b", "verdict": "Failure", "rationale": "wrong",
         "scores": {"correctness": 1, "completeness": 2, "fitness_for_purpose": 1, "soundness": 1}},
    ])
    out = parse_judge_response(resp)
    assert set(out.keys()) == {"a", "b"}
    assert out["a"]["verdict"] == "Adequate"
    assert out["b"]["scores"]["correctness"] == 1


def test_parse_judge_response_strips_markdown_fences() -> None:
    resp = "```json\n" + json.dumps([
        {"eval_id": "a", "verdict": "Adequate", "rationale": "",
         "scores": {"correctness": 4, "completeness": 4,
                  "fitness_for_purpose": 4, "soundness": 4}},
    ]) + "\n```"
    assert "a" in parse_judge_response(resp)


def test_parse_judge_response_rejects_non_array() -> None:
    with pytest.raises(JudgeParseError, match="not a JSON array"):
        parse_judge_response('{"single": "object"}')


def test_parse_judge_response_skips_malformed_entries() -> None:
    """One good + one bad: returns just the good one. Caller marks
    missing eval_ids as errors so the good one doesn't waste a retry."""
    resp = json.dumps([
        {"eval_id": "a", "verdict": "Adequate", "rationale": "",
         "scores": {"correctness": 4, "completeness": 4,
                  "fitness_for_purpose": 4, "soundness": 4}},
        {"eval_id": "b", "verdict": "Bogus",  # not a valid verdict
         "scores": {"correctness": 4, "completeness": 4,
                  "fitness_for_purpose": 4, "soundness": 4}},
        {"eval_id": "c", "verdict": "Adequate",
         "scores": {"correctness": 5, "completeness": 4,
          "fitness_for_purpose": 4, "soundness": 4}},  # 5 out of range
    ])
    out = parse_judge_response(resp)
    assert set(out.keys()) == {"a"}


# ─────────────────────────────────────────────────────────────────────
# 3. Seeder
# ─────────────────────────────────────────────────────────────────────

def _setup_db_with_routed_and_gold(tmp_path: Path) -> tuple[Path, int]:
    """Builds a DB with: 2 queries, pass1 records, tier_answers for
    each, and gold answers."""
    db = bootstrap_db(tmp_path, QUERIES)  # loads upstream gold
    rid = create_run(
        db,
        router_config_path=make_router_yaml(tmp_path),
        models_config_path=make_models_yaml(tmp_path),
    )
    with session_scope(db) as s:
        # q1 routed to tier 1, two models.
        s.add(Pass1Result(
            run_id=rid, query_id="q1",
            router_selected_model="tier1", router_selected_tier=1,
            router_selected_specs=["general"],
            meets_minimum_tier=1, matches_specialization=1, latency_ms=10,
            raw_routing_metadata={}, status="success",
            attempted_at=datetime.now(UTC),
        ))
        for slot, mid, prov in [(0, "qwen3-tiny", "Local"), (1, "gpt-4o-mini", "OpenAI")]:
            s.add(TierAnswer(
                run_id=rid, query_id="q1", tier_level=1, model_id=mid,
                model_slot=slot, provider=prov, tier_name="tier1",
                response_text=f"answer from {mid}", status="success",
                latency_ms=100, attempted_at=datetime.now(UTC),
            ))
        # Add a second gold to q1 (cross with upstream "Opus" already there).
        s.add(GoldAnswer(
            query_id="q1", model_id="GPT-5.5", provider="OpenAI",
            answer="OpenAI gold", generated_at=datetime.now(UTC),
        ))
    return db, rid


def test_seeder_creates_one_row_per_routed_x_gold_x_evaluator(tmp_path) -> None:
    db, rid = _setup_db_with_routed_and_gold(tmp_path)
    evaluators = [
        EvaluatorSlot(slot=1, url="u", served_model_name="judge-A"),
        EvaluatorSlot(slot=2, url="u", served_model_name="judge-B"),
    ]
    result = seed_pending_evaluations(db, rid, evaluators)
    # q1: 2 routed × 2 gold × 2 evaluators = 8
    assert result.seeded == 8
    assert result.kept == 0
    with session_scope(db) as s:
        rows = list(s.execute(select(Evaluation)).scalars())
    assert len(rows) == 8
    assert {r.status for r in rows} == {"pending"}


def test_seeder_is_idempotent(tmp_path) -> None:
    """Re-running the seeder doesn't duplicate rows; existing rows
    are kept regardless of their status."""
    db, rid = _setup_db_with_routed_and_gold(tmp_path)
    ev = [EvaluatorSlot(slot=1, url="u", served_model_name="judge")]
    seed_pending_evaluations(db, rid, ev)
    # Flip one row to success and re-seed.
    with session_scope(db) as s:
        row = list(s.execute(select(Evaluation)).scalars())[0]
        row.status = "success"
        row.verdict = "Adequate"
    result2 = seed_pending_evaluations(db, rid, ev)
    assert result2.seeded == 0
    assert result2.kept == 4  # 2 routed × 2 gold × 1 evaluator
    with session_scope(db) as s:
        statuses = {r.status for r in s.execute(select(Evaluation)).scalars()}
    assert statuses == {"pending", "success"}


def test_seeder_no_evaluators_is_noop(tmp_path) -> None:
    db, rid = _setup_db_with_routed_and_gold(tmp_path)
    result = seed_pending_evaluations(db, rid, [])
    assert result.seeded == 0 and result.kept == 0


# ─────────────────────────────────────────────────────────────────────
# 4. Batch worker (judge call mocked)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class _FakeJudgeClient:
    """Async-compatible stub for OAIClient.chat. Each call records its
    args and returns the next prepared response."""
    responses: list[str]
    calls: list[dict[str, Any]]

    async def chat(self, prompt, **kwargs):  # noqa: D401
        self.calls.append({"prompt": prompt, **kwargs})
        if not self.responses:
            raise RuntimeError("FakeJudgeClient out of prepared responses")
        return ChatResult(
            content=self.responses.pop(0), model="judge",
            prompt_tokens=0, completion_tokens=0, latency_ms=0, raw={},
        )


def _good_response_for_items(items: list[dict]) -> str:
    return json.dumps([
        {
            "eval_id": it["eval_id"],
            "verdict": "Adequate",
            "rationale": "ok",
            "scores": {"correctness": 4, "completeness": 4,
                       "fitness_for_purpose": 4, "soundness": 4},
        }
        for it in items
    ])


@pytest.mark.asyncio
async def test_worker_happy_path(tmp_path) -> None:
    db, rid = _setup_db_with_routed_and_gold(tmp_path)
    ev_slot = EvaluatorSlot(slot=1, url="u", served_model_name="judge")
    seed_pending_evaluations(db, rid, [ev_slot])

    # The worker will issue ONE batch (4 items fits in batch_size=50 queries).
    # We don't yet know the eval_ids before the worker runs (they're
    # computed from rows), so let the fake build its response inline by
    # parsing the prompt for eval_ids.
    fake = _FakeJudgeClient(responses=[""], calls=[])

    async def chat(prompt, **kwargs):
        fake.calls.append({"prompt": prompt, **kwargs})
        # Pull eval_ids out of the rendered pairing block.
        lines = [ln for ln in prompt.splitlines() if ln.strip().startswith("eval_id=")]
        ids = [ln.split("eval_id=")[1].split()[0] for ln in lines]
        return ChatResult(
            content=json.dumps([
                {"eval_id": e, "verdict": "Adequate", "rationale": "ok",
                 "scores": {"correctness": 4, "completeness": 4,
                  "fitness_for_purpose": 4, "soundness": 4}}
                for e in ids
            ]),
            model="judge", prompt_tokens=0, completion_tokens=0,
            latency_ms=0, raw={},
        )
    fake.chat = chat  # type: ignore[assignment]

    report = await run_evaluations(
        db, rid, evaluators=[ev_slot], batch_size=50,
        clients_by_evaluator={"judge": fake},  # type: ignore[arg-type]
    )

    assert report.attempted_batches == 1
    assert report.succeeded_rows == 4
    assert report.errored_rows == 0

    with session_scope(db) as s:
        rows = list(s.execute(select(Evaluation)).scalars())
    assert all(r.status == "success" for r in rows)
    assert all(r.verdict == "Adequate" for r in rows)
    assert all(r.correctness == 4 for r in rows)
    assert all(r.soundness == 4 for r in rows)


@pytest.mark.asyncio
async def test_worker_partial_response_marks_missing_as_error(tmp_path) -> None:
    """The judge returns verdicts for SOME items in the batch but not
    others. The matched ones flip to success; the missing ones flip to
    error (so the next run retries just those)."""
    db, rid = _setup_db_with_routed_and_gold(tmp_path)
    ev_slot = EvaluatorSlot(slot=1, url="u", served_model_name="judge")
    seed_pending_evaluations(db, rid, [ev_slot])

    fake = _FakeJudgeClient(responses=[], calls=[])

    async def chat(prompt, **kwargs):
        fake.calls.append({})
        lines = [ln for ln in prompt.splitlines() if ln.strip().startswith("eval_id=")]
        ids = [ln.split("eval_id=")[1].split()[0] for ln in lines]
        # Return verdicts for only the FIRST eval_id; omit the rest.
        return ChatResult(
            content=json.dumps([
                {"eval_id": ids[0], "verdict": "Adequate", "rationale": "ok",
                 "scores": {"correctness": 4, "completeness": 4,
                  "fitness_for_purpose": 4, "soundness": 4}},
            ]),
            model="judge", prompt_tokens=0, completion_tokens=0,
            latency_ms=0, raw={},
        )
    fake.chat = chat  # type: ignore[assignment]

    report = await run_evaluations(
        db, rid, evaluators=[ev_slot], batch_size=50,
        clients_by_evaluator={"judge": fake},  # type: ignore[arg-type]
    )
    assert report.succeeded_rows == 1
    assert report.errored_rows == 3
    with session_scope(db) as s:
        rows = list(s.execute(select(Evaluation)).scalars())
    statuses = {r.status for r in rows}
    assert statuses == {"success", "error"}


@pytest.mark.asyncio
async def test_worker_judge_call_failure_errors_whole_batch(tmp_path) -> None:
    """ChatError from the judge means we can't attribute any of the
    batch's verdicts — all rows in the batch are marked error and
    retried next time."""
    from benchmark.tiers import ChatError

    db, rid = _setup_db_with_routed_and_gold(tmp_path)
    ev_slot = EvaluatorSlot(slot=1, url="u", served_model_name="judge")
    seed_pending_evaluations(db, rid, [ev_slot])

    class _BoomClient:
        async def chat(self, *a, **k):
            raise ChatError("timeout")
    report = await run_evaluations(
        db, rid, evaluators=[ev_slot], batch_size=50,
        clients_by_evaluator={"judge": _BoomClient()},  # type: ignore[arg-type]
    )
    assert report.succeeded_rows == 0
    assert report.errored_rows == 4
    with session_scope(db) as s:
        rows = list(s.execute(select(Evaluation)).scalars())
    assert {r.status for r in rows} == {"error"}
    assert all("ChatError" in (r.error_msg or "") for r in rows)


@pytest.mark.asyncio
async def test_worker_resumes_from_error_rows(tmp_path) -> None:
    """A second run only re-processes pending/error rows; success rows
    are left untouched."""
    db, rid = _setup_db_with_routed_and_gold(tmp_path)
    ev_slot = EvaluatorSlot(slot=1, url="u", served_model_name="judge")
    seed_pending_evaluations(db, rid, [ev_slot])
    # Pre-mark half the rows as 'success' so they should be skipped.
    with session_scope(db) as s:
        rows = list(s.execute(select(Evaluation)).scalars())
        for r in rows[:2]:
            r.status = "success"
            r.verdict = "Adequate"
            r.correctness = 4
            r.completeness = 4
            r.fitness_for_purpose = 4
            r.soundness = 4
            r.rationale = "from prior run"

    class _CountingClient:
        def __init__(self): self.items_seen = 0

        async def chat(self, prompt, **kwargs):
            lines = [ln for ln in prompt.splitlines() if ln.strip().startswith("eval_id=")]
            ids = [ln.split("eval_id=")[1].split()[0] for ln in lines]
            self.items_seen += len(ids)
            return ChatResult(
                content=json.dumps([
                    {"eval_id": e, "verdict": "Adequate", "rationale": "new",
                     "scores": {"correctness": 3, "completeness": 3,
                                "fitness_for_purpose": 3, "soundness": 3}}
                    for e in ids
                ]),
                model="judge", prompt_tokens=0, completion_tokens=0,
                latency_ms=0, raw={},
            )
    c = _CountingClient()
    await run_evaluations(
        db, rid, evaluators=[ev_slot], batch_size=50,
        clients_by_evaluator={"judge": c},  # type: ignore[arg-type]
    )
    # Only the two pending rows were re-judged.
    assert c.items_seen == 2
    with session_scope(db) as s:
        rationales = {r.rationale for r in s.execute(select(Evaluation)).scalars()}
    assert rationales == {"from prior run", "new"}


@pytest.mark.asyncio
async def test_worker_batch_size_chunks_queries(tmp_path) -> None:
    """With batch_size=1 and N queries, we get N batches. Verifies the
    chunking groups by query rather than by row."""
    db = bootstrap_db(tmp_path, QUERIES)  # 2 queries
    rid = create_run(
        db,
        router_config_path=make_router_yaml(tmp_path),
        models_config_path=make_models_yaml(tmp_path),
    )
    with session_scope(db) as s:
        for qid, tier in [("q1", 1), ("q2", 3)]:
            s.add(Pass1Result(
                run_id=rid, query_id=qid, router_selected_model=f"tier{tier}",
                router_selected_tier=tier, router_selected_specs=["general"],
                meets_minimum_tier=1, matches_specialization=1, latency_ms=10,
                raw_routing_metadata={}, status="success",
                attempted_at=datetime.now(UTC),
            ))
            s.add(TierAnswer(
                run_id=rid, query_id=qid, tier_level=tier, model_id=f"m{tier}",
                model_slot=0, provider=None, tier_name=f"tier{tier}",
                response_text="ans", status="success",
                latency_ms=10, attempted_at=datetime.now(UTC),
            ))

    ev_slot = EvaluatorSlot(slot=1, url="u", served_model_name="judge")
    seed_pending_evaluations(db, rid, [ev_slot])

    class _Client:
        batches_seen = 0

        async def chat(self, prompt, **kwargs):
            _Client.batches_seen += 1
            lines = [ln for ln in prompt.splitlines() if ln.strip().startswith("eval_id=")]
            ids = [ln.split("eval_id=")[1].split()[0] for ln in lines]
            return ChatResult(
                content=json.dumps([
                    {"eval_id": e, "verdict": "Adequate", "rationale": "",
                     "scores": {"correctness": 4, "completeness": 4,
                  "fitness_for_purpose": 4, "soundness": 4}}
                    for e in ids
                ]),
                model="judge", prompt_tokens=0, completion_tokens=0,
                latency_ms=0, raw={},
            )

    client = _Client()

    await run_evaluations(
        db, rid, evaluators=[ev_slot], batch_size=1,
        clients_by_evaluator={"judge": client},  # type: ignore[arg-type]
    )
    # Two queries × batch_size=1 = 2 batches.
    assert _Client.batches_seen == 2


def test_default_batch_size_is_50() -> None:
    """Empirically-justified default — locked so a careless refactor
    doesn't silently change the cost profile."""
    assert DEFAULT_BATCH_SIZE == 50


# ─────────────────────────────────────────────────────────────────────
# Required by the env-walker tests above.
# ─────────────────────────────────────────────────────────────────────
import os  # noqa: E402 — kept low so the test module imports first
