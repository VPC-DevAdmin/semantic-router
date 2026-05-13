"""Human-review TUI tests (driven without a TTY via injectable callables)."""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from sqlalchemy import select

from benchmark.config import ScoringConfig
from benchmark.db import Pass2Result, Query, Score, session_scope
from benchmark.review import _stratified_sample, human_review
from benchmark.runs import create_run, seed_pending

from ._helpers import bootstrap_db, make_models_yaml, make_router_yaml


def _rubric() -> ScoringConfig:
    return ScoringConfig(
        rubric_version="v1",
        scale={1: "unusable", 2: "weak", 3: "ok", 4: "close", 5: "matches"},
    )


def _quiet_console() -> Console:
    return Console(file=open("/dev/null", "w"), force_terminal=False)  # noqa: SIM115


def _populate(tmp_path: Path, queries: list[dict]) -> tuple[Path, int]:
    db = bootstrap_db(tmp_path, queries)
    rid = create_run(
        db,
        router_config_path=make_router_yaml(tmp_path),
        models_config_path=make_models_yaml(tmp_path),
    )
    seed_pending(db, rid)
    # Mark every pass2 as success with a response.
    with session_scope(db) as s:
        for p2 in s.execute(select(Pass2Result)).scalars():
            p2.response_text = f"response-{p2.query_id}"
            p2.status = "success"
    return db, rid


QUERIES_SIMPLE = [
    {
        "id": "a1", "prompt": "p1",
        "expected_answer": "gold-a1",
        "expected_min_tier": 1, "specializations": ["general"],
    },
    {
        "id": "a2", "prompt": "p2",
        "expected_answer": "gold-a2",
        "expected_min_tier": 1, "specializations": ["general"],
    },
]


def _scripted(answers: list[str]):
    it = iter(answers)
    return lambda _rubric: next(it)


def _no_rationale():
    return ""


# ---- Stratified sampler ----

def test_stratified_sample_proportional() -> None:
    import random
    rng = random.Random(0)
    items = [{"k": "coding"}] * 60 + [{"k": "math"}] * 30 + [{"k": "general"}] * 10
    out = _stratified_sample(items, 20, key=lambda x: x["k"], rng=rng)
    assert len(out) == 20
    buckets = {"coding": 0, "math": 0, "general": 0}
    for x in out:
        buckets[x["k"]] += 1
    for v in buckets.values():
        assert v >= 1


def test_stratified_sample_returns_all_when_n_exceeds() -> None:
    import random
    items = [{"k": "a"}, {"k": "b"}]
    out = _stratified_sample(items, 10, key=lambda x: x["k"], rng=random.Random(0))
    assert len(out) == 2


# ---- human_review ----

def test_review_persists_scores(tmp_path: Path) -> None:
    db, rid = _populate(tmp_path, QUERIES_SIMPLE)
    report = human_review(
        db, rid,
        reviewer_id="alice", scoring_config_path=Path("/dev/null"),
        ask_score=_scripted(["4", "2"]), ask_rationale=_no_rationale,
        console=_quiet_console(), rubric=_rubric(),
    )
    assert report.reviewed == 2
    assert report.score_histogram == {4: 1, 2: 1}
    with session_scope(db) as s:
        rows = {r.query_id: r for r in s.execute(select(Score)).scalars()}
        assert rows["a1"].score == 4
        assert rows["a2"].score == 2


def test_review_skip_and_quit(tmp_path: Path) -> None:
    db, rid = _populate(tmp_path, QUERIES_SIMPLE)
    report = human_review(
        db, rid,
        reviewer_id="bob", scoring_config_path=Path("/dev/null"),
        ask_score=_scripted(["s", "q"]), ask_rationale=_no_rationale,
        console=_quiet_console(), rubric=_rubric(),
    )
    assert report.skipped == 1
    assert report.quit_early is True


def test_review_idempotent_per_reviewer(tmp_path: Path) -> None:
    db, rid = _populate(tmp_path, QUERIES_SIMPLE)
    human_review(
        db, rid,
        reviewer_id="alice", scoring_config_path=Path("/dev/null"),
        ask_score=_scripted(["3", "3"]), ask_rationale=_no_rationale,
        console=_quiet_console(), rubric=_rubric(),
    )
    r2 = human_review(
        db, rid,
        reviewer_id="alice", scoring_config_path=Path("/dev/null"),
        ask_score=_scripted([]), ask_rationale=_no_rationale,
        console=_quiet_console(), rubric=_rubric(),
    )
    assert r2.candidates == 0
    r3 = human_review(
        db, rid,
        reviewer_id="bob", scoring_config_path=Path("/dev/null"),
        ask_score=_scripted(["5", "5"]), ask_rationale=_no_rationale,
        console=_quiet_console(), rubric=_rubric(),
    )
    assert r3.reviewed == 2


def test_review_invalid_input_is_skipped(tmp_path: Path) -> None:
    db, rid = _populate(tmp_path, QUERIES_SIMPLE)
    report = human_review(
        db, rid,
        reviewer_id="alice", scoring_config_path=Path("/dev/null"),
        ask_score=_scripted(["nope", "9"]), ask_rationale=_no_rationale,
        console=_quiet_console(), rubric=_rubric(),
    )
    assert report.reviewed == 0
    assert report.skipped == 2


def test_review_skips_rows_without_gold(tmp_path: Path) -> None:
    db, rid = _populate(tmp_path, QUERIES_SIMPLE)
    with session_scope(db) as s:
        q = s.execute(select(Query).where(Query.query_id == "a1")).scalar_one()
        q.gold_answer = None
    report = human_review(
        db, rid,
        reviewer_id="alice", scoring_config_path=Path("/dev/null"),
        ask_score=_scripted(["3"]), ask_rationale=_no_rationale,
        console=_quiet_console(), rubric=_rubric(),
    )
    assert report.candidates == 1
    assert report.reviewed == 1


def test_review_sample_size(tmp_path: Path) -> None:
    many = [
        {
            "id": f"q{i:03d}", "prompt": "p",
            "expected_answer": "gold",
            "expected_min_tier": 1, "specializations": ["general"],
        }
        for i in range(20)
    ]
    db, rid = _populate(tmp_path, many)
    report = human_review(
        db, rid,
        reviewer_id="alice", scoring_config_path=Path("/dev/null"),
        sample=5, seed=42,
        ask_score=_scripted(["3"] * 5), ask_rationale=_no_rationale,
        console=_quiet_console(), rubric=_rubric(),
    )
    assert report.candidates == 20
    assert report.reviewed == 5
