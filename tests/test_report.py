"""Report aggregation tests.

Builds a controlled DB state — known pass1 outcomes, pass2 statuses, and
score rows — and asserts that compute_report returns the right counts,
percentages, and per-spec breakdown. Also smoke-tests JSON/CSV export.
"""
from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from benchmark.db import Pass1Result, Pass2Result, Score, init_db, session_scope
from benchmark.report import compute_report, export_csv, export_json, render_console, to_dict
from benchmark.runs import create_run, seed_pending
from benchmark.seed import seed_from_yaml

QUERIES = """
- id: r1
  prompt: "p1"
  expected_min_tier: 2
  specializations: [general]
- id: r2
  prompt: "p2"
  expected_min_tier: 3
  specializations: [code]
- id: r3
  prompt: "p3"
  expected_min_tier: 4
  specializations: [math, reasoning]
- id: r4
  prompt: "p4"
  expected_min_tier: 1
  specializations: [general]
"""


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
    return db, rid


def _set_pass1(db: Path, qid: str, *,
               status: str = "success",
               tier: int | None = 3,
               specs: list[str] | None = None,
               meets: int | None = 1,
               matches: int | None = 1) -> None:
    with session_scope(db) as s:
        row = s.execute(
            select(Pass1Result).where(Pass1Result.query_id == qid)
        ).scalar_one()
        row.status = status
        row.router_selected_tier = tier
        row.router_selected_specs = specs
        row.meets_minimum_tier = meets
        row.matches_specialization = matches


def _set_pass2(db: Path, qid: str, status: str = "success") -> None:
    with session_scope(db) as s:
        row = s.execute(
            select(Pass2Result).where(Pass2Result.query_id == qid)
        ).scalar_one()
        row.status = status
        row.response_text = f"response-{qid}"


def _add_score(db: Path, rid: int, qid: str, *,
               scorer: str = "judge", reviewer: str = "judge-x", score: int = 4) -> None:
    with session_scope(db) as s:
        s.add(Score(
            run_id=rid, query_id=qid, scorer=scorer, reviewer_id=reviewer,
            score=score, rubric_version="v1", rationale=None,
            scored_at=datetime.now(UTC),
        ))


def test_basic_topline(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _set_pass1(db, "r1", tier=3, meets=1, matches=1)
    _set_pass1(db, "r2", tier=2, meets=0, matches=0)
    _set_pass1(db, "r3", tier=5, meets=1, matches=1)
    _set_pass1(db, "r4", status="error", meets=None, matches=None)

    _set_pass2(db, "r1", "success")
    _set_pass2(db, "r2", "success")
    _set_pass2(db, "r3", "error")
    # r4 left pending

    _add_score(db, rid, "r1", score=5)
    _add_score(db, rid, "r2", score=2)
    _add_score(db, rid, "r1", scorer="human", reviewer="alice", score=4)

    rep = compute_report(db, rid)
    assert rep.run_id == rid
    assert rep.pass1.total == 4
    assert rep.pass1.success == 3
    assert rep.pass1.error == 1
    assert rep.pass1.pending == 0
    assert rep.pass1.meets_min_tier == 2
    assert rep.pass1.matches_spec == 2

    assert rep.pass2_success == 2
    assert rep.pass2_error == 1
    assert rep.pass2_pending == 1

    assert "judge:judge-x" in rep.scorers
    assert rep.scorers["judge:judge-x"].n == 2
    assert rep.scorers["judge:judge-x"].mean == 3.5
    assert "human:alice" in rep.scorers
    assert rep.scorers["human:alice"].n == 1


def test_unknown_tier_counted(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _set_pass1(db, "r1", tier=None, meets=None, matches=None)
    _set_pass1(db, "r2", tier=None, meets=None, matches=None)
    _set_pass1(db, "r3", tier=4, meets=1, matches=1)
    _set_pass1(db, "r4", tier=1, meets=1, matches=1)
    rep = compute_report(db, rid)
    assert rep.pass1.unknown_tier == 2


def test_per_spec_breakdown(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    # r1=general, r2=code, r3=math+reasoning, r4=general
    _set_pass1(db, "r1", tier=3, meets=1, matches=1)
    _set_pass1(db, "r2", tier=4, meets=1, matches=1)
    _set_pass1(db, "r3", tier=3, meets=0, matches=0)
    _set_pass1(db, "r4", tier=1, meets=1, matches=1)

    rep = compute_report(db, rid)
    assert rep.per_spec_pass1["general"].total == 2
    assert rep.per_spec_pass1["general"].meets_min_tier == 2
    assert rep.per_spec_pass1["code"].total == 1
    assert rep.per_spec_pass1["math"].total == 1
    assert rep.per_spec_pass1["reasoning"].total == 1
    assert rep.per_spec_pass1["math"].meets_min_tier == 0


def test_per_spec_judge_mean(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    for qid in ("r1", "r2", "r3", "r4"):
        _set_pass1(db, qid, tier=3, meets=1, matches=1)
        _set_pass2(db, qid, "success")
    _add_score(db, rid, "r1", score=5)  # general
    _add_score(db, rid, "r4", score=3)  # general → mean 4
    _add_score(db, rid, "r2", score=4)  # code → mean 4
    _add_score(db, rid, "r3", score=2)  # math+reasoning → mean 2 for both

    rep = compute_report(db, rid)
    assert rep.per_spec_judge_mean["general"] == 4.0
    assert rep.per_spec_judge_mean["code"] == 4.0
    assert rep.per_spec_judge_mean["math"] == 2.0
    assert rep.per_spec_judge_mean["reasoning"] == 2.0


def test_json_export(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _set_pass1(db, "r1", tier=3, meets=1, matches=1)
    rep = compute_report(db, rid)
    out = tmp_path / "out.json"
    export_json(rep, out)
    loaded = json.loads(out.read_text())
    assert loaded["run_id"] == rid
    assert loaded["pass1"]["total"] == 4


def test_csv_export(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _set_pass1(db, "r1", tier=3, meets=1, matches=1)
    _set_pass1(db, "r2", tier=4, meets=1, matches=1)
    rep = compute_report(db, rid)
    out = tmp_path / "out.csv"
    export_csv(rep, out)

    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert any(r["specialization"] == "_all" for r in rows)
    assert any(r["specialization"] == "general" for r in rows)
    assert any(r["specialization"] == "code" for r in rows)


def test_render_console_does_not_crash(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    _set_pass1(db, "r1", tier=3, meets=1, matches=1)
    _set_pass2(db, "r1", "success")
    _add_score(db, rid, "r1", score=4)

    rep = compute_report(db, rid)
    from rich.console import Console
    render_console(rep, Console(file=open("/dev/null", "w"), force_terminal=False))  # noqa: SIM115


def test_to_dict_roundtrip_keys(tmp_path: Path) -> None:
    db, rid = _bootstrap(tmp_path)
    rep = compute_report(db, rid)
    d = to_dict(rep)
    assert set(d.keys()) >= {
        "run_id", "status", "pass1", "pass2",
        "scorers", "per_spec_pass1", "per_spec_judge_mean",
    }
