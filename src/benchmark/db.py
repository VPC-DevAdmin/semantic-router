"""SQLAlchemy schema and session helpers.

SQLite is the canonical run store. Schema mirrors PLAN.md exactly. Per-row commits
during pass execution are what give us resume safety; nothing in this module needs
to know about that — it just provides the models and a session factory.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

DEFAULT_DB_PATH = Path("data/router_benchmark.db")


class Base(DeclarativeBase):
    pass


class Query(Base):
    __tablename__ = "queries"

    query_id: Mapped[str] = mapped_column(String, primary_key=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String, nullable=False)
    attachments: Mapped[list | None] = mapped_column("attachments_json", JSON)
    expected_min_tier: Mapped[int] = mapped_column(Integer, nullable=False)
    specializations: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    domain_tags: Mapped[list[str] | None] = mapped_column(JSON)
    notes: Mapped[str | None] = mapped_column(Text)
    # Per-provider gold lives in the gold_answers table (keyed by
    # (query_id, model_id)). There is no longer a single-value gold
    # mirror on Query — use the gold_answers join instead.


class Run(Base):
    __tablename__ = "runs"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    router_config_hash: Mapped[str] = mapped_column(String, nullable=False)
    models_config_hash: Mapped[str] = mapped_column(String, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, nullable=False)  # running|done|aborted


class Pass1Result(Base):
    __tablename__ = "pass1_results"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.run_id"), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("queries.query_id"), primary_key=True)
    router_selected_model: Mapped[str | None] = mapped_column(String)
    router_selected_tier: Mapped[int | None] = mapped_column(Integer)
    router_selected_specs: Mapped[list[str] | None] = mapped_column(JSON)
    meets_minimum_tier: Mapped[int | None] = mapped_column(Integer)
    matches_specialization: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    raw_routing_metadata: Mapped[dict | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String, nullable=False)  # pending|success|error
    error_msg: Mapped[str | None] = mapped_column(Text)
    attempted_at: Mapped[datetime | None] = mapped_column(DateTime)


class TierAnswer(Base):
    """One row per (run, query, tier_level, model) — `make answers` fills these.

    A tier can front several models (Anthropic / OpenAI / Google …); the
    router picks a tier and we call EVERY model configured for it, so a
    routed query produces one row per model in that tier. Each row stores
    the response from calling that model's endpoint directly (NOT through
    the router). `make export` reads these as the per-query routed answers.
    """

    __tablename__ = "tier_answers"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.run_id"), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("queries.query_id"), primary_key=True)
    tier_level: Mapped[int] = mapped_column(Integer, primary_key=True)
    # The model's served name — unique within a tier, so it completes the
    # PK. One row per model that the routed tier fronts.
    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_slot: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    provider: Mapped[str | None] = mapped_column(String)  # optional label → export JSON
    tier_name: Mapped[str] = mapped_column(String, nullable=False)  # router_alias
    response_text: Mapped[str | None] = mapped_column(Text)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String, nullable=False)  # pending|success|error
    error_msg: Mapped[str | None] = mapped_column(Text)
    attempted_at: Mapped[datetime | None] = mapped_column(DateTime)


class Evaluation(Base):
    """One LLM-judge verdict for a (routed answer × gold answer) pair.

    Keyed by (run_id, query_id, routed_tier, routed_model, gold_model_id,
    evaluator) so every comparison can be judged independently and so
    multiple evaluators can be run side-by-side without collisions.

    Populated by `make evaluate` (the batched judge worker) and by the
    one-off `benchmark import-evaluations` CLI for externally-produced
    judgments. `make export` reads these rows to emit
    `data/evaluations.json`.

    Score scale (per dimension, 1-4):
      4 — fully meets the dimension
      3 — meets with minor issues
      2 — partially meets
      1 — does not meet

    Four dimensions:
      correctness          — is the CORE answer factually/logically right?
      completeness         — does it cover what the question requires?
      fitness_for_purpose  — is the format, length, and tone appropriate?
      soundness            — are the SUPPORTING claims factually accurate
                              (no misleading errors beyond the core answer)?

    Verdict alphabet:
      Adequate  — correct and fit for purpose; minor verbosity /
                  formatting / style differences vs the gold are fine.
      Marginal  — partially correct or useful but has notable gaps,
                  factual errors in supporting content (typically
                  flagged by low soundness), or quality issues a real
                  user would notice.
      Failure   — factually wrong on the core question, misleading,
                  or so incomplete it fails the user.
    """

    __tablename__ = "evaluations"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.run_id"), primary_key=True)
    query_id: Mapped[str] = mapped_column(
        ForeignKey("queries.query_id"), primary_key=True
    )
    # Which routed answer was being evaluated.
    routed_tier: Mapped[int] = mapped_column(Integer, primary_key=True)
    routed_model: Mapped[str] = mapped_column(String, primary_key=True)
    # Which gold answer it was compared against (gold_answers.model_id).
    gold_model_id: Mapped[str] = mapped_column(String, primary_key=True)
    # Which judge model produced this verdict.
    evaluator: Mapped[str] = mapped_column(String, primary_key=True)

    # Optional labels mirrored from the source row for export readability.
    routed_provider: Mapped[str | None] = mapped_column(String)
    gold_provider: Mapped[str | None] = mapped_column(String)
    evaluator_provider: Mapped[str | None] = mapped_column(String)

    # Judge outputs.
    verdict: Mapped[str | None] = mapped_column(String)
    rationale: Mapped[str | None] = mapped_column(Text)
    correctness: Mapped[int | None] = mapped_column(Integer)
    completeness: Mapped[int | None] = mapped_column(Integer)
    fitness_for_purpose: Mapped[int | None] = mapped_column(Integer)
    soundness: Mapped[int | None] = mapped_column(Integer)

    # Lifecycle (mirrors TierAnswer for resume semantics).
    status: Mapped[str] = mapped_column(String, nullable=False)  # pending|success|error
    error_msg: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime)


class GoldAnswer(Base):
    """Per-provider expected answer for a query (the gold set).

    Keyed by (query_id, model_id) so each provider's gold is independent.
    Populated by `make load` (from queries.json), `make update-gold`
    (top-tier model calls), and `make import-answers` for externally
    produced answers. `make export` emits these as the query's
    `expected_answers[]`.
    """

    __tablename__ = "gold_answers"

    query_id: Mapped[str] = mapped_column(
        ForeignKey("queries.query_id"), primary_key=True
    )
    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    provider: Mapped[str | None] = mapped_column(String)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    generated_at: Mapped[datetime | None] = mapped_column(DateTime)


def make_engine(db_path: Path = DEFAULT_DB_PATH):
    # check_same_thread=False lets the async passes share a connection pool safely
    # under our per-row commit pattern; we still serialize writes via the session.
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{db_path}",
        future=True,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _enable_sqlite_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    return engine


def init_db(db_path: Path = DEFAULT_DB_PATH) -> Path:
    engine = make_engine(db_path)
    Base.metadata.create_all(engine)
    return db_path


def make_session_factory(db_path: Path = DEFAULT_DB_PATH) -> sessionmaker[Session]:
    return sessionmaker(bind=make_engine(db_path), expire_on_commit=False, future=True)


@contextmanager
def session_scope(db_path: Path = DEFAULT_DB_PATH) -> Iterator[Session]:
    factory = make_session_factory(db_path)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
