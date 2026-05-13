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

DEFAULT_DB_PATH = Path("benchmark.db")


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
    gold_answer: Mapped[str | None] = mapped_column(Text)
    gold_model: Mapped[str | None] = mapped_column(String)
    gold_generated_at: Mapped[datetime | None] = mapped_column(DateTime)


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


def make_engine(db_path: Path = DEFAULT_DB_PATH):
    # check_same_thread=False lets the async passes share a connection pool safely
    # under our per-row commit pattern; we still serialize writes via the session.
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
