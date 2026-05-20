"""Routed-tier answer collection — backs `make answers`.

The router picks ONE tier per query. That tier can front several models
(Anthropic / OpenAI / Google …); we call EVERY one of them directly
(bypassing the router — it already chose) so the demo can show how the
answer changes across providers. There is one `tier_answers` row per
(query, routed tier, model); this fills each one.

Error policy (per user design): an unreachable or erroring upstream marks
the row `status='error'` with `error_msg` populated. The pass keeps
going; nothing fails. Re-running `make answers` retries errored rows.

Resumable: workers select rows where `status IN ('pending', 'error')`.
Per-row session commits make killing the process mid-run safe.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from .config import Attachment, ModelsConfig
from .db import Query, TierAnswer, session_scope
from .tiers import OAIClient, client_from_model

# A model is addressed by (tier_level, served_model_name) — unique per run.
ModelKey = tuple[int, str]


@dataclass
class AnswersReport:
    attempted: int = 0
    succeeded: int = 0
    errors: int = 0
    # (qid, tier_level, model_id, msg)
    error_rows: list[tuple[str, int, str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"  attempted: {self.attempted}",
            f"  succeeded: {self.succeeded}",
            f"  errors:    {self.errors}",
        ]
        for qid, level, model_id, msg in self.error_rows[:10]:
            lines.append(f"    [tier {level} {model_id}] {qid}: {msg}")
        if len(self.error_rows) > 10:
            lines.append(f"    ... and {len(self.error_rows) - 10} more")
        return "\n".join(lines)


def _clients_by_model(models: ModelsConfig) -> dict[ModelKey, OAIClient]:
    """One OAIClient per (tier level, model). Reused across all queries."""
    out: dict[ModelKey, OAIClient] = {}
    for tier in models.tiers:
        for m in tier.resolved_models():
            out[(tier.level, m.served_model_name)] = client_from_model(m)
    return out


def _extra_body_by_model(models: ModelsConfig) -> dict[ModelKey, dict]:
    """Per-model `extra_body` merged into each chat request.

    Carries provider-specific knobs the OpenAI schema doesn't model —
    e.g. Qwen3's `chat_template_kwargs: {enable_thinking: false}` or the
    anti-repetition sampler. Each slot can differ (per-slot THINKING).
    """
    out: dict[ModelKey, dict] = {}
    for tier in models.tiers:
        for m in tier.resolved_models():
            if isinstance(m.extra_body, dict) and m.extra_body:
                out[(tier.level, m.served_model_name)] = m.extra_body
    return out


def _max_tokens_by_model(models: ModelsConfig) -> dict[ModelKey, int]:
    """Per-model generation cap (TIER{N}[_i]_MAX_TOKENS). Absent → global."""
    out: dict[ModelKey, int] = {}
    for tier in models.tiers:
        for m in tier.resolved_models():
            if m.max_tokens is not None:
                out[(tier.level, m.served_model_name)] = m.max_tokens
    return out


def _clients_for_mock(
    models: ModelsConfig, mock_endpoint: str
) -> dict[ModelKey, OAIClient]:
    """Every model points at one mock endpoint — used by `make answers MOCK`."""
    out: dict[ModelKey, OAIClient] = {}
    for tier in models.tiers:
        for m in tier.resolved_models():
            out[(tier.level, m.served_model_name)] = OAIClient(
                endpoint=mock_endpoint,
                model_id=m.served_model_name,
                api_key=None,
                timeout_s=float(m.timeout_s),
            )
    return out


@dataclass
class SmokeReport:
    """Outcome of `run_smoke` — connectivity probe per (tier, model)."""
    attempted: int = 0
    ok: int = 0
    errors: int = 0
    # (tier_level, model_id, provider_or_empty, msg)
    error_rows: list[tuple[int, str, str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"  attempted: {self.attempted}",
            f"  ok:        {self.ok}",
            f"  errors:    {self.errors}",
        ]
        for level, model_id, provider, msg in self.error_rows[:10]:
            prov = f" ({provider})" if provider else ""
            lines.append(f"    [tier {level} {model_id}{prov}] {msg}")
        if len(self.error_rows) > 10:
            lines.append(f"    ... and {len(self.error_rows) - 10} more")
        return "\n".join(lines)


# Smoke probe prompt + budget. Tiny enough that vendor cost is negligible,
# benign enough that no safety filter will refuse, but big enough that
# reasoning models (Gemini 2.5+, OpenAI o-series, GPT-5) which spend
# tokens on hidden reasoning still have headroom to emit visible output.
_SMOKE_PROMPT = "Reply with the single word: pong."
_SMOKE_MAX_TOKENS = 64


async def run_smoke(
    models: ModelsConfig,
    *,
    concurrency: int = 8,
    tier_level: int | None = None,
    mock_endpoint: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> SmokeReport:
    """Tiny chat probe of every (tier, model) `make answers` would call.

    No DB writes, no run id needed. Verifies for each model:
      • URL is reachable,
      • the API key is accepted,
      • the server recognises the model name.

    Scope mirrors `make answers`: by default, every non-top tier (the
    top tier is the gold reference and isn't called by answers). An
    explicit `tier_level` overrides that and probes ONLY that tier
    (including the top, useful for checking update-gold credentials).

    `mock_endpoint` points every probe at the local mock — useful for
    exercising the smoke harness itself.
    """
    top = max((t.level for t in models.tiers), default=0)
    if tier_level is not None:
        in_scope = [t for t in models.tiers if t.level == tier_level]
    else:
        in_scope = [t for t in models.tiers if t.level != top]

    # Flatten to (tier, model) targets, preserving declared order.
    targets: list[tuple[int, object, str | None]] = []  # (level, TierModel, mock_url|None)
    for tier in in_scope:
        for m in tier.resolved_models():
            targets.append((tier.level, m, mock_endpoint))

    total = len(targets)
    report = SmokeReport()

    def _emit(msg: str) -> None:
        if progress is not None:
            progress(msg)

    def _prefix(idx: int, level: int, model_id: str, provider: str | None) -> str:
        prov = f" ({provider})" if provider else ""
        return f"[{idx:>{len(str(max(total, 1)))}}/{total}] tier{level}  {model_id}{prov}"

    sem = asyncio.Semaphore(concurrency)

    async def probe(idx: int, level: int, m, mock: str | None) -> None:
        async with sem:
            provider = m.provider
            model_id = m.served_model_name
            _emit(f"{_prefix(idx, level, model_id, provider)}  ... pinging")
            try:
                # Build the client lazily so MissingApiKeyError surfaces
                # per-target instead of crashing the whole smoke.
                if mock is not None:
                    client = OAIClient(
                        endpoint=mock, model_id=model_id,
                        api_key=None, timeout_s=float(m.timeout_s),
                    )
                    extra = None
                else:
                    client = client_from_model(m)
                    extra = m.extra_body if isinstance(m.extra_body, dict) else None
                result = await client.chat(
                    _SMOKE_PROMPT, max_tokens=_SMOKE_MAX_TOKENS, extra=extra,
                )
                latency = result.latency_ms or 0
                _emit(f"{_prefix(idx, level, model_id, provider)}  OK  {latency} ms")
                report.ok += 1
            except Exception as e:  # noqa: BLE001
                label = _status_label(e)
                _emit(
                    f"{_prefix(idx, level, model_id, provider)}  "
                    f"{label}  {_one_line(str(e))}"
                )
                report.errors += 1
                report.error_rows.append(
                    (level, model_id, provider or "", f"{type(e).__name__}: {e}")
                )
            finally:
                report.attempted += 1

    await asyncio.gather(
        *(probe(i, lvl, m, mock) for i, (lvl, m, mock) in enumerate(targets, start=1))
    )
    return report


def _status_label(exc: Exception) -> str:
    """One-word outcome for the live progress line (mirrors ChatError)."""
    name = type(exc).__name__
    msg = str(exc)
    if "Timeout" in name or "Timeout" in msg:
        return "TIMEOUT"
    if msg.lstrip().startswith("HTTP ") or "HTTP " in msg[:16]:
        return "HTTP-ERR"
    if "ConnectError" in name or "could not reach the backend" in msg:
        return "UNREACHABLE"
    return "ERROR"


def _one_line(s: str, limit: int = 200) -> str:
    """Collapse whitespace and truncate so a detail fits on one line."""
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


async def run_answers(
    db_path: Path,
    run_id: int,
    *,
    models: ModelsConfig,
    concurrency: int = 8,
    max_tokens: int = 2048,
    clients_by_model: dict[ModelKey, OAIClient] | None = None,
    mock_endpoint: str | None = None,
    tier_level: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> AnswersReport:
    """Process pending/error tier_answers rows, one per (query, tier, model).

    `clients_by_model` (keyed by `(tier_level, served_model_name)`) is
    injectable for tests. `mock_endpoint` overrides every model's
    endpoint to the local mock. `tier_level` restricts the worker to a
    single tier (other tiers' rows untouched). `progress`, if given, is
    called with a preformatted one-line string at the start and end of
    each model call so `make answers` shows a running list.
    """
    if clients_by_model is None:
        if mock_endpoint:
            clients_by_model = _clients_for_mock(models, mock_endpoint)
        else:
            clients_by_model = _clients_by_model(models)

    # Per-model request knobs. Skipped for the mock (it ignores them).
    extra_by_model = {} if mock_endpoint else _extra_body_by_model(models)
    max_tokens_by_model = {} if mock_endpoint else _max_tokens_by_model(models)

    report = AnswersReport()

    # Snapshot the pending rows. No DB session is held across the async
    # fan-out; each worker opens its own short-lived session.
    with session_scope(db_path) as session:
        q = (
            select(TierAnswer, Query)
            .join(Query, TierAnswer.query_id == Query.query_id)
            .where(TierAnswer.run_id == run_id)
            .where(TierAnswer.status.in_(["pending", "error"]))
        )
        if tier_level is not None:
            q = q.where(TierAnswer.tier_level == tier_level)
        rows = session.execute(q).all()
        snapshot = [
            {
                "query_id": ta.query_id,
                "tier_level": ta.tier_level,
                "model_id": ta.model_id,
                "provider": ta.provider,
                "prompt": q.prompt,
                "attachments": list(q.attachments or []),
            }
            for (ta, q) in rows
        ]

    total = len(snapshot)

    def _emit(msg: str) -> None:
        if progress is not None:
            progress(msg)

    def _prefix(idx: int, snap: dict) -> str:
        prov = f" ({snap['provider']})" if snap.get("provider") else ""
        return (
            f"[{idx:>{len(str(total))}}/{total}] {snap['query_id']}  "
            f"tier{snap['tier_level']}  {snap['model_id']}{prov}"
        )

    def _update_row(db_path: Path, qid: str, level: int, model_id: str):
        return (
            select(TierAnswer)
            .where(TierAnswer.run_id == run_id)
            .where(TierAnswer.query_id == qid)
            .where(TierAnswer.tier_level == level)
            .where(TierAnswer.model_id == model_id)
        )

    sem = asyncio.Semaphore(concurrency)

    async def run_one(idx: int, snap: dict) -> None:
        async with sem:
            qid = snap["query_id"]
            level = snap["tier_level"]
            model_id = snap["model_id"]
            key: ModelKey = (level, model_id)
            attempted_at = datetime.now(UTC)
            _emit(f"{_prefix(idx, snap)}  ... running")
            client = clients_by_model.get(key)
            if client is None:
                msg = f"no model {model_id!r} configured for tier {level}"
                report.errors += 1
                report.error_rows.append((qid, level, model_id, msg))
                _emit(f"{_prefix(idx, snap)}  UNREACHABLE  {msg}")
                with session_scope(db_path) as session:
                    row = session.execute(
                        _update_row(db_path, qid, level, model_id)
                    ).scalar_one()
                    row.status = "error"
                    row.error_msg = msg
                    row.attempted_at = attempted_at
                report.attempted += 1
                return

            try:
                attachments = [
                    Attachment.model_validate(a) for a in snap["attachments"]
                ]
                result = await client.chat(
                    snap["prompt"],
                    attachments=attachments,
                    max_tokens=max_tokens_by_model.get(key, max_tokens),
                    extra=extra_by_model.get(key) or None,
                )
                with session_scope(db_path) as session:
                    row = session.execute(
                        _update_row(db_path, qid, level, model_id)
                    ).scalar_one()
                    row.response_text = result.content
                    row.prompt_tokens = result.prompt_tokens
                    row.completion_tokens = result.completion_tokens
                    row.latency_ms = result.latency_ms
                    row.status = "success"
                    row.error_msg = None
                    row.attempted_at = attempted_at
                report.succeeded += 1
                secs = (result.latency_ms or 0) / 1000
                toks = result.completion_tokens
                tok_str = f"{toks} tok" if toks is not None else "? tok"
                _emit(f"{_prefix(idx, snap)}  OK  {secs:.1f}s  {tok_str}")
            except Exception as e:  # noqa: BLE001
                report.errors += 1
                report.error_rows.append(
                    (qid, level, model_id, f"{type(e).__name__}: {e}")
                )
                _emit(
                    f"{_prefix(idx, snap)}  {_status_label(e)}  "
                    f"{_one_line(str(e))}"
                )
                with session_scope(db_path) as session:
                    row = session.execute(
                        _update_row(db_path, qid, level, model_id)
                    ).scalar_one()
                    row.status = "error"
                    row.error_msg = f"{type(e).__name__}: {e}"
                    row.attempted_at = attempted_at
            finally:
                report.attempted += 1

    await asyncio.gather(
        *(run_one(i, s) for i, s in enumerate(snapshot, start=1))
    )
    return report
