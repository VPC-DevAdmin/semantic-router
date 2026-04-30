"""Gold answer generation.

`make gold` calls the configured gold endpoint for each query that lacks a gold
answer, persists the result to the DB (canonical) and to `data/gold/<id>.md`
(diffable). Resumable: re-running picks up only queries still missing gold,
unless `--refresh` is passed.

TTS-only queries are skipped — text gold doesn't apply to audio output. Audio
gold is M5 territory.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from .config import Attachment, EndpointConfig, load_endpoint
from .db import Query, session_scope
from .tiers import OAIClient, client_from_endpoint


@dataclass
class GoldReport:
    generated: int = 0
    skipped_existing: int = 0
    skipped_tts: int = 0
    errors: int = 0
    error_ids: list[tuple[str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"  generated:        {self.generated}",
            f"  skipped (exists): {self.skipped_existing}",
            f"  skipped (tts):    {self.skipped_tts}",
            f"  errors:           {self.errors}",
        ]
        for qid, err in self.error_ids:
            lines.append(f"    {qid}: {err}")
        return "\n".join(lines)


def _is_tts_only(specs: list[str] | None) -> bool:
    return bool(specs) and all(s == "tts" for s in specs or [])


def _format_gold_file(query_id: str, prompt: str, model: str, generated_at: datetime, answer: str) -> str:
    return (
        f"# {query_id}\n\n"
        f"- **model**: `{model}`\n"
        f"- **generated_at**: {generated_at.isoformat()}\n\n"
        f"## Prompt\n\n{prompt}\n\n"
        f"## Gold answer\n\n{answer}\n"
    )


async def generate_gold(
    db_path: Path,
    gold_config_path: Path,
    gold_dir: Path,
    *,
    refresh: bool = False,
    only: list[str] | None = None,
    concurrency: int = 4,
    client: OAIClient | None = None,
    cfg: EndpointConfig | None = None,
) -> GoldReport:
    """Generate gold answers. `client` and `cfg` are injectable for tests."""
    cfg = cfg or load_endpoint(gold_config_path)
    client = client or client_from_endpoint(cfg)
    report = GoldReport()

    with session_scope(db_path) as session:
        stmt = select(Query)
        if only:
            stmt = stmt.where(Query.query_id.in_(only))
        rows = session.execute(stmt).scalars().all()
        # Detach from session — we re-fetch per row when persisting.
        snapshot = [
            {
                "query_id": q.query_id,
                "prompt": q.prompt,
                "attachments": list(q.attachments or []),
                "specializations": list(q.specializations or []),
                "has_gold": q.gold_answer is not None,
            }
            for q in rows
        ]

    targets: list[dict] = []
    for snap in snapshot:
        if _is_tts_only(snap["specializations"]):
            report.skipped_tts += 1
            continue
        if snap["has_gold"] and not refresh:
            report.skipped_existing += 1
            continue
        targets.append(snap)

    sem = asyncio.Semaphore(concurrency)
    gold_dir.mkdir(parents=True, exist_ok=True)

    async def run_one(snap: dict) -> None:
        async with sem:
            qid = snap["query_id"]
            try:
                attachments = [Attachment.model_validate(a) for a in snap["attachments"]]
                # mypy: client guaranteed non-None at this point
                assert client is not None
                assert cfg is not None
                result = await client.chat(
                    snap["prompt"],
                    attachments=attachments,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
                generated_at = datetime.now(UTC)
                with session_scope(db_path) as session:
                    db_q = session.execute(
                        select(Query).where(Query.query_id == qid)
                    ).scalar_one()
                    db_q.gold_answer = result.content
                    db_q.gold_model = cfg.model_id
                    db_q.gold_generated_at = generated_at
                (gold_dir / f"{qid}.md").write_text(
                    _format_gold_file(
                        qid, snap["prompt"], cfg.model_id, generated_at, result.content
                    ),
                    encoding="utf-8",
                )
                report.generated += 1
            except Exception as e:  # noqa: BLE001 — capture per-row, surface in report
                report.errors += 1
                report.error_ids.append((qid, f"{type(e).__name__}: {e}"))

    await asyncio.gather(*(run_one(s) for s in targets))
    return report
