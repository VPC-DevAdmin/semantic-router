"""LLM-as-judge scoring of pass-2 responses.

For each successful pass2_result that hasn't yet been scored by the configured
judge, build a rubric-grounded prompt comparing the response to the gold
answer, call the judge endpoint, parse a JSON verdict, and persist a row in
`scores` with scorer='judge'. Resumable on the (run_id, query_id, scorer,
reviewer_id) primary key.

The judge is just another OpenAI-compatible endpoint — same `OAIClient` we
use for gold and tier calls.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from .config import EndpointConfig, ScoringConfig, load_endpoint, load_scoring
from .db import Pass2Result, Query, Score, session_scope
from .tiers import OAIClient, client_from_endpoint


@dataclass
class JudgeReport:
    attempted: int = 0
    scored: int = 0
    skipped_no_gold: int = 0
    skipped_already_scored: int = 0
    parse_errors: int = 0
    other_errors: int = 0
    error_ids: list[tuple[str, str]] = field(default_factory=list)
    score_histogram: dict[int, int] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"  attempted:        {self.attempted}",
            f"  scored:           {self.scored}",
            f"  skipped (no gold):       {self.skipped_no_gold}",
            f"  skipped (already scored):{self.skipped_already_scored}",
            f"  parse errors:     {self.parse_errors}",
            f"  other errors:     {self.other_errors}",
        ]
        if self.score_histogram:
            lines.append("  histogram:")
            for k in sorted(self.score_histogram):
                lines.append(f"    {k}: {self.score_histogram[k]}")
        for qid, msg in self.error_ids[:10]:
            lines.append(f"    {qid}: {msg}")
        return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are evaluating LLM responses against a gold-standard reference. "
    "Be strict but fair. Output only valid JSON."
)


def build_judge_prompt(rubric: ScoringConfig, query: str, gold: str, response: str) -> str:
    """Render a rubric-grounded comparison prompt.

    The model is asked to emit JSON. We tolerate fences and extra prose via
    `_parse_verdict`, but ask for clean JSON to keep parse rate high.
    """
    scale = "\n".join(f"  {k}: {v}" for k, v in sorted(rubric.scale.items()))
    return (
        f"Rubric (score from 1 to {max(rubric.scale)}):\n{scale}\n\n"
        f"Query:\n{query}\n\n"
        f"Gold answer:\n{gold}\n\n"
        f"Candidate response:\n{response}\n\n"
        'Output ONLY this JSON object (no prose, no fences): '
        '{"score": <integer in scale>, "rationale": "<one or two sentences>"}'
    )


_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_verdict(text: str, max_score: int) -> tuple[int, str]:
    """Extract {score, rationale} from a judge response.

    Tolerates code fences and a small amount of leading/trailing prose. Raises
    ValueError if no parseable object is found or the score is out of range.
    """
    # Try direct first.
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
    candidate = re.sub(r"\s*```\s*$", "", candidate)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            raise ValueError(f"no JSON object found in judge output: {text!r}") from None
        obj = json.loads(match.group(0))

    if not isinstance(obj, dict) or "score" not in obj:
        raise ValueError(f"verdict missing 'score' key: {obj!r}")
    score = int(obj["score"])
    if not 1 <= score <= max_score:
        raise ValueError(f"score {score} outside 1..{max_score}")
    rationale = str(obj.get("rationale", "")).strip()
    return score, rationale


async def judge_run(
    db_path: Path,
    run_id: int,
    *,
    judge_config_path: Path,
    scoring_config_path: Path,
    concurrency: int = 4,
    client: OAIClient | None = None,
    cfg: EndpointConfig | None = None,
    rubric: ScoringConfig | None = None,
) -> JudgeReport:
    cfg = cfg or load_endpoint(judge_config_path)
    rubric = rubric or load_scoring(scoring_config_path)
    client = client or client_from_endpoint(cfg)
    report = JudgeReport()
    reviewer_id = cfg.model_id

    with session_scope(db_path) as session:
        rows = session.execute(
            select(Pass2Result, Query)
            .join(Query, Pass2Result.query_id == Query.query_id)
            .where(Pass2Result.run_id == run_id)
            .where(Pass2Result.status == "success")
        ).all()

        existing = {
            r[0]
            for r in session.execute(
                select(Score.query_id)
                .where(Score.run_id == run_id)
                .where(Score.scorer == "judge")
                .where(Score.reviewer_id == reviewer_id)
            ).all()
        }

        snapshot = []
        for p2, q in rows:
            if p2.query_id in existing:
                report.skipped_already_scored += 1
                continue
            if not q.gold_answer:
                report.skipped_no_gold += 1
                continue
            snapshot.append(
                {
                    "query_id": p2.query_id,
                    "prompt": q.prompt,
                    "gold": q.gold_answer,
                    "response": p2.response_text or "",
                }
            )

    max_score = max(rubric.scale)
    sem = asyncio.Semaphore(concurrency)

    async def score_one(snap: dict) -> None:
        async with sem:
            qid = snap["query_id"]
            try:
                judge_prompt = build_judge_prompt(
                    rubric, snap["prompt"], snap["gold"], snap["response"]
                )
                result = await client.chat(
                    judge_prompt,
                    system=SYSTEM_PROMPT,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
                try:
                    score, rationale = _parse_verdict(result.content, max_score)
                except ValueError as ve:
                    report.parse_errors += 1
                    report.error_ids.append((qid, f"parse: {ve}"))
                    return

                with session_scope(db_path) as session:
                    session.add(
                        Score(
                            run_id=run_id,
                            query_id=qid,
                            scorer="judge",
                            reviewer_id=reviewer_id,
                            score=score,
                            rubric_version=rubric.rubric_version,
                            rationale=rationale,
                            scored_at=datetime.now(UTC),
                        )
                    )
                report.scored += 1
                report.score_histogram[score] = report.score_histogram.get(score, 0) + 1
            except Exception as e:  # noqa: BLE001
                report.other_errors += 1
                report.error_ids.append((qid, f"{type(e).__name__}: {e}"))
            finally:
                report.attempted += 1

    await asyncio.gather(*(score_one(s) for s in snapshot))
    return report
