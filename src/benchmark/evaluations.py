"""Batched LLM-judge evaluator — backs `make evaluate`.

For each pending `Evaluation` row, we need a verdict from a judge LLM
comparing a routed-tier answer against a gold answer. A naive
implementation would issue one API call per row (640 queries × 3
routed × 2 gold × 1 evaluator = 3,840 calls). External operators
observed that batching 50 queries' worth of comparisons into a single
call reduces token cost by ~75% without measurable quality loss, so
we follow the same pattern.

Lifecycle per evaluator slot:

  1. Snapshot all `pending` / `error` rows for the active run.
  2. Group rows by `query_id` and chunk into batches of `BATCH_SIZE`
     queries (default 50). All comparison pairs for those queries land
     in a single batch.
  3. Build a multi-query judge prompt, call the evaluator, parse the
     JSON-array response.
  4. Match each response entry back to its DB row by
     (query_id, routed_model, gold_model) and upsert with status =
     'success' (if matched and valid) or 'error' (if missing,
     malformed, or rejected by judge).
  5. Concurrency: one in-flight batch per evaluator slot at a time
     (the BATCH_SIZE is the parallelism budget; per-batch parallelism
     would just multiply token spend with no quality gain).

The on-disk schema in `db.Evaluation` is one row per
(run, query, routed_model, gold_model, evaluator). Per-row resume
preserves the standard pattern: workers re-process where status IN
('pending', 'error').
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from .config import EvaluatorSlot
from .db import Evaluation, GoldAnswer, Query, TierAnswer, session_scope
from .tiers import ChatError, OAIClient, client_from_evaluator

DEFAULT_BATCH_SIZE = 50
DEFAULT_CONCURRENCY = 1  # one in-flight batch per evaluator; the BATCH is the parallelism

# Score scale used in the judge prompt; surfaces in DB columns 1-4.
_DIMENSIONS = ("correctness", "completeness", "fitness_for_purpose", "soundness")
_VERDICTS = ("Adequate", "Marginal", "Failure")


# ─────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationsReport:
    attempted_batches: int = 0
    succeeded_rows: int = 0
    errored_rows: int = 0
    by_evaluator: dict[str, dict[str, int]] = field(default_factory=dict)

    def bump(self, evaluator: str, status: str) -> None:
        d = self.by_evaluator.setdefault(evaluator, {"success": 0, "error": 0})
        d[status] = d.get(status, 0) + 1
        if status == "success":
            self.succeeded_rows += 1
        elif status == "error":
            self.errored_rows += 1


# ─────────────────────────────────────────────────────────────────────────
# Seeding — populate `evaluations` rows from existing routed + gold data
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class SeedResult:
    seeded: int = 0
    kept: int = 0


def seed_pending_evaluations(
    db_path: Path,
    run_id: int,
    evaluators: list[EvaluatorSlot],
) -> SeedResult:
    """Insert one pending row per (query, routed answer, gold answer,
    evaluator) combination that doesn't already exist for this run.

    Source data:
      • `tier_answers` with status='success' supply the routed answers.
      • `gold_answers` supply the gold answers (one per (query, model_id)).
    Cross-product within each query.

    Idempotent: existing rows are left alone.
    """
    result = SeedResult()
    if not evaluators:
        return result

    with session_scope(db_path) as session:
        # Snapshot routed answers (one per routed-tier × routed-model × query).
        routed = list(
            session.execute(
                select(TierAnswer)
                .where(TierAnswer.run_id == run_id)
                .where(TierAnswer.status == "success")
            ).scalars()
        )
        # Snapshot gold answers (per-query × per-provider).
        gold = list(session.execute(select(GoldAnswer)).scalars())
        gold_by_qid: dict[str, list[GoldAnswer]] = {}
        for g in gold:
            gold_by_qid.setdefault(g.query_id, []).append(g)

        # Existing evaluation rows (for idempotency).
        existing = {
            (e.query_id, e.routed_tier, e.routed_model, e.gold_model_id, e.evaluator)
            for e in session.execute(
                select(Evaluation).where(Evaluation.run_id == run_id)
            ).scalars()
        }

        for ra in routed:
            for ga in gold_by_qid.get(ra.query_id, []):
                for ev in evaluators:
                    key = (
                        ra.query_id, ra.tier_level, ra.model_id,
                        ga.model_id, ev.served_model_name,
                    )
                    if key in existing:
                        result.kept += 1
                        continue
                    session.add(Evaluation(
                        run_id=run_id,
                        query_id=ra.query_id,
                        routed_tier=ra.tier_level,
                        routed_model=ra.model_id,
                        gold_model_id=ga.model_id,
                        evaluator=ev.served_model_name,
                        routed_provider=ra.provider,
                        gold_provider=ga.provider,
                        evaluator_provider=ev.provider,
                        status="pending",
                    ))
                    result.seeded += 1
    return result


# ─────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT_HEADER = """\
You are evaluating whether AI candidate responses adequately answer
queries, compared to known-good reference answers.

For EACH item below, score the candidate on three 1-4 dimensions and
assign an overall verdict.

Score scale (per dimension):
  4 — fully meets the dimension (no concerns)
  3 — meets with minor issues
  2 — partially meets (real-user impact)
  1 — does not meet

Dimensions (each 1-4):
  correctness          — is the CORE answer factually and logically right?
  completeness         — does it cover what the question requires?
  fitness_for_purpose  — appropriate format, length, tone?
  soundness            — are the SUPPORTING claims also factually accurate?
                          (Distinct from correctness — a candidate can have
                          a right core answer but include misleading
                          supporting details. Soundness flags THAT case.)

Overall verdict alphabet:
  Adequate  — correct and fit for purpose. Minor verbosity, formatting,
              or style differences vs the reference are fine; the
              candidate doesn't have to match exactly, it just has to
              serve the user correctly.
  Marginal  — partially correct or useful but has notable gaps, factual
              errors in supporting content, or quality issues that
              would matter to a real user.
  Failure   — factually wrong on the core question, misleading, or so
              incomplete it fails the user.

Respond with EXACTLY a JSON array (no markdown, no surrounding text):
[
  {
    "eval_id": "<the eval_id provided for this item>",
    "verdict": "Adequate" | "Marginal" | "Failure",
    "rationale": "1-3 sentences focused on what tipped your verdict",
    "scores": {
      "correctness": 1-4,
      "completeness": 1-4,
      "fitness_for_purpose": 1-4,
      "soundness": 1-4
    }
  },
  ...
]

The array MUST contain exactly one entry per `eval_id` provided below.
"""


def build_judge_prompt(items: list[dict]) -> str:
    """Compose the multi-query judge prompt from a list of items.

    Each item carries:
      • eval_id        — opaque key the judge echoes back so we can match
      • query_id, prompt
      • routed_model, routed_answer
      • gold_model, gold_answer

    Items are grouped by query in the rendered prompt so the judge can
    re-use the prompt context across the (routed × gold) pairs.
    """
    by_query: dict[str, list[dict]] = {}
    for it in items:
        by_query.setdefault(it["query_id"], []).append(it)

    blocks: list[str] = [JUDGE_PROMPT_HEADER]
    for qid, group in by_query.items():
        prompt = group[0]["prompt"]
        blocks.append("\n══════════════════════════════════════════")
        blocks.append(f"QUERY {qid}:\n{prompt}")
        # List the reference answers once (deduplicated by gold_model).
        seen_gold: set[str] = set()
        for it in group:
            if it["gold_model"] in seen_gold:
                continue
            seen_gold.add(it["gold_model"])
            blocks.append(
                f"\nREFERENCE (gold from {it['gold_model']}):\n{it['gold_answer']}"
            )
        # List the candidate answers once (deduplicated by routed_model).
        seen_routed: set[str] = set()
        for it in group:
            if it["routed_model"] in seen_routed:
                continue
            seen_routed.add(it["routed_model"])
            blocks.append(
                f"\nCANDIDATE ({it['routed_model']}):\n{it['routed_answer']}"
            )
        # Then the (eval_id × routed × gold) pairing for the judge to fill.
        blocks.append("\nEVALUATE these (eval_id, routed, gold) pairs:")
        for it in group:
            blocks.append(
                f"  eval_id={it['eval_id']}  "
                f"routed={it['routed_model']}  gold={it['gold_model']}"
            )

    return "\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────

class JudgeParseError(ValueError):
    """The judge response didn't match the contract."""


def parse_judge_response(text: str) -> dict[str, dict]:
    """Parse a JSON-array response into {eval_id: verdict_dict}.

    Validates each entry has verdict, rationale, scores (with all three
    dimensions, each int in 1..4). Raises JudgeParseError on any
    structural problem. Returns a partial dict if some entries are
    valid and others aren't — the caller treats missing eval_ids as
    errors so this preserves "succeeded ones don't waste a retry."
    """
    # Strip common LLM markdown fences if the model ignored the
    # "no markdown" instruction.
    s = text.strip()
    if s.startswith("```"):
        # `lstrip` on the first line to drop ```json / ```
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]

    try:
        arr = json.loads(s)
    except json.JSONDecodeError as e:
        raise JudgeParseError(f"response is not valid JSON: {e}") from e
    if not isinstance(arr, list):
        raise JudgeParseError(f"response is not a JSON array (got {type(arr).__name__})")

    out: dict[str, dict] = {}
    for entry in arr:
        if not isinstance(entry, dict):
            continue  # skip; caller marks missing eval_ids as errors
        eval_id = entry.get("eval_id")
        verdict = entry.get("verdict")
        rationale = entry.get("rationale") or ""
        scores = entry.get("scores") or {}
        if not isinstance(eval_id, str) or not eval_id:
            continue
        if verdict not in _VERDICTS:
            continue
        if not isinstance(scores, dict):
            continue
        try:
            dims = {d: int(scores[d]) for d in _DIMENSIONS}
        except (KeyError, ValueError, TypeError):
            continue
        if not all(1 <= dims[d] <= 4 for d in _DIMENSIONS):
            continue
        out[eval_id] = {
            "verdict": verdict,
            "rationale": str(rationale),
            "scores": dims,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────
# Batch worker
# ─────────────────────────────────────────────────────────────────────────

def _eval_id(query_id: str, routed_provider: str | None, routed_model: str,
             gold_provider: str | None, gold_model: str, evaluator: str) -> str:
    """Stable eval_id with evaluator suffix so multi-evaluator runs
    don't collide. Matches the format we export to evaluations.json."""

    def _slug(s: str | None) -> str:
        return (s or "unknown").replace(" ", "-").replace(".", "_").lower()

    return (
        f"{query_id}-{_slug(routed_provider)}-{_slug(routed_model)}"
        f"-vs-{_slug(gold_provider)}-{_slug(gold_model)}"
        f"--{_slug(evaluator)}"
    )


async def run_evaluations(
    db_path: Path,
    run_id: int,
    *,
    evaluators: list[EvaluatorSlot],
    batch_size: int = DEFAULT_BATCH_SIZE,
    clients_by_evaluator: dict[str, OAIClient] | None = None,
    progress: Callable[[str], None] | None = None,
) -> EvaluationsReport:
    """Process pending/error evaluation rows for the active run.

    `clients_by_evaluator` is injectable for tests; production calls
    build one OAIClient per evaluator slot at startup.
    """
    report = EvaluationsReport()
    if not evaluators:
        return report

    if clients_by_evaluator is None:
        clients_by_evaluator = {
            ev.served_model_name: client_from_evaluator(ev) for ev in evaluators
        }

    # One pass per evaluator. Batches inside an evaluator run sequentially
    # so we don't burst-spend on a slow judge; users can run multiple
    # evaluators concurrently by issuing separate make-evaluate calls.
    for ev in evaluators:
        client = clients_by_evaluator[ev.served_model_name]
        await _run_one_evaluator(
            db_path, run_id, ev, client,
            batch_size=batch_size, report=report, progress=progress,
        )

    return report


async def _run_one_evaluator(
    db_path: Path,
    run_id: int,
    evaluator: EvaluatorSlot,
    client: OAIClient,
    *,
    batch_size: int,
    report: EvaluationsReport,
    progress: Callable[[str], None] | None,
) -> None:
    """Snapshot pending rows for this evaluator, group into query
    batches, run each batch end-to-end."""
    # Snapshot the rows + everything we need to render the prompt.
    with session_scope(db_path) as session:
        rows = session.execute(
            select(Evaluation, Query, TierAnswer, GoldAnswer)
            .join(Query, Evaluation.query_id == Query.query_id)
            .join(
                TierAnswer,
                (TierAnswer.run_id == Evaluation.run_id)
                & (TierAnswer.query_id == Evaluation.query_id)
                & (TierAnswer.tier_level == Evaluation.routed_tier)
                & (TierAnswer.model_id == Evaluation.routed_model),
            )
            .join(
                GoldAnswer,
                (GoldAnswer.query_id == Evaluation.query_id)
                & (GoldAnswer.model_id == Evaluation.gold_model_id),
            )
            .where(Evaluation.run_id == run_id)
            .where(Evaluation.evaluator == evaluator.served_model_name)
            .where(Evaluation.status.in_(["pending", "error"]))
        ).all()
        # Detach into plain dicts so we don't hold a session across the
        # async fan-out.
        snapshot: list[dict] = []
        for ev_row, q, ta, ga in rows:
            snapshot.append({
                "eval_id": _eval_id(
                    q.query_id, ev_row.routed_provider, ev_row.routed_model,
                    ev_row.gold_provider, ev_row.gold_model_id,
                    evaluator.served_model_name,
                ),
                "query_id": q.query_id,
                "prompt": q.prompt,
                "routed_tier": ev_row.routed_tier,
                "routed_model": ev_row.routed_model,
                "routed_answer": ta.response_text or "",
                "gold_model": ev_row.gold_model_id,
                "gold_answer": ga.answer,
            })

    if not snapshot:
        return

    # Group by query_id, then chunk by batch_size queries.
    by_query: dict[str, list[dict]] = {}
    for item in snapshot:
        by_query.setdefault(item["query_id"], []).append(item)
    query_ids = list(by_query.keys())
    batches: list[list[dict]] = []
    for i in range(0, len(query_ids), batch_size):
        chunk_qids = query_ids[i : i + batch_size]
        batches.append([it for qid in chunk_qids for it in by_query[qid]])

    if progress:
        progress(
            f"[evaluator {evaluator.served_model_name}] "
            f"{len(snapshot)} rows in {len(batches)} batches "
            f"(batch_size={batch_size} queries)"
        )

    for bi, batch in enumerate(batches, start=1):
        report.attempted_batches += 1
        await _run_one_batch(
            db_path, run_id, evaluator, client, batch,
            report=report, progress=progress, batch_idx=bi, batch_count=len(batches),
        )


async def _run_one_batch(
    db_path: Path,
    run_id: int,
    evaluator: EvaluatorSlot,
    client: OAIClient,
    items: list[dict],
    *,
    report: EvaluationsReport,
    progress: Callable[[str], None] | None,
    batch_idx: int,
    batch_count: int,
) -> None:
    """Send one batch to the judge; upsert each row by eval_id."""
    prompt = build_judge_prompt(items)
    t0 = time.perf_counter()
    try:
        result = await client.chat(
            prompt, max_tokens=evaluator.max_tokens, temperature=0.0,
        )
        parsed = parse_judge_response(result.content)
        batch_error: str | None = None
    except (ChatError, JudgeParseError) as e:
        parsed = {}
        batch_error = f"{type(e).__name__}: {e}"
    latency_ms = int((time.perf_counter() - t0) * 1000)

    with session_scope(db_path) as session:
        for it in items:
            row = session.execute(
                select(Evaluation)
                .where(Evaluation.run_id == run_id)
                .where(Evaluation.query_id == it["query_id"])
                .where(Evaluation.routed_tier == it["routed_tier"])
                .where(Evaluation.routed_model == it["routed_model"])
                .where(Evaluation.gold_model_id == it["gold_model"])
                .where(Evaluation.evaluator == evaluator.served_model_name)
            ).scalar_one()
            verdict = parsed.get(it["eval_id"])
            row.evaluated_at = datetime.now(UTC)
            row.latency_ms = latency_ms
            if verdict is not None:
                row.verdict = verdict["verdict"]
                row.rationale = verdict["rationale"]
                row.correctness = verdict["scores"]["correctness"]
                row.completeness = verdict["scores"]["completeness"]
                row.fitness_for_purpose = verdict["scores"]["fitness_for_purpose"]
                row.soundness = verdict["scores"]["soundness"]
                row.status = "success"
                row.error_msg = None
                report.bump(evaluator.served_model_name, "success")
            else:
                row.status = "error"
                row.error_msg = (
                    batch_error
                    or f"judge omitted eval_id {it['eval_id']!r} from response"
                )
                report.bump(evaluator.served_model_name, "error")

    if progress:
        ok = sum(1 for it in items if it["eval_id"] in parsed)
        progress(
            f"[evaluator {evaluator.served_model_name}] "
            f"batch {batch_idx}/{batch_count}: "
            f"{ok}/{len(items)} ok ({latency_ms} ms)"
            + (f" — {batch_error}" if batch_error else "")
        )


# Cluster the asyncio import + coroutine boilerplate so unit tests can
# instantiate the worker without needing httpx mocks for every test.
def synchronous_run(*args, **kwargs) -> EvaluationsReport:
    """Sync wrapper used by the CLI."""
    return asyncio.run(run_evaluations(*args, **kwargs))
