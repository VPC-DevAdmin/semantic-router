"""Import pre-generated tier answers from a markdown file into the DB.

Backs `make import-answers`. Used when answers are produced outside the
harness — e.g., manually prompting Claude/GPT in a chat UI for a batch
of queries, or pre-generated reference answers for a specific tier.

Markdown format expected (both separators supported):

    ## q00046 — Optional Title

    Body text for q00046's answer. May span multiple paragraphs and
    contain markdown formatting, code blocks, etc.

    ---

    ## q00068: Another Optional Title
    Body text for q00068's answer.

Each `## qNNNNN` heading starts a new section. The body is everything
between that heading and the next one (with leading/trailing whitespace
and horizontal-rule separators trimmed).

Sections without a query body, or with a body that's only whitespace,
are skipped with a warning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from .config import ModelsConfig
from .db import TierAnswer, session_scope

# Matches a heading line like `## q00046 — Title`, `## q00046: Title`,
# or `## q00046` (no title). The query id is captured in group 1.
_SECTION_HEADER = re.compile(r"^##\s+(q\d{5})\b.*$", re.MULTILINE)

# Horizontal-rule separator that sometimes appears between sections.
_HRULE = re.compile(r"^---+\s*$", re.MULTILINE)


@dataclass
class ImportResult:
    """What `import_answers_file` did."""
    parsed: int             # number of qNNNNN sections found in the file
    inserted: int           # rows created
    updated: int            # rows updated
    skipped_empty: list[str]  # qids whose body was blank
    skipped_unknown: list[str]  # qids not in the queries table

    def __str__(self) -> str:
        lines = [
            f"  parsed sections: {self.parsed}",
            f"  inserted:        {self.inserted}",
            f"  updated:         {self.updated}",
        ]
        if self.skipped_empty:
            lines.append(f"  skipped (empty): {len(self.skipped_empty)} → "
                         f"{', '.join(self.skipped_empty[:5])}"
                         f"{'...' if len(self.skipped_empty) > 5 else ''}")
        if self.skipped_unknown:
            lines.append(f"  skipped (unknown qid): {len(self.skipped_unknown)} → "
                         f"{', '.join(self.skipped_unknown[:5])}"
                         f"{'...' if len(self.skipped_unknown) > 5 else ''}")
        return "\n".join(lines)


def parse_answers_markdown(text: str) -> list[tuple[str, str]]:
    """Extract (query_id, body) pairs from a markdown file.

    Body is the text between a `## qNNNNN ...` heading and the next
    such heading (or end of file), with surrounding whitespace and
    horizontal-rule separators stripped.

    Duplicate query IDs: later sections overwrite earlier ones in the
    return order — caller should treat this as last-write-wins.
    """
    matches = list(_SECTION_HEADER.finditer(text))
    out: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        qid = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        # Strip horizontal-rule separators that often appear between
        # sections in markdown (they're not part of the answer).
        body = _HRULE.sub("", body)
        body = body.strip()
        out.append((qid, body))
    return out


def import_answers_file(
    db_path: Path,
    run_id: int,
    *,
    tier_level: int,
    file_path: Path,
    models: ModelsConfig,
) -> ImportResult:
    """Parse a markdown answers file and upsert TierAnswer rows for `run_id`.

    For each (query_id, body) pair:
      • If a TierAnswer at (run_id, query_id, tier_level) exists → update
        its response_text and mark status='success'.
      • Otherwise → insert a new TierAnswer at that tier.
      • If the query_id isn't in the queries table at all → skip (the
        FK constraint would reject the insert; warn the caller).
      • If the body is blank after trimming → skip (probably an empty
        section heading without content).

    Idempotent: re-running with the same file overwrites the same rows.
    """
    text = file_path.read_text()
    parsed = parse_answers_markdown(text)

    tier_name = next(
        (t.router_alias for t in models.tiers if t.level == tier_level),
        f"tier{tier_level}",
    )

    result = ImportResult(
        parsed=len(parsed),
        inserted=0,
        updated=0,
        skipped_empty=[],
        skipped_unknown=[],
    )
    now = datetime.now(UTC)

    with session_scope(db_path) as session:
        # Pre-load the set of valid query_ids so we can warn cleanly
        # rather than tripping a FK error mid-transaction.
        from .db import Query
        valid_qids: set[str] = {
            qid for (qid,) in session.execute(select(Query.query_id)).all()
        }

        for qid, body in parsed:
            if not body:
                result.skipped_empty.append(qid)
                continue
            if qid not in valid_qids:
                result.skipped_unknown.append(qid)
                continue

            existing = session.execute(
                select(TierAnswer)
                .where(TierAnswer.run_id == run_id)
                .where(TierAnswer.query_id == qid)
                .where(TierAnswer.tier_level == tier_level)
            ).scalar_one_or_none()

            if existing is not None:
                existing.response_text = body
                existing.tier_name = tier_name
                existing.status = "success"
                existing.error_msg = None
                existing.attempted_at = now
                result.updated += 1
            else:
                session.add(
                    TierAnswer(
                        run_id=run_id,
                        query_id=qid,
                        tier_level=tier_level,
                        tier_name=tier_name,
                        response_text=body,
                        status="success",
                        attempted_at=now,
                    )
                )
                result.inserted += 1

    return result
