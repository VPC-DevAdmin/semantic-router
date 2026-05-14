"""Diagnostic — per-query routing trace for the misroute queries.

Backs `make scores`. For each query the router under-routed in the latest
run, invoke `vllm-sr eval --json --prompt <text>` to get back the actual
routing decision trace, then surface the most diagnostic facts:

  • What complexity level each signal matched at (none / medium / hard).
  • Which band the projected `request_difficulty` landed in.
  • Which decision fired.

Why subprocess and not HTTP: vllm-sr's CLI `eval` command is documented
and stable across versions; the underlying `/api/v1/eval` HTTP endpoint
returned 500 on our first attempt with `{"prompt": ...}`. The CLI is the
supported entry point and abstracts the wire format from us.

Schema notes (vllm-sr v0.3 projections design, observed shape):
  decision_result:
    decision_name: route_tier1
    matched_signals:
      complexity: [needs_reasoning:medium, needs_expertise:medium, ...]
      projection: [tier1_band]
    unmatched_signals:
      complexity: [needs_reasoning, ...]    # bare name = matched at none
      projection: [tier2_band, tier3_band, tier4_band, tier5_band]

A signal that appears in `matched_signals.complexity` as `<id>:<level>`
matched at that level. A signal that appears in `unmatched_signals.complexity`
as bare `<id>` did not match at any level on this query.

If parsing fails, the tool dumps the raw response for the first misroute
so we can iterate on the parser.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .misroutes import list_misroutes

DEFAULT_APISERVER = "http://localhost:8080"  # passed via --endpoint to the CLI
VLLM_SR_BINARY = "vllm-sr"
EVAL_TIMEOUT_S = 30.0

# Levels a complexity signal can match at, in ascending order of strength.
COMPLEXITY_LEVELS = ("none", "easy", "medium", "hard")


@dataclass
class ComplexityMatch:
    """A complexity signal's match outcome on one query."""
    name: str          # e.g. "needs_reasoning"
    level: str         # one of COMPLEXITY_LEVELS

    @property
    def is_match(self) -> bool:
        return self.level not in ("none", "easy")


@dataclass
class EvalSnapshot:
    """Everything we extracted from one vllm-sr eval response."""
    matches: list[ComplexityMatch]    # one per complexity signal seen
    decision: str | None              # e.g. "route_tier1"
    matched_band: str | None          # e.g. "tier1_band"


def parse_eval_response(data: Any) -> EvalSnapshot:
    """Pull the routing trace out of a vllm-sr eval response.

    The decision_result block carries the trace we want:
      • matched_signals.complexity[]   — entries like "<id>:<level>"
      • unmatched_signals.complexity[] — bare "<id>" = matched at none
      • matched_signals.projection[]   — one band name (the winner)
      • decision_name                  — e.g. "route_tier3"
    """
    empty = EvalSnapshot(matches=[], decision=None, matched_band=None)
    if not isinstance(data, dict):
        return empty

    decision_result = data.get("decision_result")
    if not isinstance(decision_result, dict):
        return empty

    decision = decision_result.get("decision_name")
    if not isinstance(decision, str):
        decision = None

    matched_signals = decision_result.get("matched_signals") or {}
    unmatched_signals = decision_result.get("unmatched_signals") or {}

    matches: dict[str, str] = {}

    # Matched complexity signals carry their level after a colon.
    for entry in _as_str_list(matched_signals.get("complexity")):
        name, _, level = entry.partition(":")
        if name and level in COMPLEXITY_LEVELS:
            matches[name] = level
        elif name and not level:
            # Defensive: bare name in matched_signals is unusual but
            # treat as "matched at some unspecified positive level".
            matches.setdefault(name, "medium")

    # Bare names in unmatched_signals = did not match at any level here.
    for entry in _as_str_list(unmatched_signals.get("complexity")):
        name, _, level = entry.partition(":")
        if name and not level:
            matches.setdefault(name, "none")

    # Projection: the matched band is the winner.
    matched_band: str | None = None
    proj_matched = _as_str_list(matched_signals.get("projection"))
    if proj_matched:
        matched_band = proj_matched[0]

    return EvalSnapshot(
        matches=[ComplexityMatch(name=n, level=lvl) for n, lvl in matches.items()],
        decision=decision,
        matched_band=matched_band,
    )


def _as_str_list(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    return [s for s in v if isinstance(s, str)]


async def fetch_eval(
    prompt: str,
    apiserver: str = DEFAULT_APISERVER,
    *,
    timeout: float = EVAL_TIMEOUT_S,
    binary: str = VLLM_SR_BINARY,
) -> dict:
    """Invoke `vllm-sr eval --json --prompt <text>` and return the parsed JSON."""
    if shutil.which(binary) is None:
        raise RuntimeError(
            f"{binary!r} not on PATH; install with `make setup` or set "
            f"VLLM_SR_BINARY env var"
        )

    proc = await asyncio.create_subprocess_exec(
        binary, "eval", "--json", "--prompt", prompt,
        "--endpoint", apiserver,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"{binary} eval timed out after {timeout}s") from None

    if proc.returncode != 0:
        stderr = stderr_b.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"{binary} eval exited {proc.returncode}: {stderr[:500]}"
        )

    stdout = stdout_b.decode("utf-8", errors="replace").strip()
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"{binary} eval output was not JSON: {stdout[:300]}"
        ) from e


# Visual marker per level — quick scan for "did any signal go hard?"
_LEVEL_MARK = {"none": "·", "easy": "·", "medium": "○", "hard": "●"}


def _format_match_line(m: ComplexityMatch, *, width: int = 20) -> str:
    mark = _LEVEL_MARK.get(m.level, "?")
    return f"    {mark} {m.name:<{width}}  {m.level}"


def _diagnose(snap: EvalSnapshot, expected_tier: int) -> str:
    """One-line summary of why this query landed where it did."""
    if snap.matched_band:
        n_hard = sum(1 for m in snap.matches if m.level == "hard")
        n_med = sum(1 for m in snap.matches if m.level == "medium")
        n_none = sum(1 for m in snap.matches if m.level in ("none", "easy"))
        return (
            f"DIAGNOSIS: {n_hard} hard / {n_med} medium / {n_none} none → "
            f"{snap.matched_band}; expected ≥tier{expected_tier}."
        )
    if not snap.matches:
        return ""
    n_hard = sum(1 for m in snap.matches if m.level == "hard")
    if n_hard == 0:
        return (
            "DIAGNOSIS: no signal matched at :hard — either the hard-side "
            "exemplars don't cover this query shape, or the signal threshold "
            "is set too high to clear."
        )
    return ""


def render_query_scores(
    query_id: str,
    expected_tier: int,
    routed_tier: int | None,
    prompt: str,
    snap: EvalSnapshot,
    *,
    max_prompt_chars: int = 80,
) -> str:
    """One stanza per query in the report."""
    import textwrap
    short_prompt = textwrap.shorten(prompt.replace("\n", " "), max_prompt_chars)
    lines = [
        f"  {query_id}  expected≥{expected_tier}  routed→{routed_tier or '?'}",
        f'    "{short_prompt}"',
    ]

    if not snap.matches and not snap.matched_band:
        lines.append("    (no routing trace parsed; see raw output)")
        return "\n".join(lines)

    if snap.matched_band or snap.decision:
        band = snap.matched_band or "?"
        decision = snap.decision or "?"
        lines.append(f"    band={band}   decision={decision}")

    # Order: strongest match first so the hard signals stand out.
    level_rank = {"hard": 0, "medium": 1, "easy": 2, "none": 3}
    matches_sorted = sorted(snap.matches, key=lambda m: (level_rank.get(m.level, 99), m.name))
    for m in matches_sorted:
        lines.append(_format_match_line(m))

    diag = _diagnose(snap, expected_tier)
    if diag:
        lines.append(f"    {diag}")
    return "\n".join(lines)


async def report_scores(
    db_path: Path,
    *,
    run_id: int | None = None,
    apiserver: str = DEFAULT_APISERVER,
) -> str:
    """Build the full report text for all misroutes in the latest run."""
    misroutes = list_misroutes(db_path, run_id=run_id)
    if not misroutes:
        return "No misroutes — nothing to score."

    lines = [f"Per-query routing trace for {len(misroutes)} misroute(s):\n"]

    raw_dump_printed = False
    for m in misroutes:
        try:
            data = await fetch_eval(m.prompt, apiserver=apiserver)
        except Exception as e:  # noqa: BLE001 — surface in report, keep going
            lines.append(
                f"  {m.query_id}  ERROR fetching eval: {type(e).__name__}: {e}"
            )
            lines.append("")
            continue

        snap = parse_eval_response(data)
        lines.append(render_query_scores(
            m.query_id, m.expected_min_tier, m.routed_tier, m.prompt, snap,
        ))
        lines.append("")

        # If the first response yielded nothing parseable, dump it raw so we
        # can see the actual schema and adapt the parser.
        if not snap.matches and not snap.matched_band and not raw_dump_printed:
            lines.append("Raw response for first misroute (schema discovery):")
            lines.append(json.dumps(data, indent=2)[:2000])
            lines.append("")
            raw_dump_printed = True

    return "\n".join(lines)
