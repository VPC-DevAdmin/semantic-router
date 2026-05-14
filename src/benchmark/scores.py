"""Diagnostic — per-signal scores for the misroute queries.

Backs `make scores`. For each query the router under-routed in the latest
run, invoke `vllm-sr eval --json --prompt <text>` to get back the actual
signal scores and thresholds, then print the gap. Answers the question:
"how close are the misroutes to flipping the right way?"

Why subprocess and not HTTP: vllm-sr's CLI `eval` command is documented
and stable across versions; the underlying `/api/v1/eval` HTTP endpoint
returned 500 on our first attempt with `{"prompt": ...}`, suggesting the
request shape is different than guessed. The CLI is the supported entry
point and abstracts the wire format from us.

If parsing the eval JSON fails, the tool dumps the raw output for the
first misroute so we can iterate on the parser.
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


@dataclass
class SignalScore:
    name: str         # e.g. "reasoning_hard"
    score: float
    threshold: float
    matched: bool

    @property
    def gap(self) -> float:
        """Positive when above threshold (matched), negative when below (missed)."""
        return self.score - self.threshold


def _coerce_score(raw: Any) -> SignalScore | None:
    """Tolerate several possible per-signal JSON shapes.

    Known/guessed shapes we try:
      {name, score, threshold, matched}
      {signal, confidence, threshold, ...}
      {name, value, threshold, matched}
      "embedding:reasoning_hard: 0.41 / threshold 0.42"  (string form, unlikely but safe)
    """
    if not isinstance(raw, dict):
        return None
    name = raw.get("name") or raw.get("signal") or raw.get("rule")
    score = raw.get("score") or raw.get("confidence") or raw.get("value")
    threshold = raw.get("threshold")
    matched = raw.get("matched")
    if name is None or score is None or threshold is None:
        return None
    return SignalScore(
        name=str(name),
        score=float(score),
        threshold=float(threshold),
        matched=bool(matched) if matched is not None else (float(score) >= float(threshold)),
    )


def parse_eval_response(data: Any) -> list[SignalScore]:
    """Walk the response looking for signal-score-shaped objects.

    The upstream schema may put scores at any of several keys. We look at
    a few common ones, then walk anything list-shaped that we can coerce.
    """
    if not isinstance(data, dict):
        return []

    candidates: list[Any] = []
    for key in ("signal_confidences", "signals", "rules", "rule_evaluations", "confidences"):
        val = data.get(key)
        if isinstance(val, list):
            candidates.extend(val)
        elif isinstance(val, dict):
            # Maybe nested: {"signals": {"used": [...], "matched": [...]}}
            for sub in val.values():
                if isinstance(sub, list):
                    candidates.extend(sub)

    out: list[SignalScore] = []
    for c in candidates:
        coerced = _coerce_score(c)
        if coerced is not None:
            out.append(coerced)

    # Deduplicate by name (some shapes may repeat in matched + used lists).
    seen: set[str] = set()
    unique: list[SignalScore] = []
    for s in out:
        if s.name in seen:
            continue
        seen.add(s.name)
        unique.append(s)
    return unique


async def fetch_eval(
    prompt: str,
    apiserver: str = DEFAULT_APISERVER,
    *,
    timeout: float = EVAL_TIMEOUT_S,
    binary: str = VLLM_SR_BINARY,
) -> dict:
    """Invoke `vllm-sr eval --json --prompt <text>` and return the parsed JSON.

    Uses subprocess rather than direct HTTP so we stay on the supported CLI
    surface. Slightly slower per call (~1s startup overhead) but reliable
    across vllm-sr versions.
    """
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


def _format_signal_line(s: SignalScore, *, width: int = 24) -> str:
    sign = "+" if s.gap >= 0 else "-"
    marker = "✓" if s.matched else "·"
    return (
        f"    {marker} {s.name:<{width}}  score={s.score:.3f}  "
        f"threshold={s.threshold:.3f}  gap={sign}{abs(s.gap):.3f}"
    )


def render_query_scores(
    query_id: str,
    expected_tier: int,
    routed_tier: int | None,
    prompt: str,
    signals: list[SignalScore],
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
    if not signals:
        lines.append("    (no signal scores parsed; see raw output)")
        return "\n".join(lines)

    # Order: closest to flipping (smallest negative gap) first, so a quick
    # scan shows which signal would tip the rule with the smallest threshold
    # adjustment.
    signals_sorted = sorted(signals, key=lambda s: -s.gap)
    for s in signals_sorted:
        lines.append(_format_signal_line(s))
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

    lines = [f"Per-signal scores for {len(misroutes)} misroute(s):\n"]

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

        signals = parse_eval_response(data)
        lines.append(render_query_scores(
            m.query_id, m.expected_min_tier, m.routed_tier, m.prompt, signals,
        ))
        lines.append("")

        # If the first response yielded no signals, dump it raw so we can
        # see the actual schema and adapt the parser.
        if not signals and not raw_dump_printed:
            lines.append("Raw response for first misroute (schema discovery):")
            lines.append(json.dumps(data, indent=2)[:2000])
            lines.append("")
            raw_dump_printed = True

    return "\n".join(lines)
