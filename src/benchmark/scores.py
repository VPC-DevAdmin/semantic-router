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


# Per-signal thresholds we set in router-exemplars.yaml. The eval response
# doesn't echo these back, so we hardcode here. Update if the YAML changes.
HARD_THRESHOLD = 0.42
EASY_THRESHOLD = 0.50


@dataclass
class SignalScore:
    name: str                  # e.g. "reasoning_hard"
    score: float               # raw embedding similarity from vllm-sr
    threshold: float           # our config threshold for this signal
    matched_by_router: bool    # whether vllm-sr's matched_signals lists this signal

    @property
    def gap(self) -> float:
        """Positive when score is above our threshold, negative when below.

        Note: this is "would match if threshold were the only criterion."
        vllm-sr appears to use a single-winner policy in practice — see
        `matched_by_router` for the runtime's verdict.
        """
        return self.score - self.threshold

    @property
    def above_threshold(self) -> bool:
        return self.score >= self.threshold


def _threshold_for(signal_name: str) -> float:
    return HARD_THRESHOLD if signal_name.endswith("_hard") else EASY_THRESHOLD


def parse_eval_response(data: Any) -> list[SignalScore]:
    """Extract per-signal scores from vllm-sr's eval JSON.

    Real shape (vllm-sr v0.3, observed):
      {
        "signal_values": {
          "embedding:reasoning_hard": 0.436,
          "embedding:reasoning_hard:best": 0.437,
          "embedding:reasoning_hard:support": 0.435,
          "embedding:reasoning_hard:prototype_count": 8,
          "embedding:reasoning_easy": 0.573,
          ... (one main + 3 sub-keys per signal)
        },
        "matched_signals": {"embeddings": ["expertise_easy"]},
        "decision_result": {...},
        ...
      }

    We pull the main score (the unsuffixed `embedding:<name>` key) for each
    signal, look up our config threshold, and note whether vllm-sr's
    matched_signals lists it.
    """
    if not isinstance(data, dict):
        return []

    signal_values = data.get("signal_values")
    if not isinstance(signal_values, dict):
        return []

    # Build "main score per signal" map. Skip sub-keys (`:best`, `:support`,
    # `:prototype_count`) which appear alongside the main score.
    scores: dict[str, float] = {}
    for key, val in signal_values.items():
        if not (isinstance(key, str) and key.startswith("embedding:")):
            continue
        if not isinstance(val, int | float):
            continue
        suffix = key[len("embedding:"):]
        if ":" in suffix:
            continue
        scores[suffix] = float(val)

    # Which signals does vllm-sr's runtime consider "matched"?
    matched_set: set[str] = set()
    matched_signals = data.get("matched_signals", {})
    if isinstance(matched_signals, dict):
        emb_matched = matched_signals.get("embeddings")
        if isinstance(emb_matched, list):
            matched_set = {str(s) for s in emb_matched}

    out: list[SignalScore] = []
    for name, score in scores.items():
        out.append(SignalScore(
            name=name,
            score=score,
            threshold=_threshold_for(name),
            matched_by_router=(name in matched_set),
        ))
    return out


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


def _format_signal_line(s: SignalScore, *, width: int = 20) -> str:
    sign = "+" if s.gap >= 0 else "-"
    above_marker = "✓" if s.above_threshold else "·"
    # Two pieces of info: whether the score is above our config threshold,
    # AND whether vllm-sr's runtime put this signal in matched_signals.
    # These often disagree — that's the point of showing both.
    router_tag = "ROUTER-MATCHED" if s.matched_by_router else ""
    return (
        f"    {above_marker} {s.name:<{width}}  score={s.score:.3f}  "
        f"threshold={s.threshold:.3f}  gap={sign}{abs(s.gap):.3f}  {router_tag}"
    ).rstrip()


def _diagnose(signals: list[SignalScore]) -> str:
    """One-line summary of where this misroute sits, threshold-wise vs router-wise."""
    above = [s for s in signals if s.above_threshold]
    matched = [s for s in signals if s.matched_by_router]
    hard_above = [s for s in above if s.name.endswith("_hard")]

    if not above:
        return "DIAGNOSIS: no signal above threshold — exemplar gap, not a threshold issue."
    if hard_above and not any(s.matched_by_router for s in hard_above):
        return (
            f"DIAGNOSIS: {len(hard_above)} hard signal(s) above threshold but vllm-sr "
            f"matched only {sorted({s.name for s in matched})} — single-winner semantics."
        )
    if matched and not any(s.name.endswith("_hard") for s in matched):
        return (
            "DIAGNOSIS: router matched an easy signal as winner; hard signals were "
            "outscored even though some are above threshold."
        )
    return ""


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

    diag = _diagnose(signals)
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
