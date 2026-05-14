"""Diagnostic — per-signal scores for the misroute queries.

Backs `make scores`. For each query the router under-routed in the latest
run, invoke `vllm-sr eval --json --prompt <text>` to get back the actual
signal scores, then show the per-signal confidences and (when available)
the projected `request_difficulty` and matched `tier_band`.

Why subprocess and not HTTP: vllm-sr's CLI `eval` command is documented
and stable across versions; the underlying `/api/v1/eval` HTTP endpoint
returned 500 on our first attempt with `{"prompt": ...}`, suggesting the
request shape is different than guessed. The CLI is the supported entry
point and abstracts the wire format from us.

Schema notes (v0.3 projections design):
  • `signal_values` contains `embedding:<signal_name>` keys for each
    complexity signal (e.g. `embedding:needs_reasoning`). The signal's
    own threshold is configured in router-exemplars.yaml.
  • Under projections, the routing decision is driven by which BAND the
    `request_difficulty` score lands in — not by which signal "wins".
    matched_signals is therefore less informative than it used to be.
  • If the eval response carries the projected score and/or matched
    band, we surface them. If not, we still show the underlying signal
    confidences so an operator can reason about the score.

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

# Single per-signal threshold — matches `threshold:` in router-exemplars.yaml.
# Under the projections design every complexity signal shares this; the
# decision is driven by the projected band, not by per-signal gating.
SIGNAL_THRESHOLD = 0.55


@dataclass
class SignalScore:
    name: str                  # e.g. "needs_reasoning"
    score: float               # confidence in [0, 1] from vllm-sr
    threshold: float           # config threshold for this signal

    @property
    def gap(self) -> float:
        """Positive when score is above our threshold, negative when below."""
        return self.score - self.threshold

    @property
    def above_threshold(self) -> bool:
        return self.score >= self.threshold


@dataclass
class EvalSnapshot:
    """Everything we extracted from one vllm-sr eval response."""
    signals: list[SignalScore]
    request_difficulty: float | None      # projected score, if surfaced
    tier_band: str | None                  # matched band name, if surfaced


def parse_eval_response(data: Any) -> EvalSnapshot:
    """Extract per-signal scores and (best-effort) projection outputs.

    `signal_values` carries the underlying complexity signal confidences:
      {
        "embedding:needs_reasoning": 0.43,
        "embedding:needs_reasoning:best": ...,    # sub-keys — skipped
        "embedding:needs_reasoning:support": ...,
        "embedding:needs_reasoning:prototype_count": 8,
        "embedding:needs_expertise": 0.52,
        "embedding:needs_judgment": 0.61,
        ...
      }
    The projected `request_difficulty` and matched `tier_band` may appear
    under various keys depending on vllm-sr version; we look for them
    defensively and tolerate absence.
    """
    empty = EvalSnapshot(signals=[], request_difficulty=None, tier_band=None)
    if not isinstance(data, dict):
        return empty

    signal_values = data.get("signal_values")
    if not isinstance(signal_values, dict):
        return empty

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

    signals = [
        SignalScore(name=name, score=score, threshold=SIGNAL_THRESHOLD)
        for name, score in scores.items()
    ]

    # Best-effort projection extraction. Try a handful of plausible keys —
    # if the schema differs in this vllm-sr build, leave None and the
    # caller renders signals alone.
    rd = _coerce_float(signal_values.get("projection:request_difficulty"))
    if rd is None:
        rd = _coerce_float(signal_values.get("request_difficulty"))
    if rd is None:
        proj = data.get("projections") if isinstance(data.get("projections"), dict) else None
        if proj:
            rd = _coerce_float(proj.get("request_difficulty"))

    tier_band: str | None = None
    band_raw = signal_values.get("mapping:tier_band")
    if isinstance(band_raw, str):
        tier_band = band_raw
    elif isinstance(data.get("mappings"), dict):
        mb = data["mappings"].get("tier_band")
        if isinstance(mb, str):
            tier_band = mb

    return EvalSnapshot(signals=signals, request_difficulty=rd, tier_band=tier_band)


def _coerce_float(v: Any) -> float | None:
    if isinstance(v, int | float):
        return float(v)
    return None


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
    return (
        f"    {above_marker} {s.name:<{width}}  score={s.score:.3f}  "
        f"threshold={s.threshold:.3f}  gap={sign}{abs(s.gap):.3f}"
    ).rstrip()


def _diagnose(snap: EvalSnapshot, expected_tier: int) -> str:
    """One-line summary of why this query landed below expected_tier."""
    if snap.request_difficulty is not None and snap.tier_band:
        return (
            f"DIAGNOSIS: request_difficulty={snap.request_difficulty:.3f} → "
            f"{snap.tier_band}; expected ≥tier{expected_tier}."
        )
    above = [s for s in snap.signals if s.above_threshold]
    if not snap.signals:
        return ""
    if not above:
        return (
            "DIAGNOSIS: no complexity signal above threshold — "
            "exemplar gap on the hard side, not a cutoff issue."
        )
    return (
        f"DIAGNOSIS: {len(above)}/{len(snap.signals)} signal(s) above threshold; "
        f"projected difficulty was still below the cutoff for tier{expected_tier}."
    )


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
    if not snap.signals and snap.request_difficulty is None:
        lines.append("    (no signal scores parsed; see raw output)")
        return "\n".join(lines)

    if snap.request_difficulty is not None:
        band = snap.tier_band or "?"
        lines.append(
            f"    request_difficulty={snap.request_difficulty:.3f}   band={band}"
        )

    # Order: closest to flipping (smallest negative gap) first.
    signals_sorted = sorted(snap.signals, key=lambda s: -s.gap)
    for s in signals_sorted:
        lines.append(_format_signal_line(s))

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

        snap = parse_eval_response(data)
        lines.append(render_query_scores(
            m.query_id, m.expected_min_tier, m.routed_tier, m.prompt, snap,
        ))
        lines.append("")

        # If the first response yielded nothing parseable, dump it raw so we
        # can see the actual schema and adapt the parser.
        if not snap.signals and snap.request_difficulty is None and not raw_dump_printed:
            lines.append("Raw response for first misroute (schema discovery):")
            lines.append(json.dumps(data, indent=2)[:2000])
            lines.append("")
            raw_dump_printed = True

    return "\n".join(lines)
