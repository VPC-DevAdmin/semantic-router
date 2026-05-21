#!/usr/bin/env python3
"""Rewrite `eval_id` in an evaluations JSON file to the canonical
`benchmark.evaluations._eval_id()` form.

Why this exists: the externally-produced `data/evaluations.json` uses
slightly different normalization rules than the harness's
`_eval_id()`. The import → export round-trip auto-canonicalizes on
its own, but if you want to update the committed file WITHOUT going
through the DB, this script does that in place.

Usage:
    .venv/bin/python tools/normalize_evaluations.py data/evaluations.json
    .venv/bin/python tools/normalize_evaluations.py data/evaluations.json --dry-run

Idempotent: re-running on an already-canonical file is a no-op.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `benchmark` importable without `pip install -e`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark.evaluations import _eval_id  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to evaluations.json")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the diff but don't modify the file.",
    )
    parser.add_argument(
        "--limit", type=int, default=5,
        help="Max number of before/after diffs to print (default 5).",
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"error: {args.path} does not exist", file=sys.stderr)
        return 2

    entries = json.loads(args.path.read_text())
    if not isinstance(entries, list):
        print(
            f"error: expected a JSON array, got {type(entries).__name__}",
            file=sys.stderr,
        )
        return 2

    changes: list[tuple[str, str]] = []
    for e in entries:
        canonical = _eval_id(
            e["query_id"],
            e.get("routed_provider"), e["routed_model"],
            e.get("expected_provider"), e["expected_model"],
            e["evaluator"],
        )
        if e["eval_id"] != canonical:
            changes.append((e["eval_id"], canonical))
            e["eval_id"] = canonical

    if changes:
        print(f"normalizing {len(changes)} of {len(entries)} eval_ids")
        for before, after in changes[: args.limit]:
            print(f"  - {before}")
            print(f"  + {after}")
        if len(changes) > args.limit:
            print(f"  … and {len(changes) - args.limit} more")
    else:
        print(f"all {len(entries)} eval_ids already canonical")

    if args.dry_run:
        if changes:
            print("\n(dry run — file NOT modified; drop --dry-run to write)")
        return 0

    if changes:
        args.path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
