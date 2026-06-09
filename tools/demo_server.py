#!/usr/bin/env python3
"""Static file server for the replay demo with caching disabled.

Identical to `python -m http.server` except every response carries
`Cache-Control: no-store`, so edits to demo.js / demo.css / the dataset /
the assets show up on a normal refresh instead of being masked by the
browser cache. Stdlib-only — no extra deps, matching the demo's
"works on a bare clone with just python3" guarantee.

Usage:
    python tools/demo_server.py [PORT] [--directory DIR]
"""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("port", nargs="?", type=int, default=8000)
    p.add_argument("--directory", default=".")
    args = p.parse_args()

    handler = partial(NoCacheHandler, directory=args.directory)
    httpd = ThreadingHTTPServer(("", args.port), handler)
    print(f"Serving {args.directory} at http://localhost:{args.port}/ (no-store)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
