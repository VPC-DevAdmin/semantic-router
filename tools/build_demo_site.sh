#!/usr/bin/env bash
# Build the path-mounted static site for Cloudflare Workers (Static Assets).
#
# Cloudflare's path-mount rule has two halves and this satisfies both:
#   1. NEST the output so asset keys mirror the mount path:
#        demo/  ->  dist/demos/semantic-router/   (wrangler assets.directory = "dist")
#      A request for /demos/semantic-router/demo.css then matches a real asset key.
#   2. PREFIX every relative ref with the mount path, via an injected <base> tag,
#      so URLs resolve correctly regardless of trailing slash.
#
# The source demo/ stays root-relative (no <base>), so local `make demo` keeps
# working at http://localhost:8000/ — the prefix is only added to the built copy.
#
# Runs in Cloudflare Workers Builds (Ubuntu, bash+coreutils+sed present) before
# `npx wrangler deploy`, and locally for verification.
set -euo pipefail

MOUNT="demos/semantic-router"
OUT="dist/${MOUNT}"

rm -rf dist
mkdir -p "$(dirname "$OUT")"
cp -R demo "$OUT"

# Strip build-only and junk files from the deploy (pricing.json is build-time
# only; demo.js fetches just data/demo_data.json at runtime).
rm -f "$OUT/pricing.json"
find "$OUT" -name '.DS_Store' -delete

# Prefix every relative asset URL with the mount path.
sed -i.bak "s#<head>#<head><base href=\"/${MOUNT}/\">#" "$OUT/index.html"
rm -f "$OUT/index.html.bak"

echo "Built ${OUT}:"
find "$OUT" -type f | sort | sed "s#^#  #"
