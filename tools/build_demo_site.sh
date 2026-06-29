#!/usr/bin/env bash
# Build the path-mounted static demo for Cloudflare Workers (Static Assets).
#
# Three things, so a redeploy never needs a manual cache purge:
#   1. NEST demo/ -> dist/demos/semantic-router/ so asset keys mirror the mount
#      path: a request for /demos/semantic-router/demo.css matches a real key.
#   2. FINGERPRINT demo.css / demo.js / demo_data.json with a content hash and
#      rewrite every reference. Cloudflare serves all of these
#      `max-age=0, must-revalidate` yet still edge-HITs stale copies under a reused
#      URL after a redeploy (the symptom that forced a manual purge). A content
#      change becomes a NEW URL the edge has never seen, so it's always fresh.
#   3. Emit dist/_headers (parsed by Workers, not served): the fingerprinted assets
#      cache immutable/forever; the one unhashable URL — index.html — is `no-store`,
#      so the entry point is always re-fetched and always references the latest
#      hashes. Brand images in assets/ are left unhashed (they're stable).
#
# Also injects <base href="/demos/semantic-router/"> into the BUILT index.html only;
# source demo/ stays root-relative so local `make demo` still serves at :8000.
#
# Runs in Cloudflare Workers Builds (Ubuntu: bash/coreutils/sed/openssl present)
# before `npx wrangler deploy`, and locally for verification.
set -euo pipefail

MOUNT="demos/semantic-router"
DIST="dist"
OUT="${DIST}/${MOUNT}"

# First 8 hex of the file's SHA-256 (openssl is present on macOS + Ubuntu).
hash8() { openssl dgst -sha256 "$1" | awk '{print $NF}' | cut -c1-8; }

rm -rf "$DIST"
mkdir -p "$(dirname "$OUT")"
cp -R demo "$OUT"

# Strip build-only / junk files (pricing.json is build-time only).
rm -f "$OUT/pricing.json"
find "$OUT" -name '.DS_Store' -delete

# 1) Fingerprint the data file and rewrite its fetch() in demo.js BEFORE demo.js
#    is itself hashed, so the js hash reflects the rewritten reference.
DATA_HASH="$(hash8 "$OUT/data/demo_data.json")"
mv "$OUT/data/demo_data.json" "$OUT/data/demo_data.${DATA_HASH}.json"
sed -i.bak "s#'data/demo_data.json'#'data/demo_data.${DATA_HASH}.json'#" "$OUT/demo.js"
rm -f "$OUT/demo.js.bak"

# 2) Fingerprint css + js.
CSS_HASH="$(hash8 "$OUT/demo.css")"; mv "$OUT/demo.css" "$OUT/demo.${CSS_HASH}.css"
JS_HASH="$(hash8 "$OUT/demo.js")";   mv "$OUT/demo.js"  "$OUT/demo.${JS_HASH}.js"

# 3) Rewrite index.html: inject <base>, point at the fingerprinted css/js.
sed -i.bak \
  -e "s#<head>#<head><base href=\"/${MOUNT}/\">#" \
  -e "s#href=\"demo.css\"#href=\"demo.${CSS_HASH}.css\"#" \
  -e "s#src=\"demo.js\"#src=\"demo.${JS_HASH}.js\"#" \
  "$OUT/index.html"
rm -f "$OUT/index.html.bak"

# 4) Cache policy. Fingerprinted assets are immutable (safe to cache forever — the
#    URL changes when content changes); the entry point is never stored.
cat > "${DIST}/_headers" <<EOF
/${MOUNT}/demo.*
  Cache-Control: public, max-age=31536000, immutable
/${MOUNT}/data/*
  Cache-Control: public, max-age=31536000, immutable
/${MOUNT}/
  Cache-Control: no-store
/${MOUNT}/index.html
  Cache-Control: no-store
EOF

echo "Built ${OUT} (fingerprinted):"
find "$OUT" -type f | sort | sed "s#^#  #"
echo "Wrote ${DIST}/_headers:"
sed "s#^#  #" "${DIST}/_headers"
