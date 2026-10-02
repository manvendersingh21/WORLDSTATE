#!/usr/bin/env bash
# Publish the running WORLDSTATE container through a Cloudflare quick tunnel.
# Finds the port Docker actually published, checks /api/health locally, then keeps
# cloudflared (and the Mac) awake and prints the public trycloudflare.com URL.
set -eu

NAME="${WORLDSTATE_CONTAINER:-worldstate}"
LOG="${TMPDIR:-/tmp}/worldstate-tunnel.log"

PORT="$(docker port "$NAME" 2>/dev/null | awk -F: '/->/ {print $NF; exit}')"
PORT="${PORT:-${PORT_OVERRIDE:-8000}}"
if ! curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null; then
  echo "WORLDSTATE is not answering on 127.0.0.1:${PORT}; run scripts/docker-run.sh first." >&2
  exit 1
fi
command -v cloudflared >/dev/null || { echo "install cloudflared: brew install cloudflared" >&2; exit 1; }

pkill -f "cloudflared tunnel --url http://localhost:${PORT}" 2>/dev/null || true
: > "$LOG"
# caffeinate -i keeps the Mac from idle-sleeping for as long as cloudflared runs.
nohup caffeinate -i cloudflared tunnel --no-autoupdate --url "http://localhost:${PORT}" >>"$LOG" 2>&1 &

for _ in $(seq 1 60); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -1 || true)"
  [ -n "$URL" ] && break
  sleep 1
done
if [ -z "${URL:-}" ]; then
  echo "No tunnel URL yet; see $LOG" >&2
  exit 1
fi
echo "Local:  http://127.0.0.1:${PORT}"
echo "Public: ${URL}"
echo "Log:    ${LOG}  (stop with: pkill -f 'cloudflared tunnel')"
