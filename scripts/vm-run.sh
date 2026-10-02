#!/usr/bin/env bash
# Run WORLDSTATE on the VAST workshop VM. Reads the team config the VM already has
# (/config/<team>.config) and maps it to WORLDSTATE's environment; nothing is written to disk.
#   bash scripts/vm-run.sh            # serves on 0.0.0.0:${PORT:-8000}
set -eu
cd "$(dirname "$0")/.."

CONFIG="$(ls /config/*.config 2>/dev/null | head -1 || true)"
if [ -n "$CONFIG" ]; then
  set -a; . "$CONFIG"; set +a
fi
# Event Cosmos3 endpoint (CoreWeave GPU) and team bearer token.
export COSMOS_BASE_URL="${COSMOS_BASE_URL:-${COSMOS3_REASON_URL:-http://166.19.38.112:8001}}"
export COSMOS_API_KEY="${COSMOS_API_KEY:-${GPU_BEARER_TOKEN:-}}"
# Inside the VM the VastDB endpoint resolves, so the optional mirror can run here.
export VAST_ACCESS_KEY="${VAST_ACCESS_KEY:-${ACCESS_KEY:-}}"
export VAST_SECRET_KEY="${VAST_SECRET_KEY:-${SECRET_KEY:-}}"
export WORLDSTATE_DATASET="${WORLDSTATE_DATASET:-process}"
export WORLDSTATE_PERCEPTION="${WORLDSTATE_PERCEPTION:-yolo}"
PORT="${PORT:-8000}"

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker build --build-arg WITH_YOLO=1 -t worldstate .
  docker rm -f worldstate >/dev/null 2>&1 || true
  ENV_ARGS=()
  for v in COSMOS_BASE_URL COSMOS_API_KEY VDB_ENDPOINT S3_ENDPOINT VAST_ACCESS_KEY VAST_SECRET_KEY VASTDB_BUCKET WORLDSTATE_DATASET WORLDSTATE_PERCEPTION; do
    [ -n "${!v:-}" ] && ENV_ARGS+=(-e "$v")
  done
  docker run -d --name worldstate -p "0.0.0.0:${PORT}:${PORT}" -e PORT="$PORT" "${ENV_ARGS[@]}" worldstate
  echo "Learning the graph at boot (a few minutes)…"
  until curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null; do sleep 5; done
else
  PY="$(command -v python3.12 || command -v python3)"
  [ -x .venv/bin/python ] || "$PY" -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
  .venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install -q -r requirements-yolo.txt
  [ -f data/model/world_model.json ] || .venv/bin/python -m worldstate.learn
  nohup .venv/bin/python -m uvicorn worldstate.server:app --host 0.0.0.0 --port "$PORT" > /tmp/worldstate.log 2>&1 &
  until curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null; do sleep 5; done
fi
echo "WORLDSTATE on the VM: http://127.0.0.1:${PORT}"
