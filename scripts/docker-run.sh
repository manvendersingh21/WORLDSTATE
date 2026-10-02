#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PORT="${PORT:-8000}"
BUILD=1

for arg in "$@"; do
  case "$arg" in
    --no-build)
      BUILD=0
      ;;
  esac
done

if [ "$BUILD" -eq 1 ]; then
  echo "Building image worldstate..."
  docker build --build-arg WITH_YOLO="${WITH_YOLO:-1}" -t worldstate "$REPO_ROOT"
fi

ENV_ARGS=("-e" "PORT=${PORT}")

if [ -f "$REPO_ROOT/.env" ]; then
  ENV_ARGS+=("--env-file" "$REPO_ROOT/.env")
fi

FORWARD_VARS=(
  NVIDIA_API_KEY
  NGC_API_KEY
  NV_API_KEY
  NVIDIA_COSMOS_MODEL
  COSMOS_BASE_URL
  NVIDIA_BASE_URL
  VAST_API_URL
  VAST_API_KEY
  COSMOS_PREDICT_URL
  WORLDSTATE_DATASET
  WORLDSTATE_PERCEPTION
  WORLDSTATE_YOLO_WEIGHTS
  WORLDSTATE_REAL_MANIFEST
  WORLDSTATE_TEXT_ENCODER
)

for var in "${FORWARD_VARS[@]}"; do
  if [ "${!var+set}" = "set" ]; then
    # Name only: docker reads the value from this environment, so it never appears in argv.
    export "$var"
    ENV_ARGS+=("-e" "$var")
  fi
done

echo "Starting container worldstate..."
docker rm -f worldstate >/dev/null 2>&1 || true

docker run -d \
  --name worldstate \
  --restart unless-stopped \
  -p "0.0.0.0:${PORT}:${PORT}" \
  "${ENV_ARGS[@]}" \
  worldstate

HEALTH_URL="http://127.0.0.1:${PORT}/api/health"
echo "Waiting for health check at $HEALTH_URL..."

max_wait=900
ready=0
for i in $(seq 1 "$max_wait"); do
  if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done

if [ "$ready" -ne 1 ]; then
  echo "Error: Timed out waiting for WORLDSTATE container to become healthy." >&2
  docker logs worldstate >&2 || true
  exit 1
fi

LAN_IP=""
if command -v ipconfig >/dev/null 2>&1; then
  LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || true)
  if [ -z "$LAN_IP" ]; then
    def_if=$(route -n get default 2>/dev/null | grep 'interface:' | awk '{print $2}' || true)
    if [ -n "$def_if" ]; then
      LAN_IP=$(ipconfig getifaddr "$def_if" 2>/dev/null || true)
    fi
  fi
elif command -v hostname >/dev/null 2>&1; then
  LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
fi

echo "WORLDSTATE is running!"
echo "Local URL: http://127.0.0.1:${PORT}"
if [ -n "$LAN_IP" ]; then
  echo "LAN URL:   http://${LAN_IP}:${PORT}"
fi
