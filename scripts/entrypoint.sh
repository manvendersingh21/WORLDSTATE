#!/bin/sh
set -eu

if [ -d "/app" ] && [ "$PWD" != "/app" ]; then
  cd /app
fi

if [ ! -f "data/model/world_model.json" ]; then
  echo "World model not found. Learning graph at boot..."
  python -m worldstate.learn
fi

if [ $# -gt 0 ]; then
  exec "$@"
fi

exec uvicorn worldstate.server:app --host 0.0.0.0 --port "${PORT:-8000}"
