# WORLDSTATE image: python:3.12-slim, CPU-only, multi-arch friendly.
# All configuration (API keys, model endpoints, dataset choice, PORT) comes
# from environment variables at run time. No secrets are baked into layers.

FROM python:3.12-slim

# 1 = install CPU-only torch + ultralytics (requirements-yolo.txt); 0 = skip.
ARG WITH_YOLO=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libglib2.0-0 is required by opencv (even headless); libgl1 by the
# ultralytics-pulled opencv-python build; ffmpeg encodes the synthetic
# dataset videos generated at boot (worldstate/synthetic.py).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# What enters the image is governed by .dockerignore (exclude-all with an
# explicit re-include list): worldstate/, web/, scripts/, data/process/,
# models/, requirements*.txt. data/process/ and models/ may be absent in the
# repo; the build must not fail when they are.
COPY requirements.txt requirements-yolo.txt /app/

RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$WITH_YOLO" = "1" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
        && pip install --no-cache-dir -r requirements-yolo.txt; \
    fi

COPY . /app/

# Non-root runtime; /app (and therefore data/ created at boot) is writable.
RUN useradd --uid 1000 --create-home worldstate \
 && mkdir -p /app/data \
 && chown -R worldstate:worldstate /app

USER worldstate

# Documentation only; the published port is ${PORT:-8000} at run time.
EXPOSE 8000

# Learned-at-boot entrypoint; see scripts/entrypoint.sh (owned by peer b):
# learns the world model if data/model/world_model.json is missing, then
# execs uvicorn on 0.0.0.0:${PORT:-8000}.
ENTRYPOINT ["sh", "/app/scripts/entrypoint.sh"]
