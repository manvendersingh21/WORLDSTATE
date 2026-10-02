"""Paths and tunable constants for the offline demo."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VIDEO_DIR = DATA / "videos"
MODEL_DIR = DATA / "model"
TRACKS_DIR = MODEL_DIR / "tracks"
UPLOAD_DIR = DATA / "uploads"
WEB_DIR = ROOT / "web"

MANIFEST_PATH = VIDEO_DIR / "manifest.json"
PROCESS_DIR = DATA / "process"
PROCESS_MANIFEST = PROCESS_DIR / "manifest.json"
YOLO_WEIGHTS = ROOT / "models" / "worldstate-yolo.pt"
MODEL_PATH = MODEL_DIR / "world_model.json"
MEMORY_DB = MODEL_DIR / "memory.db"
VECTOR_PATH = MODEL_DIR / "text_vectors.npz"

GENERATOR_VERSION = 1
PERCEPTION_VERSION = 2

FPS = 15
WIDTH = 960
HEIGHT = 540
DURATION = 8.0

WINDOW = 8
STRIDE = 4
TRAJ_SAMPLES = 48

for _path in (VIDEO_DIR, MODEL_DIR, TRACKS_DIR, UPLOAD_DIR, WEB_DIR):
    _path.mkdir(parents=True, exist_ok=True)


def env_flag(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()
