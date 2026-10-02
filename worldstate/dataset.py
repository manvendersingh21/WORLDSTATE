"""Load the repeated-process dataset, the synthetic cell, or a manifest of real mp4s."""

from __future__ import annotations

import json
import os
from pathlib import Path

from worldstate.config import MANIFEST_PATH, PROCESS_MANIFEST, VIDEO_DIR


def process_manifest_path() -> Path | None:
    """A repeated-process manifest: WORLDSTATE_REAL_MANIFEST in the episodes schema, else data/process."""
    real = real_manifest_path()
    if real is not None and "episodes" in json.loads(real.read_text()):
        return real
    return PROCESS_MANIFEST if PROCESS_MANIFEST.exists() else None


def process_episodes(path: Path | None = None) -> list[dict]:
    """One fixed camera, same pick-and-place every run (robosuite Lift by default)."""
    path = path or process_manifest_path()
    manifest = json.loads(path.read_text())
    videos = path.parent / "videos" if (path.parent / "videos").is_dir() else path.parent
    episodes = []
    for ep in manifest["episodes"]:
        episodes.append(
            {
                "id": ep["id"],
                "role": ep["role"],
                "split": ep["split"],
                "kind": ep["kind"],
                "file": ep["file"],
                "video": str(videos / ep["file"]),
                "listed": ep["role"] != "similar_hidden",
                "dataset": manifest.get("generator", "process"),
                "failure_type": "displacement" if ep["kind"].startswith("miss") else "none",
            }
        )
    return episodes


def synthetic_episodes() -> list[dict]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    episodes = []
    for ep in manifest["episodes"]:
        episodes.append(
            {
                "id": ep["id"],
                "role": ep["role"],
                "split": ep["split"],
                "kind": ep["kind"],
                "file": ep["file"],
                "video": str(VIDEO_DIR / ep["file"]),
                "listed": ep["role"] != "similar_hidden",
                "dataset": "synthetic",
                "failure_type": "displacement" if ep["kind"].startswith("miss") else "none",
            }
        )
    return episodes


def real_manifest_path() -> Path | None:
    raw = os.environ.get("WORLDSTATE_REAL_MANIFEST", "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


def real_episodes(manifest_path: Path | None = None) -> list[dict]:
    path = manifest_path or real_manifest_path()
    if path is None:
        return []
    manifest = json.loads(path.read_text())
    if "clips" not in manifest:
        return []
    root = path.parent
    successes = [clip for clip in manifest["clips"] if clip["label"] == "success"]
    failures = [clip for clip in manifest["clips"] if clip["label"] == "failure"]
    episodes = []
    for index, clip in enumerate(successes):
        split = "train" if index < 16 else "test"
        episodes.append(_real_episode(clip, root, split, "reference" if split == "train" else "heldout_normal"))
    for index, clip in enumerate(failures):
        if index == 0:
            role = "unseen_failure"
        elif index == 1:
            role = "similar_hidden"
        else:
            role = "eval_failure"
        episodes.append(_real_episode(clip, root, "test", role))
    return episodes


def _real_episode(clip: dict, root: Path, split: str, role: str) -> dict:
    video = root / clip["path"]
    return {
        "id": clip["episode_id"],
        "role": role,
        "split": split,
        "kind": clip.get("failure_type") or clip["label"],
        "file": Path(clip["path"]).name,
        "video": str(video),
        "listed": role != "similar_hidden",
        "dataset": "exylos-front",
        "failure_type": clip.get("failure_type", "none"),
        "label": clip["label"],
    }


def episodes_from_folder(folder: Path, split: str = "external") -> list[dict]:
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi"})
    episodes = []
    for video in videos:
        episodes.append(
            {
                "id": video.stem,
                "role": "external",
                "split": split,
                "kind": "external",
                "file": video.name,
                "video": str(video),
                "listed": True,
                "dataset": "folder",
                "failure_type": "unknown",
            }
        )
    return episodes
