"""Synthetic top-down robot cell: normal pick-and-place and align-then-miss runs.

Videos are drawn with OpenCV and encoded H.264 so the browser can play them.
A manifest records roles (reference, held-out normal, unseen failure, similar).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from worldstate.config import (
    DURATION,
    FPS,
    GENERATOR_VERSION,
    HEIGHT,
    MANIFEST_PATH,
    VIDEO_DIR,
    WIDTH,
)

BIN_X = 0.78
BIN_Y = 0.705


def _ease(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def _params(kind: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    if kind == "miss_left":
        base_mx = 0.595
    elif kind.startswith("miss"):
        base_mx = 0.925
    else:
        base_mx = BIN_X
    return {
        "sx": 0.30 + float(rng.uniform(-0.012, 0.012)),
        "sy": 0.695 + float(rng.uniform(-0.008, 0.008)),
        "hy": 0.36 + float(rng.uniform(-0.01, 0.01)),
        "time_scale": float(rng.uniform(0.985, 1.015)),
        "mx": base_mx + float(rng.uniform(-0.01, 0.01)),
        "my": 0.735 + float(rng.uniform(-0.008, 0.008)),
        "frame_rng": np.random.default_rng(seed + 17),
        "kind": kind,
    }


def pose_at(t: float, p: dict) -> tuple[float, float, float, float, float]:
    """Return gripper x/y, object x/y, and finger openness in 0..1."""
    t = float(np.clip(t * p["time_scale"], 0.0, DURATION))
    sx, sy, hy = p["sx"], p["sy"], p["hy"]
    bx, by = BIN_X, BIN_Y
    kind = p["kind"]

    def object_normal(t_s: float) -> tuple[float, float]:
        if t_s < 1.70:
            return sx, sy
        if t_s < 2.55:
            u = _ease((t_s - 1.70) / 0.85)
            return sx, _lerp(sy, hy + 0.02, u)
        if t_s < 4.35:
            u = _ease((t_s - 2.55) / 1.80)
            return _lerp(sx, bx, u), _lerp(hy + 0.02, hy, u)
        if t_s < 5.05:
            u = _ease((t_s - 4.35) / 0.70)
            return bx, _lerp(hy, hy + 0.05, u)
        if t_s < 6.15:
            u = _ease((t_s - 5.05) / 1.10)
            return bx, _lerp(hy + 0.05, by, u)
        return bx, by

    if kind == "normal" or t <= 5.05:
        ox, oy = object_normal(t)
        slipped = False
    else:
        u = _ease(min(1.0, (t - 5.05) / 1.15))
        ox = _lerp(bx, p["mx"], u)
        oy = _lerp(hy + 0.05, p["my"], u)
        slipped = True

    if t < 1.15:
        u = _ease(t / 1.15)
        gx = _lerp(0.14, sx, u)
        gy = _lerp(0.30, sy - 0.055, u)
        opened = 1.0
    elif t < 1.70:
        u = _ease((t - 1.15) / 0.55)
        gx, gy = sx, _lerp(sy - 0.055, sy - 0.03, u)
        opened = _lerp(1.0, 0.0, u)
    elif not slipped and t < 6.15:
        gx, gy = ox, oy - 0.032
        opened = 0.0
    elif slipped and t < 5.75:
        gu = _ease((t - 5.05) / 0.70)
        gx = bx
        gy = _lerp(hy + 0.05, by, gu) - 0.032
        opened = 0.0
    elif slipped and t < 6.30:
        gx, gy = bx, by - 0.03
        opened = _ease((t - 5.75) / 0.45)
    elif t < 6.55 and not slipped:
        gx, gy = ox, oy - 0.032
        opened = _ease((t - 6.15) / 0.40)
    else:
        start_t = 6.30 if slipped else 6.55
        u = _ease((t - start_t) / max(0.2, DURATION - start_t))
        gx = _lerp(bx, 0.48, u)
        gy = _lerp(by - 0.03, 0.26, u)
        opened = 1.0

    noise = p["frame_rng"].normal(0.0, 0.0012, size=4)
    gx, gy, ox, oy = (gx + noise[0], gy + noise[1], ox + noise[2], oy + noise[3])
    return gx, gy, ox, oy, float(np.clip(opened, 0.0, 1.0))


def _arm(gx: int, gy: int, w: int, h: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    shoulder = (int(0.16 * w), int(0.90 * h))
    wrist = (gx, gy)
    dx = wrist[0] - shoulder[0]
    dy = wrist[1] - shoulder[1]
    dist = math.hypot(dx, dy) or 1.0
    link = max(dist * 0.55, 160.0)
    reach = min(dist, 2 * link - 1.0)
    mx = shoulder[0] + dx * (reach / dist) * 0.5
    my = shoulder[1] + dy * (reach / dist) * 0.5
    nx, ny = -dy / dist, dx / dist
    lift = math.sqrt(max(link * link - (reach * 0.5) ** 2, 0.0))
    elbow_a = (int(mx + nx * lift), int(my + ny * lift))
    elbow_b = (int(mx - nx * lift), int(my - ny * lift))
    elbow = elbow_a if elbow_a[1] < elbow_b[1] else elbow_b
    return shoulder, elbow, wrist


def render_frame(t: float, p: dict, w: int = WIDTH, h: int = HEIGHT) -> np.ndarray:
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:] = (186, 182, 174)
    cv2.rectangle(frame, (36, int(0.56 * h)), (w - 36, h - 24), (112, 120, 116), -1)
    for x in range(80, w - 40, 48):
        cv2.line(frame, (x, int(0.58 * h)), (x, h - 36), (98, 106, 102), 1)
    cv2.rectangle(frame, (48, int(0.84 * h)), (210, h - 28), (72, 74, 76), -1)
    cv2.putText(frame, "CELL 3", (56, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 68), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"t={t:05.2f}s",
        (56, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 80, 78),
        1,
        cv2.LINE_AA,
    )

    gx, gy, ox, oy, opened = pose_at(t, p)
    gxi, gyi = int(gx * w), int(gy * h)
    oxi, oyi = int(ox * w), int(oy * h)

    bw, bh = 156, 82
    bcx, bcy = int(BIN_X * w), int(BIN_Y * h)
    x0, y0 = bcx - bw // 2, bcy - bh // 2
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (40, 145, 52), -1)
    cv2.rectangle(frame, (x0 - 4, y0 - 4), (x0 + bw + 4, y0 + bh + 4), (28, 96, 36), 4)

    shoulder, elbow, wrist = _arm(gxi, gyi, w, h)
    cv2.line(frame, shoulder, elbow, (68, 70, 74), 16, cv2.LINE_AA)
    cv2.line(frame, elbow, wrist, (86, 88, 92), 11, cv2.LINE_AA)
    cv2.circle(frame, shoulder, 18, (58, 60, 64), -1, cv2.LINE_AA)

    ow = 46
    cv2.rectangle(frame, (oxi - ow // 2, oyi - ow // 2), (oxi + ow // 2, oyi + ow // 2), (230, 55, 25), -1)
    cv2.rectangle(frame, (oxi - ow // 2, oyi - ow // 2), (oxi + ow // 2, oyi + ow // 2), (255, 110, 70), 2)

    gap = int(7 + opened * 16)
    finger_w, finger_h = 11, 34
    red = (36, 36, 225)
    palm_top = gyi - finger_h // 2 - 14
    palm_bot = gyi - finger_h // 2 + 2
    cv2.rectangle(frame, (gxi - gap - finger_w, palm_top), (gxi + gap + finger_w, palm_bot), red, -1)
    cv2.rectangle(
        frame,
        (gxi - gap - finger_w, gyi - finger_h // 2),
        (gxi - gap, gyi + finger_h // 2),
        red,
        -1,
    )
    cv2.rectangle(
        frame,
        (gxi + gap, gyi - finger_h // 2),
        (gxi + gap + finger_w, gyi + finger_h // 2),
        red,
        -1,
    )
    return frame


def _encode(path: Path, n_frames: int, draw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for i in range(n_frames):
            frame = draw(i / FPS)
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
    err = proc.stderr.read().decode() if proc.stderr else ""
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {err}")
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if count < int(n_frames * 0.9):
        raise RuntimeError(f"{path} has {count} frames, expected {n_frames}")


def episode_specs() -> list[dict]:
    specs = []
    for i in range(20):
        specs.append(
            {
                "id": f"normal_{i:02d}",
                "file": f"normal_{i:02d}.mp4",
                "kind": "normal",
                "seed": i,
                "role": "reference" if i < 16 else "heldout_normal",
                "split": "train" if i < 16 else "test",
            }
        )
    specs.extend(
        [
            {
                "id": "miss_unseen",
                "file": "miss_unseen.mp4",
                "kind": "miss_right",
                "seed": 100,
                "role": "unseen_failure",
                "split": "test",
            },
            {
                "id": "miss_eval_right",
                "file": "miss_eval_right.mp4",
                "kind": "miss_right",
                "seed": 101,
                "role": "eval_failure",
                "split": "test",
            },
            {
                "id": "miss_eval_left",
                "file": "miss_eval_left.mp4",
                "kind": "miss_left",
                "seed": 102,
                "role": "eval_failure",
                "split": "test",
            },
            {
                "id": "miss_similar",
                "file": "miss_similar.mp4",
                "kind": "miss_right",
                "seed": 103,
                "role": "similar_hidden",
                "split": "test",
            },
        ]
    )
    return specs


def dataset_ready() -> bool:
    if not MANIFEST_PATH.exists():
        return False
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest.get("generator_version") != GENERATOR_VERSION:
        return False
    for ep in manifest.get("episodes", []):
        if not (VIDEO_DIR / ep["file"]).exists():
            return False
    return True


def generate_dataset(force: bool = False) -> dict:
    if dataset_ready() and not force:
        return json.loads(MANIFEST_PATH.read_text())
    specs = episode_specs()
    n_frames = int(round(DURATION * FPS))
    for spec in specs:
        params = _params(spec["kind"], spec["seed"])
        path = VIDEO_DIR / spec["file"]
        print(f"rendering {spec['id']} ({spec['kind']})")
        _encode(path, n_frames, lambda t, params=params: render_frame(t, params))
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "fps": FPS,
        "width": WIDTH,
        "height": HEIGHT,
        "duration": DURATION,
        "episodes": specs,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(specs)} videos to {VIDEO_DIR}")
    return manifest


def ensure_dataset() -> dict:
    return generate_dataset(force=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic WORLDSTATE cell videos")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate_dataset(force=args.force)


if __name__ == "__main__":
    main()
