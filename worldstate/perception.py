"""Perception adapters.

Default path: color blobs for the synthetic cell, otherwise motion blobs.
Optional path: Ultralytics YOLO / YOLO-World when installed and selected.
Both return the same trajectory schema.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

from worldstate.config import PERCEPTION_VERSION, TRACKS_DIR, YOLO_WEIGHTS

COLOR_RANGES = {
    "gripper": [
        ((0, 80, 70), (12, 255, 255)),
        ((168, 80, 70), (180, 255, 255)),
    ],
    "part": [((95, 70, 60), (135, 255, 255))],
    "fixture": [((35, 50, 40), (90, 255, 255))],
}
MIN_AREA = {"gripper": 180, "part": 180, "fixture": 400}


def _mask(hsv: np.ndarray, ranges: list[tuple]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for low, high in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(low), np.array(high)))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _largest(mask: np.ndarray, min_area: float) -> dict | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = min_area
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < best_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        best_area = area
        best = (area, x, y, w, h)
    if best is None:
        return None
    area, x, y, w, h = best
    return {
        "cx": x + w / 2.0,
        "cy": y + h / 2.0,
        "w": float(w),
        "h": float(h),
        "area": area,
    }


def _quality(frames: list[dict], labels: tuple[str, ...] = ("gripper", "part")) -> float:
    if not frames:
        return 0.0
    hits = 0
    for frame in frames:
        found = {obj["label"] for obj in frame["objects"]}
        if all(label in found for label in labels):
            hits += 1
    return hits / len(frames)


def _normalize(det: dict, width: int, height: int, label: str) -> dict:
    return {
        "label": label,
        "cx": det["cx"] / width,
        "cy": det["cy"] / height,
        "w": det["w"] / width,
        "h": det["h"] / height,
        "conf": float(min(1.0, det["area"] / 800.0)),
    }


def classical_color_track(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        objects = []
        for label, ranges in COLOR_RANGES.items():
            det = _largest(_mask(hsv, ranges), MIN_AREA[label])
            if det is not None:
                objects.append(_normalize(det, width, height, label))
        frames.append({"frame": index, "time": index / fps, "objects": objects})
        index += 1
    cap.release()
    duration = frames[-1]["time"] if frames else 0.0
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
        "perception": "classical-color",
        "perception_version": PERCEPTION_VERSION,
        "quality": _quality(frames),
        "frames": frames,
    }


def _link_blobs(raw: list[list[tuple]], width: int, height: int, fps: float) -> dict:
    """Greedy nearest-neighbor tracks for up to three moving blobs."""
    tracks: list[list[tuple[float, float, float, float] | None]] = []
    for blobs in raw:
        used = set()
        for track in tracks:
            prev = next((p for p in reversed(track) if p is not None), None)
            if prev is None or not blobs:
                track.append(None)
                continue
            best_i, best_d = None, 1e9
            for i, blob in enumerate(blobs):
                if i in used:
                    continue
                dist = math_hypot((blob[1] - prev[0]) / width, (blob[2] - prev[1]) / height)
                if dist < best_d:
                    best_d, best_i = dist, i
            if best_i is not None and best_d < 0.18:
                used.add(best_i)
                area, cx, cy, w, h = blobs[best_i]
                track.append((cx, cy, w, h, area))
            else:
                track.append(None)
        for i, blob in enumerate(blobs):
            if i in used or len(tracks) >= 3:
                continue
            area, cx, cy, w, h = blob
            tracks.append([None] * (len(tracks[0]) - 1) + [(cx, cy, w, h, area)] if tracks else [(cx, cy, w, h, area)])
            # The line above is hard to read when tracks was empty vs not.
            # Rebuild cleanly below if this gets messy — handled by the branch.
    # The initializer above is awkward for the first blob. Rebuild simply.
    return _link_blobs_clean(raw, width, height, fps)


def math_hypot(x: float, y: float) -> float:
    return float(np.hypot(x, y))


def _link_blobs_clean(raw: list[list[tuple]], width: int, height: int, fps: float) -> dict:
    series: list[list[tuple | None]] = []
    for blobs in raw:
        assigned = set()
        for track in series:
            prev = next((p for p in reversed(track) if p is not None), None)
            choice = None
            if prev is not None:
                best_d = 0.18
                for i, blob in enumerate(blobs):
                    if i in assigned:
                        continue
                    dist = math_hypot((blob[1] - prev[0]) / width, (blob[2] - prev[1]) / height)
                    if dist < best_d:
                        best_d = dist
                        choice = i
            if choice is None:
                track.append(None)
            else:
                assigned.add(choice)
                track.append(blobs[choice])
        for i, blob in enumerate(blobs):
            if i in assigned or len(series) >= 3:
                continue
            series.append([None] * (len(raw) - len(series[0]) if False else 0))
            # fix length: current frame index is len(track) of existing, i.e. len(series[0]) after appends
            # recompute properly
            series.pop()
            pad = len(series[0]) if series else 0
            # Wait, we popped the only chance. Let's do it simply:
            break
        else:
            continue
        break
    return _link_simple(raw, width, height, fps)


def _link_simple(raw: list[list[tuple]], width: int, height: int, fps: float) -> dict:
    series: list[list[tuple | None]] = []
    for frame_i, blobs in enumerate(raw):
        assigned: set[int] = set()
        for track in series:
            prev = next((p for p in reversed(track) if p is not None), None)
            choice = None
            best_d = 0.2
            if prev is not None:
                for i, blob in enumerate(blobs):
                    if i in assigned:
                        continue
                    dist = math_hypot((blob[1] - prev[0]) / width, (blob[2] - prev[1]) / height)
                    if dist < best_d:
                        best_d = dist
                        choice = i
            if choice is None:
                track.append(None)
            else:
                assigned.add(choice)
                track.append(blobs[choice])
        for i, blob in enumerate(blobs):
            if i in assigned or len(series) >= 3:
                continue
            series.append([None] * frame_i + [blob])
    # Label by motion variance: stillest -> fixture, largest travel -> part, other -> gripper.
    stats = []
    for track in series:
        pts = [(p[1] / width, p[2] / height) for p in track if p is not None]
        if len(pts) < 3:
            var = 0.0
        else:
            arr = np.array(pts)
            var = float(arr.var())
        stats.append(var)
    order = np.argsort(stats)
    labels = {}
    if len(order):
        labels[int(order[0])] = "fixture"
    if len(order) >= 2:
        labels[int(order[-1])] = "part"
    if len(order) >= 3:
        mid = [i for i in range(len(order)) if i not in (int(order[0]), int(order[-1]))]
        # order contains indices into series
        used = {labels[k] for k in labels}
        for idx in order:
            idx = int(idx)
            if idx not in labels:
                labels[idx] = "gripper" if "gripper" not in used else "part"
                used.add(labels[idx])
    frames = []
    n = len(raw)
    for frame_i in range(n):
        objects = []
        for t_i, track in enumerate(series):
            if frame_i >= len(track) or track[frame_i] is None:
                continue
            area, cx, cy, w, h = track[frame_i]
            det = {"cx": cx, "cy": cy, "w": w, "h": h, "area": area}
            objects.append(_normalize(det, width, height, labels.get(t_i, "part")))
        frames.append({"frame": frame_i, "time": frame_i / fps, "objects": objects})
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "duration": frames[-1]["time"] if frames else 0.0,
        "perception": "classical-motion",
        "perception_version": PERCEPTION_VERSION,
        "quality": _quality(frames, ("part",)),
        "frames": frames,
    }


def classical_motion_track(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    subtract = cv2.createBackgroundSubtractorMOG2(history=40, varThreshold=24, detectShadows=False)
    raw = []
    kernel = np.ones((3, 3), np.uint8)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fg = subtract.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 250:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            blobs.append((area, x + w / 2.0, y + h / 2.0, float(w), float(h)))
        blobs.sort(key=lambda b: b[0], reverse=True)
        raw.append(blobs[:4])
    cap.release()
    return _link_simple(raw, width, height, fps)


def yolo_available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except Exception:
        return False


def _map_yolo_label(name: str) -> str:
    text = name.lower()
    if any(key in text for key in ("grip", "hand", "arm", "robot")):
        return "gripper"
    if any(key in text for key in ("bin", "tray", "fixture", "container", "box")):
        return "fixture"
    return "part"


def yolo_track(path: Path) -> dict:
    from ultralytics import YOLO

    default = str(YOLO_WEIGHTS) if YOLO_WEIGHTS.exists() else "yolov8n.pt"
    weights = os.environ.get("WORLDSTATE_YOLO_WEIGHTS", "").strip() or default
    world = os.environ.get("WORLDSTATE_YOLO_WORLD", "0") == "1"
    model = YOLO(weights)
    if world and hasattr(model, "set_classes"):
        classes = os.environ.get(
            "WORLDSTATE_YOLO_CLASSES",
            "robot gripper,cube,bin",
        ).split(",")
        model.set_classes([c.strip() for c in classes if c.strip()])
    results = model.track(source=str(path), persist=True, verbose=False, stream=True)
    frames = []
    fps = 15.0
    width = height = 0
    for index, result in enumerate(results):
        height, width = result.orig_shape
        if result.boxes is None:
            frames.append({"frame": index, "time": index / fps, "objects": []})
            continue
        speed = getattr(result, "speed", None)
        names = result.names or {}
        chosen: dict[str, dict] = {}
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls_id in zip(xyxy, confs, clss):
            label = _map_yolo_label(str(names.get(int(cls_id), cls_id)))
            x1, y1, x2, y2 = box
            det = {
                "label": label,
                "cx": float((x1 + x2) / 2 / width),
                "cy": float((y1 + y2) / 2 / height),
                "w": float((x2 - x1) / width),
                "h": float((y2 - y1) / height),
                "conf": float(conf),
            }
            if label not in chosen or det["conf"] > chosen[label]["conf"]:
                chosen[label] = det
        frames.append({"frame": index, "time": index / fps, "objects": list(chosen.values())})
        del speed
    if frames:
        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
        cap.release()
        for i, frame in enumerate(frames):
            frame["time"] = i / fps
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "duration": frames[-1]["time"] if frames else 0.0,
        "perception": "yolo",
        "perception_version": PERCEPTION_VERSION,
        "weights": Path(weights).name,
        "quality": _quality(frames, ("part",)),
        "frames": frames,
    }


def perception_mode(dataset: str | None = None) -> str:
    """WORLDSTATE_PERCEPTION wins. The repeated process defaults to YOLO when it can run."""
    mode = os.environ.get("WORLDSTATE_PERCEPTION", "").strip()
    if mode:
        return mode
    if dataset not in (None, "synthetic") and yolo_available() and YOLO_WEIGHTS.exists():
        return "yolo"
    return "auto"


def perceive(path: Path, mode: str | None = None) -> dict:
    """Track actors. Auto mode keeps color tracks when they lock, else motion or YOLO."""
    mode = mode or os.environ.get("WORLDSTATE_PERCEPTION", "auto")
    if mode == "yolo":
        return yolo_track(path)
    if mode == "motion":
        return classical_motion_track(path)
    color = classical_color_track(path)
    if mode == "classical" or color["quality"] >= 0.65:
        return color
    if yolo_available():
        try:
            yolo = yolo_track(path)
            if yolo["quality"] >= color["quality"]:
                return yolo
        except Exception:
            pass
    motion = classical_motion_track(path)
    return motion if motion["quality"] > color["quality"] else color


def tracks_for(video: Path, episode_id: str, mode: str | None = None) -> dict:
    """Cached tracks, recomputed when the video, perception mode, or version changes."""
    mode = mode or os.environ.get("WORLDSTATE_PERCEPTION", "auto")
    path = TRACKS_DIR / f"{episode_id}.json"
    if path.exists() and video.exists() and path.stat().st_mtime >= video.stat().st_mtime:
        cached = json.loads(path.read_text())
        if (
            cached.get("perception_version") == PERCEPTION_VERSION
            and cached.get("source_video") == str(video)
            and cached.get("requested_mode") == mode
        ):
            return cached
    tracks = perceive(video, mode)
    tracks["episode_id"] = episode_id
    tracks["source_video"] = str(video)
    tracks["requested_mode"] = mode
    path.write_text(json.dumps(tracks))
    return tracks
