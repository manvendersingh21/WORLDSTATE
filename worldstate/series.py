"""Turn tracks or pixels into per-frame series used by state discovery."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from worldstate.config import TRAJ_SAMPLES


def _fill(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if good.sum() == 0:
        return np.zeros(len(x))
    if good.sum() == 1:
        return np.full(len(x), float(x[good][0]))
    return np.interp(idx, idx[good], x[good])


def _velocity(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    if len(values) < 5:
        return np.zeros_like(values)
    vel = np.gradient(values, time)
    # Gradient is unstable on the first and last samples.
    vel[0] = vel[1] = vel[2]
    vel[-1] = vel[-2] = vel[-3]
    return vel


def _smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    if len(x) < k:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def series_from_tracks(tracks: dict) -> dict[str, np.ndarray]:
    frames = tracks["frames"]
    n = len(frames)
    fps = float(tracks.get("fps") or 15.0)
    width = float(tracks.get("width") or 1.0)
    height = float(tracks.get("height") or 1.0)

    def column(label: str, field: str) -> np.ndarray:
        out = np.full(n, np.nan)
        for i, frame in enumerate(frames):
            for obj in frame["objects"]:
                if obj["label"] == label:
                    out[i] = obj[field]
                    break
        return out

    grip_x, grip_y = _fill(column("gripper", "cx")), _fill(column("gripper", "cy"))
    obj_x, obj_y = _fill(column("part", "cx")), _fill(column("part", "cy"))
    bin_x_s, bin_y_s = column("fixture", "cx"), column("fixture", "cy")
    bin_x = float(np.nanmedian(bin_x_s)) if np.isfinite(bin_x_s).any() else 0.78
    bin_y = float(np.nanmedian(bin_y_s)) if np.isfinite(bin_y_s).any() else 0.70
    if not np.isfinite(column("gripper", "cx")).any():
        grip_x, grip_y = obj_x.copy(), obj_y.copy()

    grip_x, grip_y = _smooth(grip_x), _smooth(grip_y)
    obj_x, obj_y = _smooth(obj_x), _smooth(obj_y)
    time = np.array([frame["time"] for frame in frames], dtype=np.float64)
    if len(time) == 0:
        time = np.zeros(1)
    obj_vx = _velocity(obj_x, time)
    obj_vy = _velocity(obj_y, time)
    grip_vx = _velocity(grip_x, time)
    grip_vy = _velocity(grip_y, time)
    dist_go = np.hypot(obj_x - grip_x, obj_y - grip_y)
    obj_rx = obj_x - bin_x
    obj_ry = obj_y - bin_y
    dist_ob = np.hypot(obj_rx, obj_ry)
    lift = np.clip(bin_y - obj_y, 0.0, None)
    contact = (dist_go < 0.08).astype(np.float64)
    obj_speed = np.hypot(obj_vx, obj_vy)
    grip_speed = np.hypot(grip_vx, grip_vy)
    return {
        "time": time,
        "fps": np.array([fps]),
        "width": np.array([width]),
        "height": np.array([height]),
        "grip_x": grip_x,
        "grip_y": grip_y,
        "obj_x": obj_x,
        "obj_y": obj_y,
        "bin_x": np.array([bin_x]),
        "bin_y": np.array([bin_y]),
        "grip_rx": grip_x - bin_x,
        "grip_ry": grip_y - bin_y,
        "obj_rx": obj_rx,
        "obj_ry": obj_ry,
        "obj_vx": obj_vx,
        "obj_vy": obj_vy,
        "grip_speed": grip_speed,
        "dist_go": dist_go,
        "dist_ob": dist_ob,
        "contact": contact,
        "obj_speed": obj_speed,
        "lift": lift,
        "feature_kind": np.array(["kinematic"]),
    }


def series_from_video_flow(path: Path) -> dict[str, np.ndarray]:
    """Appearance-robust motion series for domain-randomized real clips."""
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    ok, prev = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"unreadable video {path}")
    prev_g = cv2.resize(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (160, 120))
    rows = []
    times = []
    index = 0
    step = 2
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if index % step:
            continue
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120))
        flow = cv2.calcOpticalFlowFarneback(prev_g, gray, None, 0.5, 2, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        threshold = max(float(np.percentile(mag, 75)), 0.35)
        selected = mag >= threshold
        if int(selected.sum()) < 15:
            cx = cy = 0.5
            vx = vy = 0.0
            mean_mag = float(mag.mean())
        else:
            ys, xs = np.mgrid[0 : mag.shape[0], 0 : mag.shape[1]]
            weight = mag * selected
            cx = float((weight * xs).sum() / weight.sum() / mag.shape[1])
            cy = float((weight * ys).sum() / weight.sum() / mag.shape[0])
            vx = float(flow[..., 0][selected].mean())
            vy = float(flow[..., 1][selected].mean())
            mean_mag = float(mag[selected].mean())
        rows.append((mean_mag, vx, vy, cx, cy))
        times.append(index / fps)
        prev_g = gray
    cap.release()
    arr = np.array(rows, dtype=np.float64)
    kernel = np.ones(9) / 9.0
    smoothed = np.vstack([np.convolve(arr[:, c], kernel, mode="same") for c in range(arr.shape[1])]).T
    time = np.array(times, dtype=np.float64)
    mag_s, vx_s, vy_s, cx, cy = smoothed.T
    return {
        "time": time,
        "fps": np.array([fps]),
        "mag": mag_s,
        "flow_vx": vx_s,
        "flow_vy": vy_s,
        "cx": cx,
        "cy": cy,
        "feature_kind": np.array(["flow"]),
    }


def resample_xy(time: np.ndarray, x: np.ndarray, y: np.ndarray, samples: int = TRAJ_SAMPLES) -> np.ndarray:
    if len(time) < 2:
        return np.zeros((samples, 2))
    grid = np.linspace(float(time[0]), float(time[-1]), samples)
    return np.stack([np.interp(grid, time, x), np.interp(grid, time, y)], axis=1)


KINEMATIC_FEATURES = [
    "grip_rx",
    "grip_ry",
    "obj_rx",
    "obj_ry",
    "obj_vx",
    "obj_vy",
    "grip_speed",
    "dist_go",
    "dist_ob",
    "contact",
    "obj_speed",
    "lift",
]

FLOW_FEATURES = ["mag", "flow_vx", "flow_vy", "cx", "cy"]


def window_table(series: dict, kind: str, window: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    names = KINEMATIC_FEATURES if kind == "kinematic" else FLOW_FEATURES
    n = len(series["time"])
    cols = [series[name] for name in names]
    base = np.stack(cols, axis=1)
    rows = []
    times = []
    if n < window:
        rows.append(base.mean(axis=0))
        times.append(float(series["time"][len(series["time"]) // 2]))
    else:
        for start in range(0, n - window + 1, stride):
            sl = base[start : start + window]
            if kind == "flow":
                delta = sl[-1] - sl[0]
                rows.append(np.concatenate([sl.mean(axis=0), delta]))
            else:
                rows.append(sl.mean(axis=0))
            times.append(float(series["time"][start + window // 2]))
    return np.asarray(rows, dtype=np.float64), np.asarray(times, dtype=np.float64)
