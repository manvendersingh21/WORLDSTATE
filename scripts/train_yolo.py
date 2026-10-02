#!/usr/bin/env python
"""Train the WORLDSTATE detector from simulator ground-truth boxes.

The simulator labels exist only to supervise and score this detector. The world
model downstream never sees them: it consumes unlabeled tracks from
`worldstate.perception.yolo_track`.

Only `split == "train"` episodes (the reference normals) are used, and a few of
them are reserved as a validation split. Held-out normals and every miss stay
unseen until the tracking evaluation at the end.

    scripts/train_yolo.py --epochs 24

writes `models/worldstate-yolo.pt` and `models/yolo_metrics.json`.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worldstate.config import PROCESS_DIR, PROCESS_MANIFEST, YOLO_WEIGHTS  # noqa: E402

CLASS_NAMES = ["gripper", "part", "fixture"]
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
YOLO_DIR = ROOT / "data" / "yolo"
RUNS_DIR = ROOT / "runs" / "yolo"
METRICS_PATH = ROOT / "models" / "yolo_metrics.json"


def manifest() -> dict:
    return json.loads(PROCESS_MANIFEST.read_text())


def labels_for(episode_id: str) -> dict:
    return json.loads((PROCESS_DIR / "labels" / f"{episode_id}.json").read_text())


def to_yolo_line(name: str, box: list[int], width: int, height: int) -> str | None:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    if w <= 0 or h <= 0:
        return None
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
    w, h = min(w, 1.0), min(h, 1.0)
    return f"{CLASS_ID[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def export_episode(episode_id: str, split: str, stride: int) -> int:
    """Write every `stride`-th frame of one episode as a YOLO image/label pair."""
    video = PROCESS_DIR / "videos" / f"{episode_id}.mp4"
    lab = labels_for(episode_id)
    width, height = int(lab["width"]), int(lab["height"])
    frames = lab["frames"]
    img_dir = YOLO_DIR / "images" / split
    lbl_dir = YOLO_DIR / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video}")
    written = 0
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0 and index < len(frames):
            lines = [
                line
                for name in CLASS_NAMES
                if (line := to_yolo_line(name, frames[index]["boxes"][name], width, height))
            ]
            if lines:
                stem = f"{episode_id}_{index:04d}"
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                written += 1
        index += 1
    cap.release()
    if index != len(frames):
        print(f"  {episode_id}: {index} decoded frames vs {len(frames)} labelled")
    return written


def build_dataset(train_eps: list[str], val_eps: list[str], stride: int, val_stride: int) -> tuple[Path, int, int]:
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    n_train = sum(export_episode(eid, "train", stride) for eid in train_eps)
    n_val = sum(export_episode(eid, "val", val_stride) for eid in val_eps)
    yaml_path = YOLO_DIR / "dataset.yaml"
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        f"path: {YOLO_DIR}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )
    return yaml_path, n_train, n_val


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


AUGMENT = {
    # The misses put the cube 4-6 cm from anything in training, so translation
    # and scale matter more than usual for a fixed-camera scene.
    "hsv_h": 0.015,
    "hsv_s": 0.5,
    "hsv_v": 0.35,
    "degrees": 0.0,
    "translate": 0.18,
    "scale": 0.45,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "erasing": 0.0,
}


def train(yaml_path: Path, args: argparse.Namespace, device: str):
    from ultralytics import YOLO

    model = YOLO(args.base)
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        deterministic=True,
        device=device,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        workers=args.workers,
        cache="ram",
        patience=args.epochs,
        plots=False,
        val=True,
        verbose=False,
        **AUGMENT,
    )
    return model, results


def val_metrics(weights: Path, yaml_path: Path, args: argparse.Namespace, device: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    res = model.val(
        data=str(yaml_path),
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        plots=False,
        verbose=False,
        project=str(RUNS_DIR),
        name=f"{args.name}-val",
        exist_ok=True,
    )
    box = res.box
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        where = np.where(box.ap_class_index == i)[0]
        if len(where) == 0:
            per_class[name] = {"map50": 0.0, "precision": 0.0, "recall": 0.0}
            continue
        j = int(where[0])
        per_class[name] = {
            "map50": round(float(box.ap50[j]), 4),
            "precision": round(float(box.p[j]), 4),
            "recall": round(float(box.r[j]), 4),
        }
    return {
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": per_class,
    }


def track_episode(episode_id: str) -> dict:
    """Score one episode through the production path: worldstate.perception.yolo_track."""
    from worldstate.perception import yolo_track

    video = PROCESS_DIR / "videos" / f"{episode_id}.mp4"
    tracks = yolo_track(video)
    lab = labels_for(episode_id)
    width, height = float(lab["width"]), float(lab["height"])
    gt = lab["frames"]
    frames = tracks["frames"]
    hits = {name: 0 for name in CLASS_NAMES}
    errors: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}
    ious: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}
    for i, frame in enumerate(frames):
        if i >= len(gt):
            break
        found = {obj["label"]: obj for obj in frame["objects"]}
        for name in CLASS_NAMES:
            obj = found.get(name)
            if obj is None:
                continue
            hits[name] += 1
            x1, y1, x2, y2 = gt[i]["boxes"][name]
            gcx, gcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            pcx, pcy = obj["cx"] * width, obj["cy"] * height
            errors[name].append(float(np.hypot(pcx - gcx, pcy - gcy) / width))
            pw, ph = obj["w"] * width, obj["h"] * height
            px1, py1, px2, py2 = pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2
            iw = max(0.0, min(px2, x2) - max(px1, x1))
            ih = max(0.0, min(py2, y2) - max(py1, y1))
            inter = iw * ih
            union = pw * ph + (x2 - x1) * (y2 - y1) - inter
            ious[name].append(inter / union if union > 0 else 0.0)
    n = min(len(frames), len(gt))
    return {
        "frames": n,
        "per_class": {
            name: {
                "detection_rate": round(hits[name] / n, 4) if n else 0.0,
                "median_center_error_w": round(statistics.median(errors[name]), 4) if errors[name] else None,
                "median_iou": round(statistics.median(ious[name]), 4) if ious[name] else None,
            }
            for name in CLASS_NAMES
        },
        "_raw": {"hits": hits, "errors": errors, "ious": ious, "frames": n},
    }


def tracking_metrics(test_eps: list[str]) -> dict:
    episodes = {}
    pooled_hits = {name: 0 for name in CLASS_NAMES}
    pooled_errors: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}
    pooled_ious: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}
    pooled_frames = 0
    for eid in test_eps:
        result = track_episode(eid)
        raw = result.pop("_raw")
        pooled_frames += raw["frames"]
        for name in CLASS_NAMES:
            pooled_hits[name] += raw["hits"][name]
            pooled_errors[name].extend(raw["errors"][name])
            pooled_ious[name].extend(raw["ious"][name])
        episodes[eid] = result
        rates = {k: v["detection_rate"] for k, v in result["per_class"].items()}
        print(f"  {eid}: {rates}")
    overall = {
        name: {
            "detection_rate": round(pooled_hits[name] / pooled_frames, 4) if pooled_frames else 0.0,
            "median_center_error_w": round(statistics.median(pooled_errors[name]), 4)
            if pooled_errors[name]
            else None,
            "median_iou": round(statistics.median(pooled_ious[name]), 4) if pooled_ious[name] else None,
        }
        for name in CLASS_NAMES
    }
    return {
        "note": (
            "Measured through worldstate.perception.yolo_track on split==test episodes, "
            "which were never trained on. detection_rate is the fraction of frames whose "
            "track output carries that label; median_center_error_w is the median distance "
            "between predicted and ground-truth box centers in units of image width."
        ),
        "episodes": episodes,
        "overall": overall,
        "total_frames": pooled_frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base", default="yolov8n.pt", help="base weights to fine-tune")
    parser.add_argument("--stride", type=int, default=3, help="keep every Nth training frame")
    parser.add_argument("--val-stride", type=int, default=3, help="keep every Nth validation frame")
    parser.add_argument("--val-episodes", type=int, default=3, help="train-split episodes reserved for val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="worldstate")
    parser.add_argument("--skip-train", action="store_true", help="reuse existing weights, redo metrics only")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    man = manifest()
    episodes = man["episodes"]
    train_split = sorted(e["id"] for e in episodes if e["split"] == "train")
    test_split = sorted(e["id"] for e in episodes if e["split"] == "test")
    if len(train_split) <= args.val_episodes:
        raise SystemExit("not enough train-split episodes")
    val_eps = train_split[-args.val_episodes :]
    train_eps = train_split[: -args.val_episodes]
    print(f"train episodes: {train_eps}")
    print(f"val episodes:   {val_eps}")
    print(f"test episodes (never trained on): {test_split}")

    device = pick_device(args.device)
    started = time.time()
    yaml_path, n_train, n_val = build_dataset(train_eps, val_eps, args.stride, args.val_stride)
    print(f"dataset: {n_train} train / {n_val} val images at {yaml_path}")

    if args.skip_train:
        if not YOLO_WEIGHTS.exists():
            raise SystemExit(f"--skip-train needs {YOLO_WEIGHTS}")
        best = YOLO_WEIGHTS
    else:
        _, _ = train(yaml_path, args, device)
        best = RUNS_DIR / args.name / "weights" / "best.pt"
        if not best.exists():
            raise SystemExit(f"training produced no weights at {best}")
        YOLO_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best, YOLO_WEIGHTS)
        best = YOLO_WEIGHTS
    train_seconds = round(time.time() - started, 1)

    from ultralytics import YOLO

    names = {int(k): v for k, v in YOLO(str(YOLO_WEIGHTS)).names.items()}
    expected = dict(enumerate(CLASS_NAMES))
    if names != expected:
        raise SystemExit(f"class names {names} must be exactly {expected}")
    size_mb = round(YOLO_WEIGHTS.stat().st_size / 1e6, 2)
    print(f"weights: {YOLO_WEIGHTS} ({size_mb} MB) names={names}")

    print("validating...")
    val = val_metrics(YOLO_WEIGHTS, yaml_path, args, device)
    print(f"  mAP50={val['map50']} mAP50-95={val['map50_95']}")

    print("tracking test episodes through worldstate.perception.yolo_track...")
    tracking = tracking_metrics(test_split)

    metrics = {
        "train_config": {
            "base_weights": args.base,
            "model": "yolov8n",
            "imgsz": args.imgsz,
            "epochs": args.epochs,
            "batch": args.batch,
            "seed": args.seed,
            "device": device,
            "frame_stride": args.stride,
            "val_frame_stride": args.val_stride,
            "class_names": CLASS_NAMES,
            "train_episodes": train_eps,
            "val_episodes": val_eps,
            "train_images": n_train,
            "val_images": n_val,
            "augment": AUGMENT,
            "train_seconds": train_seconds,
            "weights_mb": size_mb,
            "label_source": "simulator ground truth, detector supervision only",
        },
        "val": val,
        "test_tracking": tracking,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {METRICS_PATH}")
    overall = {k: v["detection_rate"] for k, v in tracking["overall"].items()}
    print(f"test detection rate: {overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
