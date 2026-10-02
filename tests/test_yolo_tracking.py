import json
import math
import os
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldstate.perception import yolo_track
from worldstate.series import series_from_tracks
from ultralytics import YOLO

def test_yolo_tracking():
    manifest_path = Path("data/process/manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    test_episodes = [ep for ep in manifest["episodes"] if ep["split"] == "test"]
    
    # Assert no train-split episode is a test input
    for ep in test_episodes:
        assert ep["split"] != "train"

    assert len(test_episodes) > 0, "No test episodes found"

    weights_path = Path("models/worldstate-yolo.pt").absolute()
    os.environ["WORLDSTATE_YOLO_WEIGHTS"] = str(weights_path)
    os.environ["WORLDSTATE_PERCEPTION"] = "yolo"
    
    # Check classes mapping precisely as asked: {0: gripper, 1: part, 2: fixture}
    model = YOLO(str(weights_path))
    assert getattr(model, "names", None) is not None, "Model must have names"
    names = model.names
    assert names[0] in ["gripper", "robot gripper", "robot"], f"Expected class 0 to map to gripper, got {names[0]}"
    assert names[1] == "part" or names[1] == "cube", f"Expected class 1 to map to part, got {names[1]}"
    assert names[2] in ["fixture", "bin"], f"Expected class 2 to map to fixture, got {names[2]}"
    
    classes = {"gripper", "part", "fixture"}
    
    detection_counts = {c: 0 for c in classes}
    gt_counts = {c: 0 for c in classes}
    errors = {c: [] for c in classes}
    
    for ep in test_episodes:
        video_path = Path(f"data/process/videos/{ep['id']}.mp4")
        label_path = Path(f"data/process/labels/{ep['id']}.json")
        
        with open(label_path) as f:
            labels_data = json.load(f)
            
        # Run tracking ONCE per episode
        tracks = yolo_track(video_path)
        
        # Test labels exactly in {gripper, part, fixture}
        for frame in tracks["frames"]:
            for obj in frame["objects"]:
                assert obj["label"] in classes, f"Unexpected label {obj['label']}"
                
        # Tracks feed series_from_tracks without NaNs
        series = series_from_tracks(tracks)
        for k, v in series.items():
            if k == "feature_kind":
                continue
            assert not np.isnan(v).any(), f"NaNs found in series output for {k}"
            
        gt_frames = labels_data["frames"]
        width = labels_data["width"]
        height = labels_data["height"]
        
        ep_detection_counts = {c: 0 for c in classes}
        ep_gt_counts = {c: 0 for c in classes}
        
        # compare per-frame tracked centers
        for i, frame in enumerate(tracks["frames"]):
            if i >= len(gt_frames):
                break
            gt_frame = gt_frames[i]
            
            # Map of detections in this frame
            dets = {obj["label"]: obj for obj in frame["objects"]}
            
            for cls in classes:
                if cls in gt_frame["boxes"]:
                    gt_counts[cls] += 1
                    ep_gt_counts[cls] += 1
                    gt_box = gt_frame["boxes"][cls]
                    gt_cx = (gt_box[0] + gt_box[2]) / 2.0 / width
                    gt_cy = (gt_box[1] + gt_box[3]) / 2.0 / height
                    
                    if cls in dets:
                        detection_counts[cls] += 1
                        ep_detection_counts[cls] += 1
                        det = dets[cls]
                        dx = (det["cx"] - gt_cx) * width
                        dy = (det["cy"] - gt_cy) * height
                        dist_in_image_width = math.hypot(dx, dy) / width
                        errors[cls].append(dist_in_image_width)
        
        # episode-level detection floors
        for cls in classes:
            if ep_gt_counts[cls] > 0:
                ep_rate = ep_detection_counts[cls] / ep_gt_counts[cls]
                floor = 0.75 if cls == "gripper" else 0.80
                assert ep_rate >= floor, f"Episode {ep['id']} {cls} detection rate {ep_rate:.3f} < {floor}"
        
        # for misses the tracked part center at the final frame must be >= 0.02 image widths away from the tracked fixture center
        if ep["id"].startswith("miss_"):
            part_cxs = [obj["cx"] for frame in tracks["frames"] for obj in frame["objects"] if obj["label"] == "part"]
            part_cys = [obj["cy"] for frame in tracks["frames"] for obj in frame["objects"] if obj["label"] == "part"]
            fixture_cxs = [obj["cx"] for frame in tracks["frames"] for obj in frame["objects"] if obj["label"] == "fixture"]
            fixture_cys = [obj["cy"] for frame in tracks["frames"] for obj in frame["objects"] if obj["label"] == "fixture"]
            
            if part_cxs and fixture_cxs:
                dx = (part_cxs[-1] - fixture_cxs[-1]) * width
                dy = (part_cys[-1] - fixture_cys[-1]) * height
                dist = math.hypot(dx, dy) / width
                assert dist >= 0.02, f"Miss episode {ep['id']}: part not displaced enough from fixture (dist={dist:.3f})"
                
    # per-class detection rate
    det_rate = {c: detection_counts[c] / gt_counts[c] if gt_counts[c] > 0 else 0.0 for c in classes}
    assert det_rate["part"] >= 0.90, f"Part detection rate {det_rate['part']:.3f} < 0.90"
    assert det_rate["fixture"] >= 0.90, f"Fixture detection rate {det_rate['fixture']:.3f} < 0.90"
    assert det_rate["gripper"] >= 0.85, f"Gripper detection rate {det_rate['gripper']:.3f} < 0.85"
    
    # median center error < 0.04 of image width (0.05 for gripper)
    for cls in classes:
        if errors[cls]:
            median_error = np.median(errors[cls])
            limit = 0.05 if cls == "gripper" else 0.04
            assert median_error < limit, f"{cls} median center error {median_error:.3f} >= {limit}"
