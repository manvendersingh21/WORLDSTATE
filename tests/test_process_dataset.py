"""Independent verification of the generated data/process dataset.

Everything is recomputed from manifest.json, labels/*.json and the mp4 files;
summary.json is never trusted.
"""
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "process"
FPS, W, H, N = 20, 480, 360, 180
FFPROBE = "/opt/homebrew/bin/ffprobe"

if not (DATA / "manifest.json").exists():
    raise RuntimeError("data/process/manifest.json missing: run scripts/gen_process.py first")

MANIFEST = json.loads((DATA / "manifest.json").read_text())
EPS = MANIFEST["episodes"]
BY_ID = {e["id"]: e for e in EPS}


def labels(eid):
    return json.loads((DATA / "labels" / f"{eid}.json").read_text())


def xy_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def probe(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=codec_name,pix_fmt,width,height,nb_read_packets,r_frame_rate:format=duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def expected_episodes():
    exp = {}
    for i in range(20):
        exp[f"normal_{i:02d}"] = ("normal", "reference" if i < 16 else "heldout_normal",
                                  "train" if i < 16 else "test")
    exp["miss_unseen"] = ("miss_right", "unseen_failure", "test")
    exp["miss_eval_right"] = ("miss_right", "eval_failure", "test")
    exp["miss_eval_left"] = ("miss_left", "eval_failure", "test")
    exp["miss_similar"] = ("miss_right", "similar_hidden", "test")
    return exp


def test_manifest_schema():
    m = MANIFEST
    assert m["generator"] == "robosuite-lift"
    assert m["generator_version"] == 1
    assert (m["fps"], m["width"], m["height"]) == (FPS, W, H)
    assert m["camera"]
    assert len(EPS) == 24
    assert len(BY_ID) == 24, "duplicate ids"
    for e in EPS:
        assert set(e) == {"id", "file", "kind", "seed", "role", "split"}, e
        assert isinstance(e["seed"], int)
    exp = expected_episodes()
    assert set(BY_ID) == set(exp)
    for eid, (kind, role, split) in exp.items():
        e = BY_ID[eid]
        assert (e["kind"], e["role"], e["split"]) == (kind, role, split), eid
    assert len({e["seed"] for e in EPS}) == 24, "seeds should be unique"
    assert Counter(e["split"] for e in EPS) == {"train": 16, "test": 8}
    assert Counter(e["role"] for e in EPS) == {
        "reference": 16, "heldout_normal": 4, "unseen_failure": 1,
        "eval_failure": 2, "similar_hidden": 1}


@pytest.mark.parametrize("eid", sorted(expected_episodes()))
def test_video(eid):
    e = BY_ID[eid]
    path = DATA / "videos" / e["file"]
    assert path.exists()
    assert path.stat().st_size < 1.5 * 1024 * 1024, path.stat().st_size
    info = probe(path)
    s = info["streams"][0]
    assert s["codec_name"] == "h264"
    assert s["pix_fmt"] == "yuv420p"
    assert (s["width"], s["height"]) == (W, H)
    dur = float(info["format"]["duration"])
    assert 8.0 <= dur <= 10.0, dur
    lab = labels(eid)
    assert int(s["nb_read_packets"]) == len(lab["frames"])
    assert abs(len(lab["frames"]) / FPS - dur) < 0.1
    # faststart: moov atom before mdat
    head = path.read_bytes()[:65536]
    assert head.find(b"moov") != -1 and (head.find(b"mdat") == -1 or head.find(b"moov") < head.find(b"mdat"))


@pytest.mark.parametrize("eid", sorted(expected_episodes()))
def test_labels_schema(eid):
    lab = labels(eid)
    assert lab["id"] == eid
    assert (lab["fps"], lab["width"], lab["height"]) == (FPS, W, H)
    fr = lab["frames"]
    assert len(fr) >= 160
    for i, f in enumerate(fr):
        assert abs(f["t"] - i / FPS) < 1e-6
        assert set(f["boxes"]) >= {"gripper", "part", "fixture"}
        for name in ("gripper", "part", "fixture"):
            x1, y1, x2, y2 = f["boxes"][name]
            assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H, (i, name, f["boxes"][name])
        assert len(f["cube_pos"]) == 3 and len(f["eef_pos"]) == 3
        assert isinstance(f["gripper_closed"], bool)


def _ref():
    pad = MANIFEST.get("pad_center")
    table_z = MANIFEST.get("table_z")
    half = MANIFEST.get("cube_half_size", 0.0125)
    assert pad is not None and table_z is not None, "manifest must carry pad_center and table_z"
    return pad, table_z, half


def _rest_ok(z, table_z, half):
    return abs(z - (table_z + half)) <= 0.005


@pytest.mark.parametrize("eid", [f"normal_{i:02d}" for i in range(20)])
def test_normal_physical_success(eid):
    pad, table_z, half = _ref()
    fr = labels(eid)["frames"]
    z0 = fr[0]["cube_pos"][2]
    final = fr[-1]["cube_pos"]
    assert xy_dist(final, pad) <= 0.03, xy_dist(final, pad)
    assert _rest_ok(final[2], table_z, half), final
    assert max(f["cube_pos"][2] for f in fr) - z0 >= 0.04, "cube never lifted"
    assert any(f["gripper_closed"] for f in fr)
    assert not fr[-1]["gripper_closed"], "gripper should have opened"
    # pad box constant across frames (fixed fixture, fixed camera)
    assert all(f["boxes"]["fixture"] == fr[0]["boxes"]["fixture"] for f in fr)


def _max_step(fr):
    best, idx = 0.0, 0
    for i in range(1, len(fr)):
        d = xy_dist(fr[i]["cube_pos"], fr[i - 1]["cube_pos"])
        if d > best:
            best, idx = d, i
    return best, idx


@pytest.mark.parametrize("eid", ["miss_unseen", "miss_eval_right", "miss_eval_left", "miss_similar"])
def test_miss_physical_failure(eid):
    pad, table_z, half = _ref()
    kind = BY_ID[eid]["kind"]
    fr = labels(eid)["frames"]
    start, final = fr[0]["cube_pos"], fr[-1]["cube_pos"]
    step, idx = _max_step(fr)
    assert 0.04 <= step <= 0.06, f"teleport step {step}"
    dx = fr[idx]["cube_pos"][0] - fr[idx - 1]["cube_pos"][0]
    dy = fr[idx]["cube_pos"][1] - fr[idx - 1]["cube_pos"][1]
    assert abs(dx) < 0.01
    assert (dy > 0) if kind == "miss_right" else (dy < 0)
    # after the jump the cube stays put (never carried)
    assert max(f["cube_pos"][2] for f in fr) - start[2] < 0.01
    assert xy_dist(final, start) >= 0.04
    assert xy_dist(final, pad) > 0.04
    assert _rest_ok(final[2], table_z, half), final
    # teleport happens after alignment (gripper hovering above cube, open) and before closing
    assert not any(f["gripper_closed"] for f in fr[:idx])
    eef = fr[idx - 1]["eef_pos"]
    assert xy_dist(eef, start) < 0.02, "gripper not aligned above original cube at teleport"
    assert eef[2] > start[2] + 0.02
    # the gripper still closes afterwards (closes on air)
    assert any(f["gripper_closed"] for f in fr[idx:])
    assert max(f["cube_pos"][2] for f in fr[idx:]) - final[2] < 0.01


def test_normals_similar_but_not_identical():
    hashes = []
    starts = []
    for i in range(20):
        eid = f"normal_{i:02d}"
        hashes.append(hashlib.sha256((DATA / "videos" / BY_ID[eid]["file"]).read_bytes()).hexdigest())
        starts.append(labels(eid)["frames"][0]["cube_pos"])
    assert len(set(hashes)) == 20, "byte-identical normals"
    for a, b in zip(hashes, hashes[1:]):
        assert a != b
    mx = sum(s[0] for s in starts) / 20
    my = sum(s[1] for s in starts) / 20
    devs = [math.hypot(s[0] - mx, s[1] - my) for s in starts]
    assert max(devs) <= 0.006 * 2 + 1e-9  # jitter <= 6 mm around nominal (2x for mean offset)
    assert max(devs) > 1e-5, "no start jitter at all"
    # label trajectories differ too
    trajs = {json.dumps(labels(f"normal_{i:02d}")["frames"][90]["eef_pos"]) for i in range(20)}
    assert len(trajs) > 10
    # misses start near the normal nominal start
    for eid in ("miss_unseen", "miss_eval_right", "miss_eval_left", "miss_similar"):
        s = labels(eid)["frames"][0]["cube_pos"]
        assert math.hypot(s[0] - mx, s[1] - my) < 0.015


def test_summary_consistent_if_present():
    p = DATA / "summary.json"
    if not p.exists():
        pytest.skip("no summary.json")
    s = json.loads(p.read_text())
    assert {e["id"] for e in s["episodes"]} == set(BY_ID)
