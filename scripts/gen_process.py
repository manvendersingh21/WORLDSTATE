"""Generate the WORLDSTATE repeated process in robosuite Lift.

One fixed camera, the same scripted pick-and-place every run, and a true failure:
the cube is displaced 4-6 cm right after the gripper aligns above it, so the
open-loop grasp closes on air.

    .venv-sim/bin/python scripts/gen_process.py --out data/process [--force]

Writes manifest.json, videos/<id>.mp4 (H.264, faststart), labels/<id>.json and
summary.json. Labels are simulator ground truth for training a detector only;
the world model never reads them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

import robosuite as suite
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject
from robosuite.utils import camera_utils

FPS = 20
W, H = 480, 360
N_FRAMES = 180
CAMERA = "frontview"
FOVY = 28.0  # narrower than robosuite's 45 so the cell fills the frame
NOMINAL_XY = np.array([0.0, -0.08])  # where every run's cube starts (plus <= 5 mm jitter)
PAD_OFFSET = np.array([0.0, 0.20])  # pad center relative to the nominal cube xy
PAD_HALF = 0.04
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

# (phase, base steps, grip) — targets are filled per episode.
PHASES = [
    ("approach", 34, -1.0),
    ("align", 8, -1.0),
    ("descend", 20, -1.0),
    ("close", 12, 1.0),
    ("lift", 18, 1.0),
    ("carry", 28, 1.0),
    ("lower", 18, 1.0),
    ("release", 8, -1.0),
    ("retreat", 14, -1.0),
]


class LiftWithPad(Lift):
    """Lift plus a flat, visual-only green target pad (the fixture)."""

    def __init__(self, *args, pad_xy=(0.0, 0.20), **kwargs):
        self.pad_xy = pad_xy
        super().__init__(*args, **kwargs)

    def _load_model(self):
        super()._load_model()
        self.pad = BoxObject(
            name="pad",
            size=[PAD_HALF, PAD_HALF, 0.001],
            rgba=[0.15, 0.7, 0.25, 1.0],
            obj_type="visual",
            joints=None,
        )
        self.model.merge_objects([self.pad])
        top = self.model.mujoco_arena.table_top_abs
        self.pad.get_obj().set("pos", f"{self.pad_xy[0]} {self.pad_xy[1]} {top[2] + 0.001}")


def make_env() -> LiftWithPad:
    return LiftWithPad(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=FPS,
        horizon=10_000,
        hard_reset=False,
        ignore_done=True,
        pad_xy=tuple(NOMINAL_XY + PAD_OFFSET),
    )


def episode_specs() -> list[dict]:
    specs = []
    for i in range(20):
        specs.append(
            {
                "id": f"normal_{i:02d}",
                "kind": "normal",
                "seed": i,
                "role": "reference" if i < 16 else "heldout_normal",
                "split": "train" if i < 16 else "test",
            }
        )
    specs += [
        {"id": "miss_unseen", "kind": "miss_right", "seed": 100, "role": "unseen_failure", "split": "test"},
        {"id": "miss_eval_right", "kind": "miss_right", "seed": 101, "role": "eval_failure", "split": "test"},
        {"id": "miss_eval_left", "kind": "miss_left", "seed": 102, "role": "eval_failure", "split": "test"},
        {"id": "miss_similar", "kind": "miss_right", "seed": 103, "role": "similar_hidden", "split": "test"},
    ]
    for spec in specs:
        spec["file"] = f"{spec['id']}.mp4"
    return specs


def cube_joint(env) -> str:
    return env.cube.joints[0]


def set_cube(env, xy, z) -> None:
    env.sim.data.set_joint_qpos(cube_joint(env), np.array([xy[0], xy[1], z, 1.0, 0.0, 0.0, 0.0]))
    env.sim.data.set_joint_qvel(cube_joint(env), np.zeros(6))
    env.sim.forward()


def cube_pos(env) -> np.ndarray:
    return np.array(env.sim.data.body_xpos[env.cube_body_id])


def eef_pos(env) -> np.ndarray:
    return np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])


def eef_mat(env) -> np.ndarray:
    return np.array(env.sim.data.site_xmat[env.robots[0].eef_site_id["right"]]).reshape(3, 3)


def yaw(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotvec(m: np.ndarray) -> np.ndarray:
    angle = np.arccos(np.clip((np.trace(m) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) / (2.0 * np.sin(angle))
    return axis * angle


def box_from_points(points: np.ndarray, transform: np.ndarray) -> list[int]:
    pix = camera_utils.project_points_from_world_to_camera(points, transform, H, W)  # (row, col)
    rows, cols = pix[:, 0], pix[:, 1]
    x1, x2 = np.clip([cols.min(), cols.max()], 0, W)
    y1, y2 = np.clip([rows.min(), rows.max()], 0, H)
    x1, y1 = int(np.floor(x1)), int(np.floor(y1))
    x2, y2 = int(np.ceil(x2)), int(np.ceil(y2))
    x2 = min(W, max(x2, x1 + 1))
    y2 = min(H, max(y2, y1 + 1))
    return [x1, y1, x2, y2]


def corners(center, half) -> np.ndarray:
    half = np.broadcast_to(np.asarray(half, dtype=float), (3,))
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return np.asarray(center, dtype=float) + signs * half


def gripper_points(env) -> np.ndarray:
    pts = []
    for name in ("gripper0_right_finger1_visual", "gripper0_right_finger2_visual", "gripper0_right_hand_visual"):
        gid = env.sim.model.geom_name2id(name)
        half = 0.035 if "hand" in name else 0.012
        pts.append(corners(env.sim.data.geom_xpos[gid], [half, half, 0.02 if "hand" in name else 0.025]))
    return np.concatenate(pts)


def render(env) -> np.ndarray:
    frame = env.sim.render(camera_name=CAMERA, width=W, height=H)
    return np.ascontiguousarray(frame[::-1])


def run_episode(env, spec: dict, nominal_xy: np.ndarray, rest_z: float, transform: np.ndarray) -> tuple[list[np.ndarray], list[dict], dict]:
    rng = np.random.default_rng(spec["seed"])
    np.random.seed(spec["seed"])  # robosuite's reset noise uses the global generator
    env.reset()
    start_xy = nominal_xy + rng.uniform(-0.005, 0.005, size=2)
    set_cube(env, start_xy, rest_z)
    for _ in range(5):  # let contacts settle before the first frame
        env.sim.step()
    env.sim.forward()

    p0 = cube_pos(env)
    # Fingers close along x, so a cube displaced along y is beside them, not pushed by them.
    orient_goal = yaw(np.pi / 2) @ eef_mat(env)
    pad = np.array([nominal_xy[0] + PAD_OFFSET[0], nominal_xy[1] + PAD_OFFSET[1]])
    stretch = rng.uniform(0.95, 1.05)
    targets = {
        "approach": np.array([p0[0], p0[1], p0[2] + 0.09]),
        "align": np.array([p0[0], p0[1], p0[2] + 0.09]),
        "descend": np.array([p0[0], p0[1], p0[2] - 0.002]),
        "close": np.array([p0[0], p0[1], p0[2] - 0.002]),
        "lift": np.array([p0[0], p0[1], p0[2] + 0.13]),
        "carry": np.array([pad[0], pad[1], p0[2] + 0.13]),
        "lower": np.array([pad[0], pad[1], p0[2] + 0.012]),
        "release": np.array([pad[0], pad[1], p0[2] + 0.012]),
        "retreat": np.array([pad[0], pad[1], p0[2] + 0.17]),
    }
    plan = []
    for name, steps, grip in PHASES:
        plan += [(name, grip)] * max(1, int(round(steps * stretch)))
    idle_target = targets["retreat"]
    plan = plan[: N_FRAMES - 1]
    plan += [("idle", -1.0)] * (N_FRAMES - 1 - len(plan))

    miss = spec["kind"].startswith("miss")
    shift = rng.uniform(0.045, 0.055) * (1.0 if spec["kind"] == "miss_right" else -1.0)
    align_end = max(i for i, (name, _) in enumerate(plan) if name == "align")
    moved = None

    frames, labels = [], []
    pad_points = corners([pad[0], pad[1], env.model.mujoco_arena.table_top_abs[2] + 0.001], [PAD_HALF, PAD_HALF, 0.001])
    pad_box = box_from_points(pad_points, transform)
    grip_cmd = -1.0
    half = float(env.cube.size[2])

    def record(t_index: int) -> None:
        frames.append(render(env))
        c = cube_pos(env)
        labels.append(
            {
                "t": t_index / FPS,
                "boxes": {
                    "gripper": box_from_points(gripper_points(env), transform),
                    "part": box_from_points(corners(c, env.cube.size), transform),
                    "fixture": pad_box,
                },
                "cube_pos": [round(float(v), 5) for v in c],
                "eef_pos": [round(float(v), 5) for v in eef_pos(env)],
                "gripper_closed": bool(grip_cmd > 0),
            }
        )

    record(0)
    for step, (name, grip) in enumerate(plan, start=1):
        target = idle_target if name == "idle" else targets[name]
        delta = (target - eef_pos(env)) / 0.05
        action = np.zeros(env.action_dim)
        action[:3] = np.clip(delta * 0.9, -1.0, 1.0)
        action[3:6] = np.clip(rotvec(orient_goal @ eef_mat(env).T) / 0.5 * 0.5, -1.0, 1.0)
        action[-1] = grip
        grip_cmd = grip
        env.step(action)
        if miss and step - 1 == align_end and moved is None:
            # Displacement after alignment: the policy keeps the original grasp pose.
            c = cube_pos(env)
            set_cube(env, [c[0], c[1] + shift], c[2])
            moved = {"frame": step, "shift_y": round(float(shift), 4)}
        record(step)

    final = np.array(labels[-1]["cube_pos"])
    z_series = [f["cube_pos"][2] for f in labels]
    summary = {
        "id": spec["id"],
        "kind": spec["kind"],
        "final_cube_xy": [round(float(final[0]), 4), round(float(final[1]), 4)],
        "dist_to_pad": round(float(np.hypot(*(final[:2] - pad))), 4),
        "max_cube_z_rise": round(float(max(z_series) - z_series[0]), 4),
        "cube_half_size": half,
        "displacement": moved,
    }
    return frames, labels, summary


def encode(frames: list[np.ndarray], path: Path) -> None:
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "28", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-r", str(FPS), str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/process"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", nargs="*", help="episode ids to (re)generate")
    args = parser.parse_args()

    out: Path = args.out
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not args.force and not args.only:
        print(f"{manifest_path} exists; pass --force to regenerate")
        return
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    env = make_env()
    env.reset()
    nominal_xy = NOMINAL_XY
    table_z = float(env.model.mujoco_arena.table_top_abs[2])
    half = float(env.cube.size[2])
    rest_z = table_z + half
    env.sim.model.cam_fovy[env.sim.model.camera_name2id(CAMERA)] = FOVY
    transform = camera_utils.get_camera_transform_matrix(env.sim, CAMERA, H, W)
    pad = nominal_xy + PAD_OFFSET

    specs = episode_specs()
    summaries = {}
    old_summary = out / "summary.json"
    if args.only and old_summary.exists():
        summaries = {e["id"]: e for e in json.loads(old_summary.read_text())["episodes"]}
    for spec in specs:
        if args.only and spec["id"] not in args.only:
            continue
        frames, labels, summary = run_episode(env, spec, nominal_xy, rest_z, transform)
        encode(frames, out / "videos" / spec["file"])
        (out / "labels" / f"{spec['id']}.json").write_text(
            json.dumps({"id": spec["id"], "fps": FPS, "width": W, "height": H, "frames": labels})
        )
        summaries[spec["id"]] = summary
        print(f"{spec['id']:16} pad_dist={summary['dist_to_pad']:.3f} lift={summary['max_cube_z_rise']:.3f} {summary['displacement'] or ''}", flush=True)
    env.close()

    manifest = {
        "generator": "robosuite-lift",
        "generator_version": 1,
        "fps": FPS,
        "width": W,
        "height": H,
        "camera": CAMERA,
        "pad_center": [round(float(pad[0]), 4), round(float(pad[1]), 4), round(table_z + 0.001, 4)],
        "table_z": round(table_z, 5),
        "cube_half_size": round(half, 5),
        "episodes": [{k: spec[k] for k in ("id", "file", "kind", "seed", "role", "split")} for spec in specs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (out / "summary.json").write_text(json.dumps({"episodes": [summaries[s["id"]] for s in specs if s["id"] in summaries]}, indent=2))


if __name__ == "__main__":
    main()
