"""Generate the corrected attempt for the displaced-after-align failure.

Same robosuite cell, seed and displacement as miss_unseen, but the policy is closed
loop after alignment: it sees the cube moved, re-aligns over the new position, then
grasps and places it. Writes a camera clip and a digital-twin trajectory:

    .venv-sim/bin/python scripts/gen_recovery.py
    -> data/process/recovery/recovered_attempt.mp4
    -> web/twin/data/recovered_attempt.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_process as gp  # noqa: E402

N_FRAMES = 200  # 10 s: the re-align adds about a second
LINK_BODIES = [
    "robot0_link0", "robot0_link1", "robot0_link2", "robot0_link3", "robot0_link4", "robot0_link5",
    "robot0_link6", "robot0_link7", "gripper0_right_right_gripper", "gripper0_right_leftfinger",
    "gripper0_right_rightfinger",
]
LINK_NAMES = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "leftfinger", "rightfinger"]


def mat_to_quat(m: np.ndarray) -> list[float]:
    w = np.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    x = np.copysign(np.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0, m[2, 1] - m[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2.0, m[0, 2] - m[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2.0, m[1, 0] - m[0, 1])
    return [round(float(v), 4) for v in (w, x, y, z)]


def main() -> None:
    env = gp.make_env()
    env.reset()
    nominal_xy = gp.NOMINAL_XY
    table_z = float(env.model.mujoco_arena.table_top_abs[2])
    rest_z = table_z + float(env.cube.size[2])
    env.sim.model.cam_fovy[env.sim.model.camera_name2id(gp.CAMERA)] = gp.FOVY
    body_ids = [env.sim.model.body_name2id(b) for b in LINK_BODIES]

    spec = {"id": "recovered_attempt", "seed": 100, "kind": "miss_right"}  # miss_unseen's seed and shift
    rng = np.random.default_rng(spec["seed"])
    np.random.seed(spec["seed"])
    env.reset()
    gp.set_cube(env, nominal_xy + rng.uniform(-0.005, 0.005, size=2), rest_z)
    for _ in range(5):
        env.sim.step()
    env.sim.forward()

    p0 = gp.cube_pos(env)
    orient_goal = gp.yaw(np.pi / 2) @ gp.eef_mat(env)
    pad = nominal_xy + gp.PAD_OFFSET
    stretch = rng.uniform(0.95, 1.05)
    shift = rng.uniform(0.045, 0.055)

    def targets_for(c: np.ndarray) -> dict:
        return {
            "approach": np.array([c[0], c[1], c[2] + 0.09]),
            "align": np.array([c[0], c[1], c[2] + 0.09]),
            "realign": np.array([c[0], c[1], c[2] + 0.09]),
            "descend": np.array([c[0], c[1], c[2] - 0.002]),
            "close": np.array([c[0], c[1], c[2] - 0.002]),
            "lift": np.array([c[0], c[1], c[2] + 0.13]),
            "carry": np.array([pad[0], pad[1], c[2] + 0.13]),
            "lower": np.array([pad[0], pad[1], c[2] + 0.012]),
            "release": np.array([pad[0], pad[1], c[2] + 0.012]),
            "retreat": np.array([pad[0], pad[1], c[2] + 0.17]),
        }

    phases = list(gp.PHASES)
    phases.insert(2, ("realign", 22, -1.0))  # after align: move over the cube's new position
    plan = []
    for name, steps, grip in phases:
        plan += [(name, grip)] * max(1, int(round(steps * stretch)))
    plan = plan[: N_FRAMES - 1] + [("idle", -1.0)] * max(0, N_FRAMES - 1 - len(plan))
    align_end = max(i for i, (name, _) in enumerate(plan) if name == "align")
    targets = targets_for(p0)

    frames, twin = [], []
    grip_cmd = -1.0

    def record(i: int) -> None:
        frames.append(gp.render(env))
        c = gp.cube_pos(env)
        cube_quat = [round(float(v), 4) for v in env.sim.data.body_xquat[env.cube_body_id]]
        twin.append({
            "t": round(i / gp.FPS, 4), "qpos": [], "finger_qpos": [],
            "cube_pos": [round(float(v), 4) for v in c], "cube_quat": cube_quat,
            "eef_pos": [round(float(v), 4) for v in gp.eef_pos(env)], "eef_quat": mat_to_quat(gp.eef_mat(env)),
            "gripper_closed": bool(grip_cmd > 0),
            "link_pos": [[round(float(v), 4) for v in env.sim.data.body_xpos[b]] for b in body_ids],
            "link_quat": [[round(float(v), 4) for v in env.sim.data.body_xquat[b]] for b in body_ids],
        })

    record(0)
    for step, (name, grip) in enumerate(plan, start=1):
        target = targets["retreat"] if name == "idle" else targets[name]
        action = np.zeros(env.action_dim)
        action[:3] = np.clip((target - gp.eef_pos(env)) / 0.05 * 0.9, -1.0, 1.0)
        action[3:6] = np.clip(gp.rotvec(orient_goal @ gp.eef_mat(env).T) / 0.5 * 0.5, -1.0, 1.0)
        action[-1] = grip
        grip_cmd = grip
        env.step(action)
        if step - 1 == align_end:
            c = gp.cube_pos(env)
            gp.set_cube(env, [c[0], c[1] + shift], c[2])
            # Closed loop: the corrected policy re-reads the cube pose and re-plans.
            targets = targets_for(gp.cube_pos(env))
        record(step)

    final = np.array(twin[-1]["cube_pos"])
    dist = float(np.hypot(*(final[:2] - pad)))
    rise = max(f["cube_pos"][2] for f in twin) - twin[0]["cube_pos"][2]
    print(f"recovered_attempt: final pad distance {dist:.3f} m, max cube rise {rise:.3f} m, shift {shift:.4f} m")
    out = Path("data/process/recovery")
    out.mkdir(parents=True, exist_ok=True)
    gp.encode(frames, out / "recovered_attempt.mp4")
    Path("web/twin/data/recovered_attempt.json").write_text(json.dumps({
        "id": "recovered_attempt", "fps": gp.FPS, "n_frames": len(twin), "links": LINK_NAMES, "frames": twin,
        "note": "Simulated corrective attempt: same seed and displacement as miss_unseen, closed-loop re-align after the shift.",
        "shift_m": round(float(shift), 4), "realign_start_s": round((align_end + 1) / gp.FPS, 3),
        "placed": dist < 0.03 and rise > 0.04,
    }))
    env.close()


if __name__ == "__main__":
    main()
