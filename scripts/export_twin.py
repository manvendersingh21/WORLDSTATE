"""Export browser-ready digital-twin assets for the WORLDSTATE robot cell.

Replays the episodes of scripts/gen_process.py (same env, same scripted policy,
same per-episode seed) without re-rendering any video, and records per frame the
arm joint angles, the finger slides, the cube and eef poses, and the world pose
of every Panda link body -- so a Three.js viewer needs no forward kinematics.

Caveat worth knowing: gen_process is not bit-reproducible under robosuite 1.5.2.
Its np.random.seed(spec["seed"]) no longer reaches the simulator, because
MujocoEnv.__init__ builds self.rng = np.random.default_rng(None) and the robot's
reset noise and the cube's half-extents are drawn from that entropy-seeded
stream. The state that produced data/process is unrecoverable. So this exporter
pins the construction seed (its own output is byte-stable) and then snaps frame
zero's end effector onto the recorded data/process/labels eef_pos with a damped
least-squares IK, which puts the twin back on top of the recorded video instead
of up to 5 cm away from it. Pass --no-align for the unaligned replay. The
residuals against the labels are written to index.json["replay"].

Also bakes the MuJoCo visual geoms of each link into one decimated GLB per link,
in that link's body frame, so the viewer just assigns link_pos/link_quat.

    .venv-sim/bin/python scripts/export_twin.py            # trajectories + meshes
    .venv-sim/bin/python scripts/export_twin.py --meshes   # meshes only
    .venv-sim/bin/python scripts/export_twin.py --traj     # trajectories only
    .venv-sim/bin/python scripts/export_twin.py --verify   # check what is on disk

Writes data/process/twin/<id>.json, data/process/twin/index.json and
web/twin/assets/<link>.glb (+ NOTICE.txt). Nothing under data/process/videos,
data/process/labels or scripts/gen_process.py is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
TWIN_DIR = ROOT / "data" / "process" / "twin"
LABELS_DIR = ROOT / "data" / "process" / "labels"
MANIFEST = ROOT / "data" / "process" / "manifest.json"
ASSETS_DIR = ROOT / "web" / "twin" / "assets"

SCHEMA_VERSION = 1
DECIMALS = 4
POS_DECIMALS = 5  # cube_pos / eef_pos, to match data/process/labels exactly
# Pins robosuite's entropy-seeded self.rng so the export is stable. 40 of the
# first 60 seeds is the one whose compiled cube half-height lands on the 0.0219
# that manifest.json recorded, so the twin's cube rests at the filmed height.
ENV_SEED = 40
FACE_BUDGET = 110_000  # total triangles across all link GLBs
FACE_FLOOR = 1_500  # small parts (the fingers) are seen close up, leave them alone
GLB_BUDGET = 8 * 1024 * 1024
# The replay cannot be bit-exact (see the module docstring), so the labels are
# matched to a tolerance instead; these are the agreed bounds with peer b. The
# eef is snapped onto the label at frame 0 and the controller transient then
# peaks around frame 7 before settling, hence the looser bound before STEADY_AT.
CUBE_TOL = 0.005
EEF_TOL = 0.025
EEF_STEADY_TOL = 0.005
EEF_MEDIAN_TOL = 0.001
STEADY_AT = 40

# (asset name, MuJoCo body name). The per-frame link_pos/link_quat arrays and
# index.json["links"] follow this order.
LINKS = [
    ("link0", "robot0_link0"),
    ("link1", "robot0_link1"),
    ("link2", "robot0_link2"),
    ("link3", "robot0_link3"),
    ("link4", "robot0_link4"),
    ("link5", "robot0_link5"),
    ("link6", "robot0_link6"),
    ("link7", "robot0_link7"),
    ("hand", "gripper0_right_right_gripper"),
    ("leftfinger", "gripper0_right_leftfinger"),
    ("rightfinger", "gripper0_right_rightfinger"),
]


def r(value, decimals: int = DECIMALS) -> float:
    out = round(float(value), decimals)
    return 0.0 if out == 0.0 else out  # kill -0.0 so the JSON stays stable


def rlist(values, decimals: int = DECIMALS) -> list[float]:
    return [r(v, decimals) for v in np.asarray(values).reshape(-1)]


def quat_from_mat(mat: np.ndarray) -> np.ndarray:
    """Rotation matrix (3x3) -> MuJoCo-order quaternion (w, x, y, z)."""
    import mujoco

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(mat, dtype=float).reshape(9))
    return quat


# --------------------------------------------------------------------------- #
# trajectories
# --------------------------------------------------------------------------- #


def build_env(gp):
    """gen_process.make_env(), but with robosuite's internal RNG pinned.

    The cube's half-extents are drawn inside the constructor, so the seed has to
    be in place before make_env() runs; gen_process does not expose it, hence the
    temporary swap of the class it instantiates.
    """
    import functools

    original = gp.LiftWithPad
    gp.LiftWithPad = functools.partial(original, seed=ENV_SEED)
    try:
        env = gp.make_env()
    finally:
        gp.LiftWithPad = original
    env.reset()
    return env


def align_eef(env, target_pos, iters: int = 80, damping: float = 0.05) -> float:
    """Nudge the 7 arm joints so the end effector sits on target_pos.

    Damped least squares from the pose robosuite just randomised into, so the
    result stays a plausible near-nominal arm configuration. Returns the residual.
    """
    import mujoco

    model, data = env.sim.model._model, env.sim.data._data
    site = env.robots[0].eef_site_id["right"]
    qpos_idx = np.asarray(env.robots[0]._ref_joint_pos_indexes, dtype=int)
    dof_idx = np.asarray(env.robots[0]._ref_joint_vel_indexes, dtype=int)
    low, high = model.jnt_range[: len(qpos_idx)].T
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    target = np.asarray(target_pos, dtype=float)

    mujoco.mj_forward(model, data)
    for _ in range(iters):
        error = target - data.site_xpos[site]
        if np.linalg.norm(error) < 1e-7:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, site)
        jac = jacp[:, dof_idx]
        step = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(3), error)
        data.qpos[qpos_idx] = np.clip(data.qpos[qpos_idx] + np.clip(step, -0.2, 0.2), low, high)
        mujoco.mj_forward(model, data)
    data.qvel[dof_idx] = 0.0
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(target - data.site_xpos[site]))


def _sim_handles(env):
    import mujoco

    model = env.sim.model._model
    data = env.sim.data._data
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for _, b in LINKS]
    if any(i < 0 for i in body_ids):
        missing = [b for (_, b), i in zip(LINKS, body_ids) if i < 0]
        raise RuntimeError(f"body names not found in the compiled model: {missing}")
    return model, data, body_ids


def replay_all(align: bool = True, progress=print) -> tuple[dict, dict]:
    """Replay every episode of gen_process and capture the twin state per frame.

    gen_process.run_episode calls render(env) as the first statement of its own
    record(), so swapping that one function out gives us the simulator state at
    exactly the frames the videos and labels were written from, with the physics
    untouched. Nothing is rendered and no video is written. set_cube is the other
    swap: it is called once right after reset, which is the only opening to put
    the arm back where the recorded episode started it.
    """
    sys.path.insert(0, str(SCRIPTS))
    import gen_process as gp
    from robosuite.utils import camera_utils

    env = build_env(gp)
    model, data, body_ids = _sim_handles(env)
    cube_bid = env.cube_body_id
    arm_q = list(env.robots[0]._ref_joint_pos_indexes)
    finger_q = list(env.robots[0]._ref_gripper_joint_pos_indexes["right"])
    eef_site = env.robots[0].eef_site_id["right"]

    captured: list[dict] = []
    blank = np.zeros((1, 1, 3), dtype=np.uint8)

    def capture(_env):
        captured.append(
            {
                "qpos": rlist(data.qpos[arm_q]),
                "finger_qpos": rlist(data.qpos[finger_q]),
                "cube_quat": rlist(data.xquat[cube_bid]),
                "eef_quat": rlist(quat_from_mat(data.site_xmat[eef_site])),
                "link_pos": [rlist(data.xpos[i]) for i in body_ids],
                "link_quat": [rlist(data.xquat[i]) for i in body_ids],
            }
        )
        return blank

    # Mirror gen_process.main()'s setup so the simulator sees the same sequence.
    nominal_xy = gp.NOMINAL_XY
    table_z = float(env.model.mujoco_arena.table_top_abs[2])
    rest_z = table_z + float(env.cube.size[2])
    env.sim.model.cam_fovy[env.sim.model.camera_name2id(gp.CAMERA)] = gp.FOVY
    transform = camera_utils.get_camera_transform_matrix(env.sim, gp.CAMERA, gp.H, gp.W)

    meta = scene_meta(env, gp, table_z)
    episodes: dict[str, dict] = {}
    real_render, real_set_cube = gp.render, gp.set_cube
    pending: list[list[float]] = []  # frame-0 eef target, consumed by the first call
    residuals: list[float] = []

    def set_cube_and_align(_env, xy, z):
        # run_episode also calls set_cube mid-episode to displace the cube on the
        # miss episodes; only the placement right after reset may move the arm.
        real_set_cube(_env, xy, z)
        if pending:
            residuals.append(align_eef(_env, pending.pop()))

    gp.render = capture
    gp.set_cube = set_cube_and_align
    try:
        for spec in gp.episode_specs():
            captured.clear()
            recorded = json.loads((LABELS_DIR / f"{spec['id']}.json").read_text())["frames"]
            pending[:] = [recorded[0]["eef_pos"]] if align else []
            _frames, labels, _summary = gp.run_episode(env, spec, nominal_xy, rest_z, transform)
            if len(captured) != len(labels):
                raise RuntimeError(f"{spec['id']}: captured {len(captured)} states for {len(labels)} frames")
            episodes[spec["id"]] = {
                "id": spec["id"],
                "fps": gp.FPS,
                "n_frames": len(labels),
                "links": [name for name, _ in LINKS],
                "frames": [
                    {
                        "t": r(label["t"], 4),
                        "qpos": state["qpos"],
                        "finger_qpos": state["finger_qpos"],
                        "cube_pos": rlist(label["cube_pos"], POS_DECIMALS),
                        "cube_quat": state["cube_quat"],
                        "eef_pos": rlist(label["eef_pos"], POS_DECIMALS),
                        "eef_quat": state["eef_quat"],
                        "gripper_closed": bool(label["gripper_closed"]),
                        "link_pos": state["link_pos"],
                        "link_quat": state["link_quat"],
                    }
                    for label, state in zip(labels, captured)
                ],
            }
            progress(f"  replayed {spec['id']:16} {len(labels)} frames")
    finally:
        gp.render, gp.set_cube = real_render, real_set_cube
        env.close()
    meta["replay"] = replay_report(episodes, align, residuals)
    return meta, episodes


def replay_report(episodes: dict, align: bool, residuals: list[float]) -> dict:
    """How far the replay lands from the recorded labels, per frame, in metres."""
    eef, cube, mismatched = [], [], 0
    for eid, episode in episodes.items():
        recorded = json.loads((LABELS_DIR / f"{eid}.json").read_text())["frames"]
        for frame, label in zip(episode["frames"], recorded):
            eef.append(np.linalg.norm(np.subtract(frame["eef_pos"], label["eef_pos"])))
            cube.append(np.linalg.norm(np.subtract(frame["cube_pos"], label["cube_pos"])))
            mismatched += frame["gripper_closed"] != label["gripper_closed"]
    return {
        "env_seed": ENV_SEED,
        "frame0_eef_aligned_to_labels": align,
        "note": (
            "gen_process is not bit-reproducible under robosuite 1.5.2: the robot reset "
            "noise and the cube half-extents come from MujocoEnv's entropy-seeded self.rng, "
            "which np.random.seed no longer reaches. These are the residuals of this replay "
            "against data/process/labels, in metres."
        ),
        "eef_err_max": r(max(eef), 6),
        "eef_err_median": r(float(np.median(eef)), 6),
        "cube_err_max": r(max(cube), 6),
        "cube_err_median": r(float(np.median(cube)), 6),
        "ik_residual_max": r(max(residuals), 6) if residuals else None,
        "gripper_closed_mismatches": int(mismatched),
        "frames_compared": len(eef),
    }


def scene_meta(env, gp, table_z: float) -> dict:
    """Static cell geometry the viewer needs: camera, table, pad, cube, base."""
    import mujoco

    model = env.sim.model._model
    data = env.sim.data._data
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, gp.CAMERA)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")
    table = np.asarray(env.model.mujoco_arena.table_full_size, dtype=float)
    pad = gp.NOMINAL_XY + gp.PAD_OFFSET
    return {
        "camera": {
            "name": gp.CAMERA,
            "fovy": float(gp.FOVY),
            "width": gp.W,
            "height": gp.H,
            "pos": rlist(data.cam_xpos[cam]),
            "quat": rlist(quat_from_mat(data.cam_xmat[cam])),
        },
        "world": {
            "table_top_z": r(table_z, POS_DECIMALS),
            "table_full_size": rlist(table, POS_DECIMALS),
            "table_half_size": rlist(table / 2.0, POS_DECIMALS),
            "pad_center": [r(pad[0]), r(pad[1]), r(table_z + 0.001)],
            "pad_half_size": [r(gp.PAD_HALF), r(gp.PAD_HALF), 0.001],
            "pad_rgba": [0.15, 0.7, 0.25, 1.0],
            "cube_half_size": rlist(env.cube.size, POS_DECIMALS),
            "robot_base_pos": rlist(data.xpos[base], POS_DECIMALS),
            "robot_base_quat": rlist(data.xquat[base]),
        },
    }


def write_trajectories(meta: dict, episodes: dict) -> None:
    TWIN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    for episode in episodes.values():
        path = TWIN_DIR / f"{episode['id']}.json"
        path.write_text(json.dumps(episode, separators=(",", ":")))
    index = {
        "generator": "robosuite-lift",
        "source_script": "scripts/gen_process.py",
        "schema_version": SCHEMA_VERSION,
        "fps": manifest["fps"],
        "n_frames": len(next(iter(episodes.values()))["frames"]),
        "units": "meters",
        "quat_order": "wxyz",
        "up_axis": "+Z",
        "decimals": DECIMALS,
        "assets_dir": "web/twin/assets",
        "camera": meta["camera"],
        "world": meta["world"],
        "replay": meta["replay"],
        "links": [
            {"name": name, "body": body, "mesh": f"{name}.glb"} for name, body in LINKS
        ],
        "episodes": [
            {**{k: ep[k] for k in ("id", "kind", "seed", "role", "split")},
             "file": f"{ep['id']}.json",
             "n_frames": len(episodes[ep["id"]]["frames"])}
            for ep in manifest["episodes"]
        ],
    }
    (TWIN_DIR / "index.json").write_text(json.dumps(index, indent=1))


# --------------------------------------------------------------------------- #
# meshes
# --------------------------------------------------------------------------- #


def _geom_color(model, geom: int) -> tuple:
    matid = int(model.geom_matid[geom])
    rgba = model.mat_rgba[matid] if matid >= 0 else model.geom_rgba[geom]
    return tuple(round(float(c), 3) for c in rgba)


def _geom_mesh(model, geom: int):
    """Visual geom vertices/faces expressed in the parent body frame.

    MuJoCo renders mesh_vert directly in the geom frame, so composing it with
    geom_pos/geom_quat is all that is needed to land in the body frame.
    """
    import mujoco

    mid = int(model.geom_dataid[geom])
    va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    verts = np.array(model.mesh_vert[va : va + vn]).reshape(-1, 3).astype(np.float64)
    faces = np.array(model.mesh_face[fa : fa + fn]).reshape(-1, 3).astype(np.int64)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(model.geom_quat[geom], dtype=float))
    verts = verts @ rot.reshape(3, 3).T + np.asarray(model.geom_pos[geom], dtype=float)
    return verts, faces


def _pbr(rgba: tuple):
    """Panda's palette is white / light gray / dark gray; keep it plasticky."""
    from trimesh.visual.material import PBRMaterial

    luma = float(np.dot(rgba[:3], (0.2126, 0.7152, 0.0722)))
    if luma > 0.8:
        metallic, roughness = 0.05, 0.45
    elif luma > 0.4:
        metallic, roughness = 0.25, 0.38
    else:
        metallic, roughness = 0.35, 0.5
    return PBRMaterial(
        name=f"panda_{int(round(luma * 255)):03d}",
        baseColorFactor=[*(float(c) for c in rgba[:3]), float(rgba[3])],
        metallicFactor=metallic,
        roughnessFactor=roughness,
        doubleSided=False,
        alphaMode="OPAQUE",
    )


def _simplify(verts: np.ndarray, faces: np.ndarray, target: int):
    target = max(target, FACE_FLOOR)
    if target >= len(faces):
        return verts, faces
    import fast_simplification

    reduction = 1.0 - target / float(len(faces))
    out_v, out_f = fast_simplification.simplify(
        verts.astype(np.float32), faces.astype(np.int32), min(max(reduction, 0.0), 0.95)
    )
    return np.asarray(out_v, dtype=np.float64), np.asarray(out_f, dtype=np.int64)


def export_meshes(progress=print) -> None:
    import trimesh

    sys.path.insert(0, str(SCRIPTS))
    import gen_process as gp
    import mujoco

    env = gp.make_env()
    env.reset()
    model = env.sim.model._model

    # Group every link's visual geoms by colour; one GLB mesh per colour.
    per_link: dict[str, dict[tuple, list]] = {}
    for name, body in LINKS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        groups: dict[tuple, list] = {}
        for geom in range(model.ngeom):
            is_visual = model.geom_bodyid[geom] == bid and model.geom_group[geom] == 1
            if not is_visual or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            groups.setdefault(_geom_color(model, geom), []).append(_geom_mesh(model, geom))
        if not groups:
            raise RuntimeError(f"no visual mesh geoms on body {body}")
        per_link[name] = groups
    env.close()

    total_faces = sum(len(f) for g in per_link.values() for parts in g.values() for _, f in parts)
    keep = min(1.0, FACE_BUDGET / float(total_faces))
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    written = {}
    for name, groups in per_link.items():
        scene = trimesh.Scene()
        for i, (rgba, parts) in enumerate(sorted(groups.items(), reverse=True)):
            merged = trimesh.util.concatenate(
                [trimesh.Trimesh(vertices=v, faces=f, process=False) for v, f in parts]
            )
            merged.merge_vertices()
            verts, faces = _simplify(
                merged.vertices, merged.faces, int(round(len(merged.faces) * keep))
            )
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
            mesh.visual = trimesh.visual.TextureVisuals(material=_pbr(rgba))
            scene.add_geometry(mesh, geom_name=f"{name}_{i}", node_name=f"{name}_{i}")
        path = ASSETS_DIR / f"{name}.glb"
        path.write_bytes(scene.export(file_type="glb"))
        written[name] = (path.stat().st_size, sum(len(m.faces) for m in scene.geometry.values()))
        progress(f"  {name:12} {written[name][1]:6d} tris  {written[name][0] / 1024:7.1f} KiB")

    total_bytes = sum(size for size, _ in written.values())
    progress(f"  total {total_bytes / 1024 / 1024:.2f} MiB of {GLB_BUDGET / 1024 / 1024:.0f} MiB budget")
    if total_bytes >= GLB_BUDGET:
        raise RuntimeError(f"GLB total {total_bytes} bytes exceeds the {GLB_BUDGET} byte budget")
    write_notice()


NOTICE = """WORLDSTATE browser digital twin -- third-party asset attributions
================================================================

The GLB files in this directory (link0..link7, hand, leftfinger, rightfinger)
were produced by scripts/export_twin.py from the visual meshes that ship with
robosuite, baked into each link's body frame and decimated for the browser.
No geometry from any other source is included.

robosuite
    Copyright (c) 2018-2024 ARISE Initiative.
    Licensed under the MIT License.
    https://github.com/ARISE-Initiative/robosuite
    Source of the MuJoCo model (robots/panda/robot.xml, grippers/panda_gripper.xml)
    and of the mesh files under models/assets/robots/panda and
    models/assets/grippers/meshes/panda_gripper.

Franka Emika Panda description
    Copyright the Franka Emika GmbH / franka_ros contributors.
    Licensed under the Apache License, Version 2.0.
    https://github.com/frankaemika/franka_ros
    http://www.apache.org/licenses/LICENSE-2.0
    The Panda arm, hand and finger meshes redistributed by robosuite derive from
    these assets. They are redistributed here unmodified in shape apart from
    mesh decimation and a rigid change of coordinate frame.

MuJoCo
    Copyright (c) Google DeepMind. Licensed under the Apache License, Version 2.0.
    https://github.com/google-deepmind/mujoco
    Used to compile the model from which these meshes were extracted.

A copy of the Apache License, Version 2.0 is available at
http://www.apache.org/licenses/LICENSE-2.0. No trademark rights of Franka Emika
GmbH are granted by these licenses.
"""


def write_notice() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    (ASSETS_DIR / "NOTICE.txt").write_text(NOTICE)


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #


def verify() -> int:
    problems: list[str] = []

    def check(condition, message):
        if not condition:
            problems.append(message)

    index_path = TWIN_DIR / "index.json"
    if not index_path.exists():
        print(f"missing {index_path}", file=sys.stderr)
        return 1
    index = json.loads(index_path.read_text())
    manifest = json.loads(MANIFEST.read_text())
    names = [name for name, _ in LINKS]

    check(index["schema_version"] == SCHEMA_VERSION, "index schema_version mismatch")
    check(index["fps"] == manifest["fps"], "index fps does not match the manifest")
    check([link["name"] for link in index["links"]] == names, "index link order mismatch")
    check([link["body"] for link in index["links"]] == [b for _, b in LINKS], "index link bodies mismatch")
    check(
        [ep["id"] for ep in index["episodes"]] == [ep["id"] for ep in manifest["episodes"]],
        "index episode ids do not match the manifest",
    )

    for entry in manifest["episodes"]:
        eid = entry["id"]
        path = TWIN_DIR / f"{eid}.json"
        if not path.exists():
            problems.append(f"{eid}: missing {path}")
            continue
        twin = json.loads(path.read_text())
        labels = json.loads((LABELS_DIR / f"{eid}.json").read_text())["frames"]
        eef_errors: list[float] = []
        check(twin["id"] == eid, f"{eid}: id mismatch")
        check(twin["links"] == names, f"{eid}: link order mismatch")
        check(len(twin["frames"]) == len(labels), f"{eid}: frame count {len(twin['frames'])} != {len(labels)}")
        for i, (frame, label) in enumerate(zip(twin["frames"], labels)):
            where = f"{eid}[{i}]"
            check(len(frame["qpos"]) == 7, f"{where}: expected 7 arm joints")
            check(len(frame["finger_qpos"]) == 2, f"{where}: expected 2 finger joints")
            check(len(frame["link_pos"]) == len(names), f"{where}: link_pos length")
            check(len(frame["link_quat"]) == len(names), f"{where}: link_quat length")
            cube_err = float(np.linalg.norm(np.subtract(frame["cube_pos"], label["cube_pos"])))
            eef_err = float(np.linalg.norm(np.subtract(frame["eef_pos"], label["eef_pos"])))
            eef_errors.append(eef_err)
            bound = EEF_TOL if i < STEADY_AT else EEF_STEADY_TOL
            check(cube_err <= CUBE_TOL, f"{where}: cube_pos {cube_err:.4f} m from the label")
            check(eef_err <= bound, f"{where}: eef_pos {eef_err:.4f} m from the label (bound {bound})")
            check(
                frame["gripper_closed"] == label["gripper_closed"],
                f"{where}: gripper_closed differs from the label",
            )
            quats = [frame["cube_quat"], frame["eef_quat"], *frame["link_quat"]]
            worst = max(abs(float(np.linalg.norm(q)) - 1.0) for q in quats)
            check(worst < 1e-3, f"{where}: quaternion norm off by {worst:.2e}")
            if problems:
                break
        median = float(np.median(eef_errors)) if eef_errors else 0.0
        check(median <= EEF_MEDIAN_TOL, f"{eid}: median eef error {median:.5f} m exceeds {EEF_MEDIAN_TOL}")
        if problems:
            break

    total_bytes = 0
    for name in names:
        path = ASSETS_DIR / f"{name}.glb"
        if not path.exists():
            problems.append(f"missing {path}")
            continue
        blob = path.read_bytes()
        total_bytes += len(blob)
        check(blob[:4] == b"glTF", f"{path.name}: not a GLB container")
    check(total_bytes < GLB_BUDGET, f"GLB total {total_bytes} bytes exceeds {GLB_BUDGET}")
    notice = ASSETS_DIR / "NOTICE.txt"
    check(notice.exists() and "Apache" in notice.read_text(), "NOTICE.txt missing or incomplete")

    if problems:
        for problem in problems[:20]:
            print(f"FAIL {problem}", file=sys.stderr)
        return 1
    print(
        f"OK  {len(manifest['episodes'])} episodes x {index['n_frames']} frames, "
        f"{len(names)} links, {total_bytes / 1024 / 1024:.2f} MiB of GLB"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traj", action="store_true", help="export trajectories only")
    parser.add_argument("--meshes", action="store_true", help="export GLB assets only")
    parser.add_argument("--verify", action="store_true", help="validate the files already on disk")
    parser.add_argument(
        "--no-align",
        dest="align",
        action="store_false",
        help="skip the frame-0 IK snap onto the recorded label eef_pos",
    )
    args = parser.parse_args()

    if args.verify:
        return verify()

    do_traj = args.traj or not args.meshes
    do_meshes = args.meshes or not args.traj
    if do_traj:
        print("replaying episodes")
        meta, episodes = replay_all(align=args.align)
        write_trajectories(meta, episodes)
        print(f"  residuals vs labels: {json.dumps(meta['replay'])}")
        print(f"wrote {len(episodes)} episode files + index.json to {TWIN_DIR}")
    if do_meshes:
        print("baking link meshes")
        export_meshes()
        print(f"wrote {len(LINKS)} GLB files + NOTICE.txt to {ASSETS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
