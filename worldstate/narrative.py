"""Searchable text for one analyzed run, derived from its tracks and the graph verdict.

The memory index stores this text so plain questions ("show me failed grasps", "when did
the cube move unexpectedly") retrieve the right runs. Nothing here uses ground-truth labels.
"""

from __future__ import annotations

import numpy as np


def run_facts(series: dict) -> dict:
    """Kinematic facts from tracked objects (normalized image coordinates)."""
    time = np.asarray(series["time"], dtype=float)
    lift = np.asarray(series["lift"], dtype=float)
    speed = np.asarray(series["obj_speed"], dtype=float)
    dist_go = np.asarray(series["dist_go"], dtype=float)
    dist_ob = np.asarray(series["dist_ob"], dtype=float)
    # Smoothing is zero-padded, so the first and last samples are not real positions.
    settle = 3 if len(lift) > 12 else 0
    core = lift[settle : len(lift) - settle] if settle else lift
    base = float(np.median(core[: max(3, len(core) // 10)])) if len(core) else 0.0
    rise = float(np.max(core) - base) if len(core) else 0.0
    lifted = rise > 0.05
    # An unexpected shift: the part jumps while resting on the surface and is still not
    # lifted a second later. When the gripper lifts a part, the jump is followed by a rise.
    fps = float(series["fps"][0]) if "fps" in series else 20.0
    hold = max(1, int(round(fps)))
    shift_t = None
    for i in np.where(speed > 0.15)[0]:
        if i < settle:
            continue
        window = lift[i : i + hold]
        if lift[i] - base < 0.02 and len(window) and np.max(window) - base < 0.02:
            shift_t = float(time[i])
            break
    tail = dist_ob[int(len(dist_ob) * 0.85) : len(dist_ob) - settle] if len(dist_ob) > 12 else dist_ob
    on_target = bool(np.median(tail) < 0.06)
    return {"lifted": lifted, "rise": rise, "shift_t": shift_t, "ends_on_target": on_target}


def describe_run(episode_id: str, detail: dict, series: dict, label: str | None = None) -> str:
    facts = run_facts(series)
    parts = []
    status = detail.get("status")
    if status == "normal":
        parts.append(f"Run {episode_id}: normal run, consistent with the learned process; the part was picked, carried and placed on the target.")
    else:
        div = detail.get("first_divergence_s")
        exp, obs = detail.get("expected", {}), detail.get("observed", {})
        kind = "known failure" if status == "known_failure" else "unusual novel transition, anomaly"
        parts.append(
            f"Run {episode_id}: {kind} at {div}s. Expected {exp.get('from')} -> {exp.get('to')}, "
            f"observed {obs.get('to')}; {(detail.get('support') or {}).get('text', '')}."
        )
    if facts["shift_t"] is not None:
        parts.append(f"The cube moved unexpectedly (object shifted) at {facts['shift_t']:.2f}s before the gripper reached it.")
    if not facts["lifted"]:
        parts.append("Failed grasp, missed grasp: the gripper closed but the part never lifted.")
    if status != "normal" and not facts["ends_on_target"]:
        parts.append("The part ended outside the target, not placed.")
    if label:
        parts.append(f"Failure class {label.replace('_', ' ')}.")
    events = " ".join(f"At {ev['time']:.2f}s {ev['action']}." for ev in detail.get("events", []))
    return " ".join(parts + [events]).strip()
