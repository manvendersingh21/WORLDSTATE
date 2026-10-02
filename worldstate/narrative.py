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
    base = float(np.median(lift[: max(3, len(lift) // 10)]))
    rise = float(np.max(lift) - base) if len(lift) else 0.0
    lifted = rise > 0.05
    # The part moving fast while the gripper is not on it: an unexpected shift.
    away = dist_go > np.percentile(dist_go, 25) if len(dist_go) else np.zeros(0, bool)
    jumps = np.where((speed > 0.25) & away)[0]
    shift_t = float(time[jumps[0]]) if len(jumps) else None
    early = time < 0.5 * float(time[-1]) if len(time) else np.zeros(0, bool)
    if shift_t is not None and not early[jumps[0]]:
        shift_t = None
    tail = dist_ob[int(len(dist_ob) * 0.85):] if len(dist_ob) else np.zeros(1)
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
