"""Unsupervised state discovery, transition graph, novelty, and self-learning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from worldstate.config import MODEL_PATH, STRIDE, TRAJ_SAMPLES, WINDOW
from worldstate.series import resample_xy, window_table

FEATURE_KIND = "kinematic"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def interpret_kinematic(feat: dict) -> tuple[str, str]:
    """Name a discovered centroid from its motion. Called only after clustering."""
    scored: list[tuple[int, str, str]] = []
    offset = abs(feat["obj_rx"])
    lift = feat["lift"]
    contact = feat["contact"]
    speed = feat["obj_speed"]
    vx = feat["obj_vx"]
    vy = feat["obj_vy"]
    dist_go = feat["dist_go"]
    if 0.09 < offset < 0.30 and lift < 0.07 and speed < 0.08:
        scored.append((0, "Outside fixture", "part resting outside the fixture"))
    if abs(vx) > 0.08 and lift < 0.14 and 0.05 < offset < 0.30 and abs(vx) > abs(vy) * 0.6:
        scored.append((1, "Slides sideways", "part sliding sideways off the target"))
    if offset < 0.08 and lift < 0.09 and contact > 0.4 and speed >= 0.05:
        scored.append((2, "Placing part", "part settling into the fixture"))
    if offset < 0.06 and lift < 0.06 and speed < 0.08:
        scored.append((3, "Seated in fixture", "part seated in the fixture"))
    if offset < 0.08 and lift >= 0.18 and speed < 0.12:
        scored.append((3, "Hovering at target", "part hovering over the fixture"))
    if vy > 0.08 and lift > 0.05 and offset < 0.12:
        scored.append((4, "Descending", "part descending toward the fixture"))
    if lift > 0.15 and abs(vx) > 0.08:
        scored.append((5, "Carrying part", "part raised and moving across the cell"))
    elif vy < -0.08 and lift > 0.05:
        scored.append((6, "Lifting part", "part lifting off the surface"))
    elif lift > 0.15:
        scored.append((7, "Part raised", "part raised off the surface"))
    if contact > 0.65 and speed < 0.06 and lift < 0.06 and feat["obj_rx"] < -0.2:
        scored.append((8, "Grasping part", "gripper closing on the part"))
    if contact < 0.35 and dist_go > 0.15 and offset < 0.08 and lift < 0.06:
        scored.append((9, "Gripper withdrawn", "gripper withdrawn, part left in place"))
    if feat["obj_rx"] < -0.25 and lift < 0.08 and speed < 0.08:
        scored.append((10, "At pickup", "part waiting at the pickup pose"))
    if dist_go > 0.15 and contact < 0.4 and feat["obj_rx"] < -0.2:
        scored.append((11, "Approaching", "gripper approaching the part"))
    if not scored:
        scored.append((20, "Motion pattern", "unclassified motion pattern"))
    scored.sort(key=lambda item: item[0])
    phrases = []
    for _, _, phrase in scored[:3]:
        if phrase not in phrases:
            phrases.append(phrase)
    return scored[0][1], " · ".join(phrases)


PRE_LIFT_NAMES = [("At pickup", "part waiting at the pickup pose"), ("Grasping part", "gripper closing on the part")]
OFF_TARGET = {"Outside fixture", "Slides sideways", "Motion pattern"}


def _name_pre_lift(names: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """States that come before the part first rises cannot be 'outside the fixture' yet.

    Thresholds in interpret_kinematic assume the synthetic layout; in other cells the
    pickup pose can sit closer to the fixture. Uses time order only, never boundaries.
    """
    lifted = next((i for i, (short, _) in enumerate(names) if short in {"Lifting part", "Carrying part", "Part raised"}), None)
    if lifted is None or lifted == 0:
        return names
    out = list(names)
    pre = list(range(lifted))
    labels = PRE_LIFT_NAMES[-len(pre):] if len(pre) <= len(PRE_LIFT_NAMES) else (
        [PRE_LIFT_NAMES[0]] * (len(pre) - len(PRE_LIFT_NAMES) + 1) + PRE_LIFT_NAMES[1:]
    )
    for i, label in zip(pre, labels):
        if out[i][0] in OFF_TARGET:
            out[i] = (label[0], label[1] + " · " + out[i][1])
    return out


def interpret_flow(feat: dict) -> tuple[str, str]:
    mag = feat["mag"]
    if mag < 0.15:
        return "Scene still", "little motion in the cell"
    if abs(feat.get("flow_vx", 0.0)) > abs(feat.get("flow_vy", 0.0)):
        return "Lateral motion", "motion mostly across the frame"
    return "Vertical motion", "motion mostly toward or away from the camera"


def failure_signature(series: dict) -> np.ndarray:
    time = series["time"]
    duration = float(time[-1]) if len(time) else 1.0
    frac = time / max(duration, 1e-6)
    late = frac >= 0.60
    if late.sum() < 3:
        late = np.ones(len(time), dtype=bool)
    if series.get("feature_kind", np.array(["kinematic"]))[0] == "flow":
        mag = np.median(series["mag"][late])
        still = float(np.mean(series["mag"][late] < 0.2))
        return np.array([mag, still, np.median(series["cx"][late]), np.median(series["cy"][late])])
    dist = float(np.median(series["dist_ob"][late]))
    off = float(np.median(np.abs(series["obj_rx"][late])))
    lift = float(np.median(series["lift"][late]))
    lateral = float(np.percentile(np.abs(series["obj_vx"][late]), 90))
    # The miss lands beside the fixture. Normals end centered on it.
    outside = 1.0 if off > 0.09 and lift < 0.08 else 0.0
    # Saturate once the part is clearly off target: a miss to either side is the same failure.
    off, dist = min(off, 0.135), min(dist, 0.135)
    return np.array([off / 0.03, dist / 0.05, outside * 3.0, lateral / 0.05, (1.0 if lift < 0.08 else 0.0)])


class WorldModel:
    def __init__(self, payload: dict | None = None):
        self.payload = payload or {}

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "WorldModel":
        if not path.exists():
            return cls({})
        return cls(json.loads(path.read_text()))

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_jsonable(self.payload), indent=2))
        tmp.replace(path)

    @property
    def ready(self) -> bool:
        return bool(self.payload.get("nodes"))

    @property
    def version(self) -> int:
        return int(self.payload.get("version", 0))

    def public(self) -> dict:
        keys = (
            "version",
            "dataset",
            "feature_kind",
            "clusterer",
            "n_reference",
            "threshold",
            "nodes",
            "edges",
            "terminal",
            "semantic_rules",
            "fallback",
            "cosmos",
            "adapters",
            "prototypes_public",
        )
        out = {key: self.payload.get(key) for key in keys}
        out["prototypes"] = self.payload.get("prototypes_public", [])
        return out

    def fit(self, episodes: list[dict], series_of: dict[str, dict], feature_kind: str) -> None:
        train = [ep for ep in episodes if ep["split"] == "train"]
        if len(train) < 3:
            raise RuntimeError("need at least 3 reference runs to learn a process")
        windows = []
        owners = []
        times = []
        for ep in train:
            matrix, when = window_table(series_of[ep["id"]], feature_kind, WINDOW, STRIDE)
            windows.append(matrix)
            owners.extend([ep["id"]] * len(matrix))
            times.append(when)
        stacked = np.vstack(windows)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(stacked)
        scale = scaler.scale_.copy()
        scale[scale < 1e-6] = 1.0
        scaler.scale_ = scale
        scaled = scaler.transform(stacked)

        clusterer_name, labels, centers = _cluster(scaled)
        groups: dict[str, list[int]] = {}
        cursor = 0
        per_episode_labels = {}
        per_episode_times = {}
        for ep, when in zip(train, times):
            count = len(when)
            per_episode_labels[ep["id"]] = labels[cursor : cursor + count]
            per_episode_times[ep["id"]] = when
            cursor += count
            groups[ep["id"]] = per_episode_labels[ep["id"]]

        # Order clusters by mean time and name them from centroids, after discovery.
        raw_centers = []
        mean_times = []
        members = []
        for cluster_id in range(centers.shape[0]):
            mask = labels == cluster_id
            raw_centers.append(stacked[mask].mean(axis=0) if mask.any() else np.zeros(stacked.shape[1]))
            member_times = np.concatenate(
                [per_episode_times[ep["id"]][per_episode_labels[ep["id"]] == cluster_id] for ep in train]
            )
            mean_times.append(float(member_times.mean()) if len(member_times) else 0.0)
            members.append(int(mask.sum()))
        order = np.argsort(mean_times)
        id_map = {int(old): f"s{new}" for new, old in enumerate(order)}

        names = {}
        full_names = {}
        nodes = []
        interpret = interpret_kinematic if feature_kind == "kinematic" else interpret_flow
        feature_names = _feature_names(feature_kind, stacked.shape[1])
        used_short = set()
        interpreted = []
        for old in order:
            feat = {name: float(raw_centers[old][i]) for i, name in enumerate(feature_names)}
            interpreted.append(interpret(feat))
        if feature_kind == "kinematic":
            interpreted = _name_pre_lift(interpreted)
        for new_index, old in enumerate(order):
            short, full = interpreted[new_index]
            if short in used_short:
                short = f"{short} {new_index + 1}"
            used_short.add(short)
            sid = f"s{new_index}"
            names[sid] = short
            full_names[sid] = full
            nodes.append(
                {
                    "id": sid,
                    "name": short,
                    "full_name": full,
                    "count": members[old],
                    "mean_t": mean_times[old],
                    "kind": "nominal",
                }
            )

        sequences = {}
        segments = {}
        edge_counts: dict[tuple[str, str], list[float]] = {}
        edge_episodes: dict[tuple[str, str], set[str]] = {}
        dwell: dict[str, list[float]] = {}
        for ep in train:
            mapped = np.array([id_map[int(x)] for x in per_episode_labels[ep["id"]]])
            segs = _collapse(mapped, per_episode_times[ep["id"]])
            segments[ep["id"]] = segs
            sequences[ep["id"]] = [seg["state"] for seg in segs]
            for seg in segs:
                dwell.setdefault(seg["state"], []).append(seg["end"] - seg["start"])
            for left, right in zip(segs, segs[1:]):
                key = (left["state"], right["state"])
                edge_counts.setdefault(key, []).append(right["start"] - left["start"])
                edge_episodes.setdefault(key, set()).add(ep["id"])

        out_total: dict[str, int] = {}
        for (src, _dst), eps in edge_episodes.items():
            out_total[src] = out_total.get(src, 0) + len(eps)
        edges = []
        for (src, dst), durations in edge_counts.items():
            k = len(edge_episodes[(src, dst)])
            edges.append(
                {
                    "src": src,
                    "dst": dst,
                    "count": k,
                    "episodes": k,
                    "p": k / max(out_total.get(src, 1), 1),
                    "mean_duration": float(np.mean(durations)),
                    "kind": "nominal",
                }
            )
        for node in nodes:
            samples = dwell.get(node["id"], [0.0])
            node["mean_dwell"] = float(np.mean(samples))

        traj = []
        for ep in train:
            traj.append(_trajectory(series_of[ep["id"]], feature_kind))
        traj_arr = np.stack(traj)
        traj_mean = traj_arr.mean(axis=0)
        traj_std = np.maximum(traj_arr.std(axis=0), 1e-3)

        dists = _distances(scaled, centers)
        # Just outside the reference cloud, so a train window is never "novel"
        # and a displaced part is.
        novel_radius = float(np.max(dists) * 1.08 + 1e-3)
        raw_mean = {}
        raw_std = {}
        for old, sid in id_map.items():
            mask = labels == old
            raw_mean[sid] = stacked[mask].mean(axis=0)
            raw_std[sid] = np.maximum(stacked[mask].std(axis=0), 0.008)
        feat_zs = []
        for row, lab in zip(stacked, labels):
            sid = id_map[int(lab)]
            feat_zs.append(float(np.max(np.abs(row - raw_mean[sid]) / raw_std[sid])))
        feat_z_limit = float(max(feat_zs) * 1.35 + 0.25)

        terminal = nodes[-1]["id"] if nodes else "s0"
        payload = {
            "version": 1,
            "feature_kind": feature_kind,
            "clusterer": clusterer_name,
            "n_reference": len(train),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "centers": centers.tolist(),
            "id_map": {str(k): v for k, v in id_map.items()},
            "names": names,
            "full_names": full_names,
            "nodes": nodes,
            "edges": edges,
            "terminal": terminal,
            "novel_radius": novel_radius,
            "feat_z_limit": feat_z_limit,
            "cluster_raw_mean": {sid: value.tolist() for sid, value in raw_mean.items()},
            "cluster_raw_std": {sid: value.tolist() for sid, value in raw_std.items()},
            "traj_mean": traj_mean.tolist(),
            "traj_std": traj_std.tolist(),
            "sequences": sequences,
            "transcripts": {},
            "episodes": [
                {
                    "id": ep["id"],
                    "role": ep["role"],
                    "split": ep["split"],
                    "kind": ep["kind"],
                    "file": ep["file"],
                    "video": ep["video"],
                    "listed": ep.get("listed", True),
                }
                for ep in episodes
            ],
            "prototypes": [],
            "prototypes_public": [],
            "semantic_rules": [],
            "dataset": episodes[0].get("dataset", feature_kind),
        }
        self.payload = payload
        # Calibrate the decision threshold on the reference runs themselves.
        scores = []
        for ep in train:
            detail = self.analyze_series(ep["id"], series_of[ep["id"]], use_threshold=False)
            scores.append(detail["score"])
        mu = float(np.mean(scores))
        sd = float(np.std(scores))
        self.payload["threshold"] = float(np.clip(max(scores) + 0.15, mu + 4 * sd, 0.72))
        self.payload["train_score_max"] = float(max(scores))
        self.payload["train_score_mean"] = mu
        transcripts = {}
        for ep in train:
            detail = self.analyze_series(ep["id"], series_of[ep["id"]])
            transcripts[ep["id"]] = _transcript(detail["events"])
        self.payload["transcripts"] = transcripts

    def analyze_series(self, episode_id: str, series: dict, use_threshold: bool = True) -> dict:
        kind = self.payload["feature_kind"]
        matrix, when = window_table(series, kind, WINDOW, STRIDE)
        mean = np.array(self.payload["scaler_mean"])
        scale = np.array(self.payload["scaler_scale"])
        scaled = (matrix - mean) / scale
        centers = np.array(self.payload["centers"])
        dists = _distances(scaled, centers)
        nearest = np.argmin(np.linalg.norm(scaled[:, None, :] - centers[None, :, :], axis=2), axis=1)
        radius = float(self.payload["novel_radius"])
        feat_z_limit = float(self.payload.get("feat_z_limit", 1e9))
        raw_mean = {sid: np.array(value) for sid, value in self.payload.get("cluster_raw_mean", {}).items()}
        raw_std = {sid: np.array(value) for sid, value in self.payload.get("cluster_raw_std", {}).items()}
        id_map = {int(k): v for k, v in self.payload["id_map"].items()}
        states = []
        for index, cluster_id in enumerate(nearest):
            sid = id_map[int(cluster_id)]
            feat_z = 0.0
            if sid in raw_mean:
                feat_z = float(np.max(np.abs(matrix[index] - raw_mean[sid]) / raw_std[sid]))
            if dists[index] > radius or feat_z > feat_z_limit:
                states.append("novel")
            else:
                states.append(sid)
        segments = _collapse(np.array(states), when)
        events = _events_from_segments(segments, series, kind, self.payload["full_names"])
        traj = _trajectory(series, kind)
        mean_t = np.array(self.payload["traj_mean"])
        std_t = np.array(self.payload["traj_std"])
        z = np.sqrt((((traj - mean_t) / std_t) ** 2).sum(axis=1))
        z = np.convolve(z, np.ones(3) / 3, mode="same")
        div_idx = None
        for i in range(max(4, int(0.15 * len(z))), len(z) - 1):
            if z[i] > 3.0 and z[i + 1] > 3.0:
                div_idx = i
                break
        duration = float(series["time"][-1]) if len(series["time"]) else 0.0
        div_time = None if div_idx is None else duration * div_idx / max(len(z) - 1, 1)
        peak = float(np.percentile(z[max(4, int(0.15 * len(z))) :], 90))
        traj_score = float(np.clip((peak - 1.8) / 3.2, 0.0, 1.0))

        surprise = 0.0
        worst = None
        for left, right in zip(segments, segments[1:]):
            probability = self._nominal_p(left["state"], right["state"])
            gap = 1.0 - probability
            if gap >= surprise:
                surprise = gap
                worst = (left, right, probability)
        if any(seg["state"] == "novel" for seg in segments):
            surprise = 1.0
        trans_score = float(surprise)
        embed_score = 1.0 if np.any(dists > radius) else float(np.clip(np.max(dists) / radius, 0.0, 1.0) * 0.4)
        score = float(0.50 * traj_score + 0.35 * trans_score + 0.15 * embed_score)

        last_good = segments[-2]["state"] if len(segments) >= 2 else (segments[0]["state"] if segments else "s0")
        observed = segments[-1]
        novel_segs = [seg for seg in segments if seg["state"] == "novel"]
        if novel_segs:
            observed = novel_segs[0]
            prior = [seg for seg in segments if seg["end"] <= observed["start"] + 1e-3 and seg["state"] != "novel"]
            if prior:
                last_good = prior[-1]["state"]
            div_time = observed["start"] if div_time is None else min(div_time, observed["start"])
        elif worst is not None and worst[2] < 0.5:
            last_good = worst[0]["state"] if worst[0]["state"] != "novel" else last_good
            observed = worst[1]
        elif div_time is not None:
            for seg in segments:
                if seg["start"] <= div_time <= seg["end"] + 1e-6:
                    observed = seg
                    break
            for seg in segments:
                if seg["end"] <= div_time and seg["state"] != "novel":
                    last_good = seg["state"]

        expected_id = self._argmax_next(last_good if last_good != "novel" else self.payload["nodes"][0]["id"])
        expected_name = self._name(expected_id) if expected_id else "end of process"
        observed_name = self._name(observed["state"]) if observed["state"] != "novel" else observed.get("caption", "unseen pattern")
        if observed["state"] == "novel":
            observed_name = observed.get("caption", "unseen pattern")
        nominal_p = self._nominal_p(last_good, expected_id) if expected_id else 0.0
        observed_p = 0.0 if observed["state"] == "novel" else self._nominal_p(last_good, observed["state"])
        support_k = 0 if observed["state"] == "novel" or observed_p == 0 else self._edge_episodes(last_good, observed["state"])
        expected_k = self._edge_episodes(last_good, expected_id) if expected_id else 0
        n_ref = int(self.payload["n_reference"])

        signature = failure_signature(series)
        matched = self._match_prototype(signature)
        threshold = float(self.payload.get("threshold", 0.45))
        if not use_threshold:
            status = "reference"
        elif matched is not None and score >= threshold * 0.45:
            status = "known_failure"
        elif score >= threshold:
            status = "novel"
        else:
            status = "normal"

        expected_path = self._greedy(last_good if last_good != "novel" else self.payload["nodes"][0]["id"])
        if status == "known_failure" and matched is not None:
            recovery_ids = [matched["id"]]
            target = matched.get("recovery_target")
            if target:
                recovery_ids.extend(self._greedy(target))
        else:
            recovery_ids = expected_path
        names = self.payload["names"]
        why = (
            f"Across {n_ref} reference runs the process moves from "
            f"“{self._name(last_good)}” to “{expected_name}” on {expected_k}/{n_ref} runs "
            f"(P={nominal_p:.2f}). "
            f"At {div_time:.2f}s this run instead shows “{observed_name}”. "
            f"That transition appeared in {support_k}/{n_ref} reference runs. "
            f"The path leaves the learned envelope there "
            f"(trajectory score {traj_score:.2f}, nearest-state distance {float(np.max(dists)):.2f} "
            f"vs radius {radius:.2f})."
            if div_time is not None
            else (
                f"The run stays inside the reference envelope "
                f"(score {score:.2f}, threshold {threshold:.2f}). "
                f"The usual next state after “{self._name(last_good)}” is “{expected_name}”."
            )
        )
        return {
            "episode_id": episode_id,
            "status": status,
            "score": round(score, 4),
            "threshold": threshold,
            "components": {
                "trajectory": round(traj_score, 4),
                "transition": round(trans_score, 4),
                "embedding": round(float(np.max(dists)), 4),
                "embedding_radius": round(radius, 4),
                "embedding_score": round(embed_score, 4),
            },
            "first_divergence_s": None if div_time is None else round(float(div_time), 3),
            "expected": {
                "from": self._name(last_good),
                "to": expected_name,
                "p": round(float(nominal_p), 3),
                "from_id": last_good,
                "to_id": expected_id,
            },
            "observed": {
                "from": self._name(last_good),
                "to": observed_name,
                "p": round(float(observed_p), 3),
                "to_id": observed["state"],
            },
            "support": {
                "k": int(support_k),
                "n": n_ref,
                "text": f"appeared in {support_k}/{n_ref} reference runs",
                "expected_k": int(expected_k),
            },
            "why": why,
            "expected_path": [{"id": sid, "name": names.get(sid, sid)} for sid in expected_path],
            "recovery_path": [
                {"id": sid, "name": names.get(sid, self._proto_name(sid))} for sid in recovery_ids
            ],
            "sequence": [
                {"state": seg["state"], "name": self._name(seg["state"]) if seg["state"] != "novel" else seg.get("caption", "unseen pattern"), "t": seg["start"]}
                for seg in segments
            ],
            "events": events,
            "z": [round(float(v), 3) for v in z[::2]],
            "known_class": None if matched is None or status != "known_failure" else matched["label"],
            "known_distance": None if matched is None else round(float(matched["distance"]), 3),
            "video_generation": {
                "provider": "nvidia-cosmos-predict",
                "status": "stub",
                "prompt": "Re-grasp the part and follow the nominal path: "
                + " → ".join(names.get(sid, sid) for sid in expected_path),
                "note": "Recovery video generation is stubbed. A Cosmos Predict endpoint would synthesize this clip.",
            },
            "signature": signature.tolist(),
            "version": self.version,
        }

    def remember(self, episode_id: str, series: dict, label: str, analysis: dict) -> dict:
        signature = np.array(analysis["signature"], dtype=np.float64)
        # Radius is set from reference runs so a later similar miss matches and a normal run does not.
        ref_ids = [ep["id"] for ep in self.payload["episodes"] if ep["split"] == "train"]
        distances = []
        for ref_id in ref_ids:
            ref_series = _load_cached_series(ref_id, self.payload["feature_kind"])
            if ref_series is None:
                continue
            distances.append(float(np.linalg.norm(signature - failure_signature(ref_series))))
        nearest = min(distances) if distances else 1.0
        radius = max(0.35, nearest * 0.55)
        from_id = analysis["expected"]["from_id"]
        target = analysis["expected"]["to_id"] or self.payload["terminal"]
        slug = "".join(ch if ch.isalnum() else "_" for ch in label.lower()).strip("_") or "failure"
        node_id = f"fail_{slug}"
        existing = next((p for p in self.payload["prototypes"] if p["label"] == label), None)
        if existing:
            n = existing["support"]
            existing["vector"] = ((np.array(existing["vector"]) * n + signature) / (n + 1)).tolist()
            existing["support"] = n + 1
            existing["radius"] = radius
            existing["source_episodes"].append(episode_id)
        else:
            self.payload["prototypes"].append(
                {
                    "id": node_id,
                    "label": label,
                    "vector": signature.tolist(),
                    "radius": radius,
                    "support": 1,
                    "source_episodes": [episode_id],
                    "from_state": from_id,
                    "recovery_target": target,
                }
            )
            self.payload["nodes"].append(
                {
                    "id": node_id,
                    "name": label.replace("_", " ").title(),
                    "full_name": label.replace("_", " "),
                    "count": 1,
                    "mean_t": analysis.get("first_divergence_s") or 0,
                    "kind": "failure",
                    "mean_dwell": 0.8,
                }
            )
            self.payload["names"][node_id] = label.replace("_", " ").title()
            self.payload["edges"].append(
                {
                    "src": from_id,
                    "dst": node_id,
                    "count": 1,
                    "episodes": 1,
                    "p": None,
                    "mean_duration": analysis.get("first_divergence_s") or 0,
                    "kind": "failure",
                }
            )
            self.payload["edges"].append(
                {
                    "src": node_id,
                    "dst": target,
                    "count": 1,
                    "episodes": 1,
                    "p": None,
                    "mean_duration": 1.0,
                    "kind": "recovery",
                }
            )
        self.payload["version"] = int(self.payload["version"]) + 1
        rule = (
            f"IF the run leaves “{analysis['expected']['from']}” into “{label.replace('_', ' ')}” "
            f"THEN it is a known failure. Recovery: return to “{analysis['expected']['to']}” "
            f"and follow the nominal path."
        )
        self.payload.setdefault("semantic_rules", []).append(
            {"version": self.payload["version"], "text": rule, "label": label}
        )
        self.payload["prototypes_public"] = [
            {
                "id": p["id"],
                "label": p["label"],
                "support": p["support"],
                "radius": p["radius"],
            }
            for p in self.payload["prototypes"]
        ]
        self.save()
        return {"version": self.version, "label": label, "radius": radius, "rule": rule}

    def analogous(self, sequence: list[str], k: int = 4) -> list[dict]:
        prefix = [sid for sid in sequence if not str(sid).startswith("novel") and not str(sid).startswith("fail_")]
        ranked = []
        for ep_id, other in self.payload.get("sequences", {}).items():
            shared = 0
            for a, b in zip(prefix, other):
                if a != b:
                    break
                shared += 1
            nxt = other[shared] if shared < len(other) else None
            ranked.append(
                {
                    "id": ep_id,
                    "shared_prefix": shared,
                    "next_state": self._name(nxt) if nxt else None,
                    "snippet": self.payload.get("transcripts", {}).get(ep_id, "")[:180],
                }
            )
        ranked.sort(key=lambda item: item["shared_prefix"], reverse=True)
        return ranked[:k]

    def _nominal_p(self, src: str, dst: str | None) -> float:
        if not dst or src == "novel" or dst == "novel":
            return 0.0
        for edge in self.payload.get("edges", []):
            if edge["src"] == src and edge["dst"] == dst and edge.get("kind", "nominal") == "nominal":
                return float(edge["p"] or 0.0)
        return 0.0

    def _edge_episodes(self, src: str, dst: str | None) -> int:
        if not dst:
            return 0
        for edge in self.payload.get("edges", []):
            if edge["src"] == src and edge["dst"] == dst and edge.get("kind", "nominal") == "nominal":
                return int(edge["episodes"])
        return 0

    def _argmax_next(self, src: str) -> str | None:
        best = None
        best_p = -1.0
        for edge in self.payload.get("edges", []):
            if edge["src"] == src and edge.get("kind", "nominal") == "nominal" and float(edge["p"] or 0) > best_p:
                best = edge["dst"]
                best_p = float(edge["p"] or 0)
        return best

    def _greedy(self, start: str, limit: int = 8) -> list[str]:
        path = [start]
        seen = {start}
        current = start
        for _ in range(limit):
            nxt = self._argmax_next(current)
            if not nxt or nxt in seen:
                break
            path.append(nxt)
            seen.add(nxt)
            current = nxt
            if current == self.payload.get("terminal"):
                break
        return path

    def _name(self, sid: str | None) -> str:
        if sid is None:
            return "unknown"
        if sid == "novel":
            return "unseen pattern"
        return self.payload.get("names", {}).get(sid, sid)

    def _proto_name(self, sid: str) -> str:
        for proto in self.payload.get("prototypes", []):
            if proto["id"] == sid:
                return proto["label"].replace("_", " ").title()
        return self._name(sid)

    def _match_prototype(self, signature: np.ndarray) -> dict | None:
        best = None
        for proto in self.payload.get("prototypes", []):
            distance = float(np.linalg.norm(signature - np.array(proto["vector"])))
            if distance <= float(proto["radius"]) and (best is None or distance < best["distance"]):
                best = {**proto, "distance": distance}
        return best


def _cluster(scaled: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(8, len(scaled) // 30))
        labels = clusterer.fit_predict(scaled)
        n_clusters = len(set(labels) - {-1})
        noise = float(np.mean(labels == -1))
        if 3 <= n_clusters <= 8 and noise < 0.08:
            centers = np.stack([scaled[labels == i].mean(axis=0) for i in range(n_clusters)])
            return "hdbscan", labels.astype(int), centers
    except Exception:
        pass
    tried = []
    for k in range(4, 8):
        if len(scaled) <= k:
            continue
        model = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = model.fit_predict(scaled)
        score = float(silhouette_score(scaled, labels))
        tried.append((score, k, labels.astype(int), model.cluster_centers_.copy()))
    if not tried:
        raise RuntimeError("not enough windows to cluster")
    best_score = max(item[0] for item in tried)
    chosen = max((item for item in tried if item[0] >= best_score - 0.04), key=lambda item: item[1])
    return "kmeans", chosen[2], chosen[3]


def _distances(scaled: np.ndarray, centers: np.ndarray) -> np.ndarray:
    deltas = scaled[:, None, :] - centers[None, :, :]
    return np.linalg.norm(deltas, axis=2).min(axis=1)


def _collapse(states: np.ndarray, times: np.ndarray) -> list[dict]:
    segments = []
    for state, when in zip(states, times):
        state = str(state)
        if not segments or segments[-1]["state"] != state:
            segments.append({"state": state, "start": float(when), "end": float(when), "n": 1})
        else:
            segments[-1]["end"] = float(when)
            segments[-1]["n"] += 1
    if len(segments) <= 1:
        return segments
        changed = True
        while changed and len(segments) > 1:
            changed = False
            merged = []
            for seg in segments:
                duration = seg["end"] - seg["start"]
                novel_boundary = seg["state"] == "novel" or (merged and merged[-1]["state"] == "novel")
                if merged and not novel_boundary and (seg["n"] < 2 or duration < 0.28):
                    merged[-1]["end"] = seg["end"]
                    merged[-1]["n"] += seg["n"]
                    changed = True
                else:
                    merged.append(seg)
            segments = merged
    # Fold a tiny leading fragment into the following state.
    if len(segments) > 1 and segments[0]["n"] < 2:
        segments[1]["start"] = segments[0]["start"]
        segments = segments[1:]
    collapsed = []
    for seg in segments:
        if collapsed and collapsed[-1]["state"] == seg["state"]:
            collapsed[-1]["end"] = seg["end"]
            collapsed[-1]["n"] += seg["n"]
        else:
            collapsed.append(seg)
    return collapsed


def _events_from_segments(segments: list[dict], series: dict, kind: str, full_names: dict) -> list[dict]:
    events = []
    previous = None
    for seg in segments:
        caption = full_names.get(seg["state"], seg["state"])
        if seg["state"] == "novel":
            caption = _caption_window(series, kind, (seg["start"] + seg["end"]) / 2)
            seg["caption"] = caption
        actors = ["gripper", "part", "fixture"] if kind == "kinematic" else ["moving region"]
        relations = _relations_at(series, kind, seg["start"])
        events.append(
            {
                "time": round(seg["start"], 3),
                "end_time": round(seg["end"], 3),
                "actors": actors,
                "action": caption,
                "relations": relations,
                "state_change": {"from": previous, "to": seg["state"], "novel": seg["state"] == "novel"},
                "source": "kinematic" if kind == "kinematic" else "optical-flow",
            }
        )
        previous = seg["state"]
    return events


def _caption_window(series: dict, kind: str, when: float) -> str:
    feat = _feat_at(series, kind, when)
    if kind == "kinematic":
        short, _full = interpret_kinematic(feat)
        return short
    short, _full = interpret_flow(feat)
    return short


def _feat_at(series: dict, kind: str, when: float) -> dict:
    time = series["time"]
    index = int(np.argmin(np.abs(time - when)))
    if kind == "kinematic":
        return {name: float(series[name][index]) for name in (
            "obj_rx", "dist_ob", "lift", "contact", "obj_speed", "obj_vx", "obj_vy", "dist_go"
        )}
    return {name: float(series[name][index]) for name in ("mag", "flow_vx", "flow_vy", "cx", "cy")}


def _relations_at(series: dict, kind: str, when: float) -> dict:
    feat = _feat_at(series, kind, when)
    return {key: round(val, 4) for key, val in feat.items()}


def _trajectory(series: dict, kind: str) -> np.ndarray:
    if kind == "kinematic":
        return resample_xy(series["time"], series["obj_rx"], series["obj_ry"], TRAJ_SAMPLES)
    return resample_xy(series["time"], series["cx"], series["cy"], TRAJ_SAMPLES)


def _feature_names(kind: str, width: int) -> list[str]:
    if kind == "kinematic":
        from worldstate.series import KINEMATIC_FEATURES

        return KINEMATIC_FEATURES
    from worldstate.series import FLOW_FEATURES

    names = list(FLOW_FEATURES) + [f"d_{name}" for name in FLOW_FEATURES]
    return names[:width]


def _transcript(events: list[dict]) -> str:
    return " ".join(f"At {event['time']:.2f}s {event['action']}" for event in events)


def _load_cached_series(episode_id: str, kind: str) -> dict | None:
    from worldstate.config import TRACKS_DIR

    path = TRACKS_DIR / f"{episode_id}.json"
    if kind == "kinematic" and path.exists():
        from worldstate.series import series_from_tracks

        return series_from_tracks(json.loads(path.read_text()))
    flow_path = TRACKS_DIR / f"{episode_id}.flow.npz"
    if flow_path.exists():
        blob = np.load(flow_path, allow_pickle=False)
        return {key: blob[key] for key in blob.files}
    return None


def apply_cosmos_names(model: WorldModel, video: Path, cosmos) -> bool:
    if not cosmos.available:
        return False
    named = cosmos.name_clusters(video, model.payload["nodes"])
    if not named:
        return False
    for node in model.payload["nodes"]:
        if node["id"] in named and node.get("kind", "nominal") == "nominal":
            node["name"] = named[node["id"]]
            model.payload["names"][node["id"]] = named[node["id"]]
    return True


def apply_cosmos_events(events: list[dict], video: Path, summary: str, cosmos) -> list[dict]:
    if not cosmos.available:
        return events
    refined = cosmos.refine_events(video, events, summary)
    return refined if refined else events
