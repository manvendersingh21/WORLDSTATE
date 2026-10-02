"""Fit a world model. Real clips are tried first; the scripted cell is the fallback."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from worldstate.adapters import CosmosReasonAdapter, credential_report
from worldstate.config import TRACKS_DIR
from worldstate.dataset import (
    process_episodes,
    process_manifest_path,
    real_episodes,
    real_manifest_path,
    synthetic_episodes,
)
from worldstate.engine import WorldModel, apply_cosmos_events, apply_cosmos_names
from worldstate.memory import MemoryStore
from worldstate.perception import perception_mode, tracks_for
from worldstate.series import series_from_tracks, series_from_video_flow
from worldstate.synthetic import ensure_dataset


def _tracks(episode: dict, mode: str) -> dict:
    return tracks_for(Path(episode["video"]), episode["id"], mode)


def _fit_tracked(episodes: list[dict], mode: str) -> tuple[WorldModel, dict[str, dict], str]:
    series_of = {}
    used = set()
    for ep in episodes:
        tracks = _tracks(ep, mode)
        used.add(tracks.get("perception", mode))
        series_of[ep["id"]] = series_from_tracks(tracks)
        print(ep["id"], tracks.get("perception"), "track quality", round(tracks["quality"], 3))
    model = WorldModel()
    model.fit(episodes, series_of, "kinematic")
    return model, series_of, "+".join(sorted(used))


def _flow(episode: dict) -> dict:
    path = TRACKS_DIR / f"{episode['id']}.flow.npz"
    video = Path(episode["video"])
    if path.exists() and path.stat().st_mtime >= video.stat().st_mtime:
        blob = np.load(path, allow_pickle=False)
        series = {key: blob[key] for key in blob.files}
        series["feature_kind"] = np.array(["flow"])
        return series
    series = series_from_video_flow(video)
    numeric = {key: value for key, value in series.items() if key != "feature_kind"}
    np.savez(path, **numeric)
    return series


def _lcs(a: list[str], b: list[str]) -> float:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1] / max(len(a), len(b), 1)


def _path_agreement(model: WorldModel) -> float:
    seqs = list(model.payload.get("sequences", {}).values())
    if len(seqs) < 2:
        return 0.0
    sims = [_lcs(seqs[i], seqs[j]) for i in range(len(seqs)) for j in range(i + 1, len(seqs))]
    return float(np.median(sims))


def _separation(model: WorldModel, episodes: list[dict], series_of: dict[str, dict]) -> dict:
    rows = []
    for ep in episodes:
        if ep["split"] == "train":
            continue
        detail = model.analyze_series(ep["id"], series_of[ep["id"]])
        normal = ep.get("failure_type", "none") in ("none", None) or ep["kind"] == "normal"
        rows.append({"id": ep["id"], "normal": normal, "score": detail["score"], "status": detail["status"]})
    normals = [row["score"] for row in rows if row["normal"]]
    failures = [row["score"] for row in rows if not row["normal"]]
    auc = None
    if normals and failures and len(set([0] * len(normals) + [1] * len(failures))) == 2:
        labels = [0] * len(normals) + [1] * len(failures)
        scores = normals + failures
        if len(set(scores)) > 1:
            auc = float(roc_auc_score(labels, scores))
    return {
        "heldout_normal_max": None if not normals else float(max(normals)),
        "failure_min": None if not failures else float(min(failures)),
        "auc": auc,
        "rows": rows,
        "path_agreement": round(_path_agreement(model), 3),
    }


def _stable(metrics: dict) -> bool:
    if metrics["auc"] is None or metrics["failure_min"] is None or metrics["heldout_normal_max"] is None:
        return False
    return (
        metrics["auc"] >= 0.85
        and metrics["failure_min"] > metrics["heldout_normal_max"]
        and metrics["path_agreement"] >= 0.75
    )


def _attach_runtime(model: WorldModel, cosmos: CosmosReasonAdapter, perception: str) -> None:
    report = credential_report()
    model.payload["cosmos"] = {
        "available": cosmos.available,
        "model": cosmos.model,
        "base_source": "custom" if cosmos.custom_base else "default",
        "attempted": cosmos.last_status["attempted"],
        "succeeded": cosmos.last_status["succeeded"],
        "failed": cosmos.last_status["failed"],
        "skipped_reason": cosmos.last_status.get("skipped_reason"),
        "last_error": cosmos.last_status.get("last_error"),
    }
    model.payload["adapters"] = {
        "perception": perception,
        "yolo_installed": _yolo_installed(),
        "cosmos_model": cosmos.model,
        "text_embedder": "pending",
        "vector_index": "pending",
        "credentials": report,
    }
    model.payload["prototypes_public"] = []


def _yolo_installed() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except Exception:
        return False


def _maybe_cosmos(model: WorldModel, episodes: list[dict], series_of: dict[str, dict], cosmos: CosmosReasonAdapter) -> None:
    if not cosmos.available:
        return
    train = [ep for ep in episodes if ep["split"] == "train"]
    if not train:
        return
    named = apply_cosmos_names(model, Path(train[0]["video"]), cosmos)
    if not named and cosmos.last_status["failed"]:
        return
    transcripts = {}
    for ep in train:
        detail = model.analyze_series(ep["id"], series_of[ep["id"]])
        summary = " ".join(event["action"] for event in detail["events"])
        events = apply_cosmos_events(detail["events"], Path(ep["video"]), summary, cosmos)
        transcripts[ep["id"]] = " ".join(f"At {event['time']:.2f}s {event['action']}" for event in events)
        if cosmos.last_status["failed"] and cosmos.last_status["succeeded"] == 0:
            transcripts = model.payload.get("transcripts", {})
            break
    else:
        model.payload["transcripts"] = transcripts


def learn_model() -> WorldModel:
    """WORLDSTATE_DATASET: process (default when present), synthetic, or real (Exylos-style clips)."""
    mode = os.environ.get("WORLDSTATE_DATASET", "auto").strip() or "auto"
    cosmos = CosmosReasonAdapter()
    fallback = None
    chosen = None
    series_of: dict[str, dict] = {}
    episodes: list[dict] = []
    perception = "classical-color"
    tracking = "auto"

    if mode in {"auto", "process"} and process_manifest_path() is not None:
        path = process_manifest_path()
        print("learning repeated process from", path)
        episodes = process_episodes(path)
        tracking = perception_mode(episodes[0]["dataset"])
        chosen, series_of, perception = _fit_tracked(episodes, tracking)
        chosen.payload["dataset"] = episodes[0]["dataset"]
    elif mode == "process":
        fallback = {
            "used_synthetic": True,
            "reason": "WORLDSTATE_DATASET=process but no data/process/manifest.json or WORLDSTATE_REAL_MANIFEST was found.",
        }

    if chosen is None and mode in {"auto", "real"} and real_manifest_path() is not None and real_episodes():
        print("assessing real front-camera clips")
        episodes = real_episodes()
        series_of = {ep["id"]: _flow(ep) for ep in episodes}
        candidate = WorldModel()
        candidate.fit(episodes, series_of, "flow")
        metrics = _separation(candidate, episodes, series_of)
        metrics["stable"] = _stable(metrics)
        metrics["n_reference"] = sum(1 for ep in episodes if ep["split"] == "train")
        print(
            "real path agreement",
            metrics["path_agreement"],
            "auc",
            metrics["auc"],
            "failure_min",
            metrics["failure_min"],
            "normal_max",
            metrics["heldout_normal_max"],
        )
        if mode == "real" or metrics["stable"]:
            chosen = candidate
            perception = "optical-flow"
            chosen.payload["dataset"] = "exylos-front"
        else:
            fallback = {
                "used_synthetic": True,
                "reason": (
                    "The Exylos front-camera clips are domain-randomized: table, background, and "
                    "object appearance change enough that success runs do not share one stable "
                    "state path, and held-out failures do not separate cleanly from held-out successes."
                ),
                "real_path_agreement": metrics["path_agreement"],
                "real_auc": metrics["auc"],
                "real_failure_min": metrics["failure_min"],
                "real_normal_max": metrics["heldout_normal_max"],
            }
            (TRACKS_DIR.parent / "real_assessment.json").write_text(json.dumps(metrics, indent=2))

    if chosen is None:
        print("learning synthetic cell")
        ensure_dataset()
        episodes = synthetic_episodes()
        tracking = perception_mode("synthetic")
        chosen, series_of, perception = _fit_tracked(episodes, tracking)
        chosen.payload["dataset"] = "synthetic"

    _maybe_cosmos(chosen, episodes, series_of, cosmos)
    _attach_runtime(chosen, cosmos, perception)
    chosen.payload["fallback"] = fallback
    chosen.payload["perception_mode"] = tracking
    for node in chosen.payload["nodes"]:
        print(f"state {node['id']} {node['name']} t={node['mean_t']:.2f} n={node['count']}")
    for edge in chosen.payload["edges"]:
        if edge.get("kind") == "nominal":
            print(f"  {edge['src']} -> {edge['dst']} p={edge['p']:.2f} n={edge['episodes']}")
    print("threshold", round(chosen.payload["threshold"], 3), "train max", round(chosen.payload["train_score_max"], 3))
    chosen.save()
    memory = MemoryStore()
    memory.rebuild(chosen.payload.get("transcripts", {}), chosen.payload["episodes"])
    chosen.payload["adapters"]["text_embedder"] = memory.embedder.mode
    chosen.payload["adapters"]["vector_index"] = memory.index.backend
    chosen.payload["adapters"]["vast"] = memory.vast.enabled
    chosen.save()
    return chosen


def main() -> None:
    learn_model()


if __name__ == "__main__":
    main()
